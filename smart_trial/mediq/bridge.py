"""Build patient state and run one MediQ turn for DoctorAgent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from smart_trial.models.model_client import ModelClient
from smart_trial.mediq.abstention import (
    generate_atomic_question,
    implicit_abstention_decision,
)
from smart_trial.mediq.config import MediQConfig
from smart_trial.mediq.prompts import build_global_clinical_task_block, build_safe_patient_initial_info
from smart_trial.mediq.question_dedupe import extract_doctor_questions


@dataclass
class MediQTurnMeta:
    abstain: bool
    letter_choice: Optional[str]
    mediq_confidence: float
    suppressed_answer: bool = False
    expert_class: str = "BasicExpert"
    shadow_letter: Optional[str] = None
    rationale: Optional[str] = None
    committed_choice: bool = False


@dataclass
class MediQSessionState:
    intermediate_choices: List[str] = field(default_factory=list)
    shadow_choices: List[str] = field(default_factory=list)
    turn_metas: List[MediQTurnMeta] = field(default_factory=list)
    final_letter_choice: Optional[str] = None
    final_rationale: Optional[str] = None


def build_interaction_history(
    conversation_history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Map doctor/patient chat turns to MediQ Q&A pairs (skip opening monologue)."""
    pairs: List[Dict[str, str]] = []
    pending_doctor: Optional[str] = None
    for msg in conversation_history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            if "?" in content:
                pending_doctor = content
        elif role == "user" and pending_doctor:
            pairs.append({"question": pending_doctor, "answer": content})
            pending_doctor = None
    return pairs


def build_patient_state(
    case: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    initial = build_safe_patient_initial_info(case)
    return {
        "initial_info": initial,
        "interaction_history": build_interaction_history(conversation_history),
    }


def normalize_options(case: Dict[str, Any]) -> Dict[str, str]:
    raw = case.get("options") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(k).upper()[:1]: str(v) for k, v in raw.items() if str(k).upper()[:1] in "ABCD"}


def option_text_for_letter(case: Dict[str, Any], letter: Optional[str]) -> str:
    if not letter:
        return ""
    opts = normalize_options(case)
    return opts.get(letter.upper()[:1], "")


def run_basic_expert_turn(
    model: ModelClient,
    *,
    case: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    mediq_config: MediQConfig,
    arm_system_injection: str,
    current_stage: int,
    turn_index: int,
) -> tuple["ImplicitAbstainResult", MediQTurnMeta]:
    from smart_trial.mediq.abstention import ImplicitAbstainResult

    patient_state = build_patient_state(case, conversation_history)
    inquiry = (case.get("question") or "Which option is correct?").strip()
    options = normalize_options(case)
    if not options:
        options = {"A": "", "B": "", "C": "", "D": ""}
    global_block = build_global_clinical_task_block(case)

    result = implicit_abstention_decision(
        model,
        patient_state=patient_state,
        inquiry=inquiry,
        options_dict=options,
        rationale_generation=mediq_config.rationale_generation,
        self_consistency=mediq_config.self_consistency,
        arm_system_injection=arm_system_injection,
        global_clinical_block=global_block,
        turn_index=turn_index,
        shadow_choice_enabled=mediq_config.shadow_choice_enabled,
    )

    suppressed = False
    effective_abstain = result.abstain

    if (
        not result.abstain
        and current_stage == 1
        and mediq_config.stage1_suppress_answer
    ):
        suppressed = True
        effective_abstain = True
        if not result.atomic_question:
            asked = extract_doctor_questions(conversation_history)
            result.atomic_question = generate_atomic_question(
                model,
                patient_state=patient_state,
                inquiry=inquiry,
                options_dict=options,
                arm_system_injection=arm_system_injection,
                global_clinical_block=global_block,
                exclude_questions=asked,
                turn_index=turn_index,
            )

    committed = result.committed_choice and not suppressed
    committed_letter = result.letter_choice if committed else None

    meta = MediQTurnMeta(
        abstain=effective_abstain,
        letter_choice=committed_letter,
        mediq_confidence=result.confidence,
        suppressed_answer=suppressed,
        expert_class=mediq_config.expert_class,
        shadow_letter=result.shadow_letter,
        rationale=result.rationale,
        committed_choice=committed,
    )
    return result, meta
