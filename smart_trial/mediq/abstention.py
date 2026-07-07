"""MediQ implicit abstention (BasicExpert) via ModelClient."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from smart_trial.models.model_client import ModelClient
from smart_trial.mediq import parsing, prompts


@dataclass
class ImplicitAbstainResult:
    abstain: bool
    atomic_question: Optional[str]
    letter_choice: Optional[str]
    confidence: float
    raw_response: str


def _options_text(options_dict: Dict[str, str]) -> str:
    return (
        f'A: {options_dict.get("A", "")}, B: {options_dict.get("B", "")}, '
        f'C: {options_dict.get("C", "")}, D: {options_dict.get("D", "")}'
    )


def _conv_log(interaction_history: List[Dict[str, str]]) -> str:
    lines = []
    for qa in interaction_history:
        lines.append(
            f"{prompts.EXPERT_SYSTEM['question_word']}: {qa['question']}\n"
            f"{prompts.EXPERT_SYSTEM['answer_word']}: {qa['answer']}"
        )
    return "\n".join(lines)


def _compose_system_prompt(
    *,
    arm_system_injection: str = "",
    global_clinical_block: str = "",
) -> str:
    """MediQ base + SMART global clinical task + arm strategy (same order as doctor_agent)."""
    system = prompts.EXPERT_SYSTEM["meditron_system_msg"]
    parts: List[str] = []
    if global_clinical_block.strip():
        parts.append(global_clinical_block.strip())
    if arm_system_injection.strip():
        parts.append(arm_system_injection.strip())
    if parts:
        system = f"{system}\n\n" + "\n\n".join(parts)
    return system


def _call_model(
    model: ModelClient,
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
) -> str:
    return model.chat(
        [{"role": "user", "content": user}],
        system_prompt=system,
        temperature=temperature,
    )


def _mock_implicit_turn(turn_index: int) -> ImplicitAbstainResult:
    """Deterministic mock: ask on odd turns, answer C on even."""
    if turn_index % 2 == 1:
        return ImplicitAbstainResult(
            abstain=True,
            atomic_question="Can you describe when your symptoms started?",
            letter_choice="C",
            confidence=0.0,
            raw_response="Can you describe when your symptoms started?",
        )
    return ImplicitAbstainResult(
        abstain=False,
        atomic_question=None,
        letter_choice="C",
        confidence=1.0,
        raw_response="C",
    )


def implicit_abstention_decision(
    model: ModelClient,
    *,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    rationale_generation: bool = False,
    self_consistency: int = 1,
    arm_system_injection: str = "",
    global_clinical_block: str = "",
    turn_index: int = 1,
) -> ImplicitAbstainResult:
    if model.provider == "mock":
        return _mock_implicit_turn(turn_index)

    prompt_key = "implicit_RG" if rationale_generation else "implicit"
    abstain_task = prompts.EXPERT_SYSTEM[prompt_key]
    patient_info = patient_state["initial_info"]
    conv = _conv_log(patient_state["interaction_history"])
    options_text = _options_text(options_dict)
    user_prompt = prompts.EXPERT_SYSTEM["curr_template"].format(
        patient_info,
        conv if conv else "None",
        inquiry,
        options_text,
        abstain_task,
    )
    system = _compose_system_prompt(
        arm_system_injection=arm_system_injection,
        global_clinical_block=global_clinical_block,
    )

    answers: List[str] = []
    questions: List[str] = []
    response_texts: Dict[str, str] = {}

    for _ in range(self_consistency):
        response_text = _call_model(model, system, user_prompt)
        if not response_text:
            continue
        response_text = response_text.replace("Confident --> Answer: ", "").replace(
            "Not confident --> Doctor Question: ", ""
        )
        if "?" not in response_text:
            letter = parsing.parse_choice(response_text, options_dict)
            if letter:
                answers.append(letter)
                response_texts[letter] = response_text
        else:
            atomic = parsing.parse_atomic_question(response_text)
            if atomic:
                questions.append(atomic)
                response_texts[atomic] = response_text

    if not answers and not questions:
        return ImplicitAbstainResult(
            abstain=True,
            atomic_question=None,
            letter_choice=None,
            confidence=0.0,
            raw_response="",
        )

    conf_score = len(answers) / (len(answers) + len(questions))
    if len(answers) > len(questions):
        final_answer = max(set(answers), key=answers.count)
        atomic_question = None
        abstain = False
        raw = response_texts[final_answer]
        letter_choice = final_answer
    else:
        import random

        atomic_question = random.choice(questions)
        abstain = True
        raw = response_texts[atomic_question]
        letter_choice = None

    if letter_choice is None:
        letter_choice = _forced_choice(
            model,
            patient_state,
            inquiry,
            options_dict,
            arm_system_injection,
            global_clinical_block,
        )

    return ImplicitAbstainResult(
        abstain=abstain,
        atomic_question=atomic_question,
        letter_choice=letter_choice,
        confidence=conf_score,
        raw_response=raw,
    )


def _forced_choice(
    model: ModelClient,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    arm_system_injection: str,
    global_clinical_block: str = "",
) -> Optional[str]:
    """Shadow MCQ choice when the model asked a question (MediQ second step)."""
    if model.provider == "mock":
        return "C"
    patient_info = patient_state["initial_info"]
    conv = _conv_log(patient_state["interaction_history"])
    options_text = _options_text(options_dict)
    user_prompt = prompts.EXPERT_SYSTEM["curr_template"].format(
        patient_info,
        conv if conv else "None",
        inquiry,
        options_text,
        prompts.EXPERT_SYSTEM["answer"],
    )
    system = _compose_system_prompt(
        arm_system_injection=arm_system_injection,
        global_clinical_block=global_clinical_block,
    )
    response_text = _call_model(model, system, user_prompt, temperature=0.1)
    return parsing.parse_choice(response_text or "", options_dict)


def generate_atomic_question(
    model: ModelClient,
    *,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    arm_system_injection: str = "",
    global_clinical_block: str = "",
) -> str:
    """Fallback question when implicit abstain chose a letter but Stage 1 suppresses answer."""
    if model.provider == "mock":
        return "Could you tell me more about your main symptom?"
    patient_info = patient_state["initial_info"]
    conv = _conv_log(patient_state["interaction_history"])
    options_text = _options_text(options_dict)
    user_prompt = prompts.EXPERT_SYSTEM["curr_template"].format(
        patient_info,
        conv if conv else "None",
        inquiry,
        options_text,
        prompts.EXPERT_SYSTEM["atomic_question_improved"],
    )
    system = _compose_system_prompt(
        arm_system_injection=arm_system_injection,
        global_clinical_block=global_clinical_block,
    )
    response_text = _call_model(model, system, user_prompt)
    atomic = parsing.parse_atomic_question(response_text or "")
    if atomic:
        return atomic
    if response_text and "?" in response_text:
        return response_text.strip()
    return "Could you tell me more about your symptoms?"
