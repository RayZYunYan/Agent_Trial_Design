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
    rationale: Optional[str] = None
    shadow_letter: Optional[str] = None
    committed_choice: bool = False


@dataclass
class GroundedMCQResult:
    letter_choice: Optional[str]
    rationale: Optional[str]
    abstained: bool
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


def _mock_implicit_turn(
    turn_index: int,
    *,
    shadow_choice_enabled: bool = False,
) -> ImplicitAbstainResult:
    """Deterministic mock: ask on odd turns, answer C on even."""
    if turn_index % 2 == 1:
        shadow = "C" if shadow_choice_enabled else None
        return ImplicitAbstainResult(
            abstain=True,
            atomic_question="Can you describe when your symptoms started?",
            letter_choice=None,
            confidence=0.0,
            raw_response="Can you describe when your symptoms started?",
            shadow_letter=shadow,
        )
    return ImplicitAbstainResult(
        abstain=False,
        atomic_question=None,
        letter_choice="C",
        confidence=1.0,
        raw_response="C",
        committed_choice=True,
    )


def _parse_implicit_response(
    response_text: str,
    options_dict: Dict[str, str],
    *,
    rationale_generation: bool,
) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Return (letter, question, rationale, abstain)."""
    if rationale_generation:
        parsed = parsing.parse_rg_response(response_text, options_dict)
        if parsed.abstain:
            return None, parsed.atomic_question, parsed.reason, True
        if parsed.letter_choice:
            return parsed.letter_choice, None, parsed.reason, False
        return None, parsed.atomic_question, parsed.reason, True

    if "?" not in response_text:
        letter = parsing.parse_choice(response_text, options_dict)
        if letter:
            return letter, None, None, False
    atomic = parsing.parse_atomic_question(response_text)
    if atomic:
        return None, atomic, None, True
    return None, None, None, True


def implicit_abstention_decision(
    model: ModelClient,
    *,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    rationale_generation: bool = True,
    self_consistency: int = 1,
    arm_system_injection: str = "",
    global_clinical_block: str = "",
    turn_index: int = 1,
    shadow_choice_enabled: bool = False,
) -> ImplicitAbstainResult:
    if model.provider == "mock":
        return _mock_implicit_turn(turn_index, shadow_choice_enabled=shadow_choice_enabled)

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
    rationales: List[str] = []
    response_texts: Dict[str, str] = {}

    for _ in range(self_consistency):
        response_text = _call_model(model, system, user_prompt)
        if not response_text:
            continue
        response_text = response_text.replace("Confident --> Answer: ", "").replace(
            "Not confident --> Doctor Question: ", ""
        )
        letter, atomic, reason, abstain = _parse_implicit_response(
            response_text,
            options_dict,
            rationale_generation=rationale_generation,
        )
        if letter and not abstain:
            answers.append(letter)
            response_texts[letter] = response_text
            if reason:
                rationales.append(reason)
        elif atomic:
            questions.append(atomic)
            response_texts[atomic] = response_text
            if reason:
                rationales.append(reason)

    if not answers and not questions:
        return ImplicitAbstainResult(
            abstain=True,
            atomic_question=None,
            letter_choice=None,
            confidence=0.0,
            raw_response="",
        )

    conf_score = len(answers) / (len(answers) + len(questions)) if (answers or questions) else 0.0
    rationale = rationales[-1] if rationales else None
    shadow_letter: Optional[str] = None

    if len(answers) > len(questions):
        final_answer = max(set(answers), key=answers.count)
        return ImplicitAbstainResult(
            abstain=False,
            atomic_question=None,
            letter_choice=final_answer,
            confidence=conf_score,
            raw_response=response_texts[final_answer],
            rationale=rationale,
            committed_choice=True,
        )

    import random

    atomic_question = random.choice(questions)
    raw = response_texts[atomic_question]

    if shadow_choice_enabled:
        shadow_letter = _shadow_choice(
            model,
            patient_state,
            inquiry,
            options_dict,
            arm_system_injection,
            global_clinical_block,
        )

    return ImplicitAbstainResult(
        abstain=True,
        atomic_question=atomic_question,
        letter_choice=None,
        confidence=conf_score,
        raw_response=raw,
        rationale=rationale,
        shadow_letter=shadow_letter,
        committed_choice=False,
    )


def _shadow_choice(
    model: ModelClient,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    arm_system_injection: str,
    global_clinical_block: str = "",
) -> Optional[str]:
    """MediQ intermediate evaluation only — not used for final MCQ scoring."""
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


def grounded_mcq_choice(
    model: ModelClient,
    *,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    arm_system_injection: str = "",
    global_clinical_block: str = "",
    rationale_generation: bool = True,
) -> GroundedMCQResult:
    """Single finalize MCQ decision grounded in conversation log (MediQ-aligned scoring)."""
    if model.provider == "mock":
        return GroundedMCQResult(
            letter_choice="C",
            rationale="mock grounded choice",
            abstained=False,
            raw_response="ANSWER: C",
        )

    task = (
        prompts.EXPERT_SYSTEM["grounded_answer_RG"]
        if rationale_generation
        else prompts.EXPERT_SYSTEM["grounded_answer"]
    )
    patient_info = patient_state["initial_info"]
    conv = _conv_log(patient_state["interaction_history"])
    options_text = _options_text(options_dict)
    user_prompt = prompts.EXPERT_SYSTEM["curr_template"].format(
        patient_info,
        conv if conv else "None",
        inquiry,
        options_text,
        task,
    )
    system = _compose_system_prompt(
        arm_system_injection=arm_system_injection,
        global_clinical_block=global_clinical_block,
    )
    response_text = _call_model(model, system, user_prompt, temperature=0.1)
    if rationale_generation:
        parsed = parsing.parse_rg_response(response_text or "", options_dict)
        return GroundedMCQResult(
            letter_choice=parsed.letter_choice,
            rationale=parsed.reason,
            abstained=parsed.abstain or parsed.letter_choice is None,
            raw_response=response_text or "",
        )
    letter = parsing.parse_choice(response_text or "", options_dict)
    abstained = letter is None or "ABSTAIN" in (response_text or "").upper()
    return GroundedMCQResult(
        letter_choice=letter,
        rationale=None,
        abstained=abstained,
        raw_response=response_text or "",
    )


def generate_atomic_question(
    model: ModelClient,
    *,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    arm_system_injection: str = "",
    global_clinical_block: str = "",
    exclude_questions: Optional[List[str]] = None,
    turn_index: int = 1,
) -> str:
    """Fallback question when implicit abstain chose a letter but Stage 1 suppresses answer."""
    from smart_trial.mediq.question_dedupe import format_exclude_block

    if model.provider == "mock":
        from smart_trial.mediq.question_dedupe import FALLBACK_QUESTION_POOL, pick_available_question

        prior = list(exclude_questions or [])
        fake_history = [{"role": "assistant", "content": q} for q in prior]
        picked = pick_available_question(
            FALLBACK_QUESTION_POOL,
            fake_history,
            start_index=max(0, turn_index - 1),
        )
        if picked:
            return picked
        idx = max(0, (turn_index - 1) % len(FALLBACK_QUESTION_POOL))
        return FALLBACK_QUESTION_POOL[idx]
    patient_info = patient_state["initial_info"]
    conv = _conv_log(patient_state["interaction_history"])
    options_text = _options_text(options_dict)
    task = prompts.EXPERT_SYSTEM["atomic_question_improved"]
    if exclude_questions:
        task = task + format_exclude_block(exclude_questions)
    user_prompt = prompts.EXPERT_SYSTEM["curr_template"].format(
        patient_info,
        conv if conv else "None",
        inquiry,
        options_text,
        task,
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
    from smart_trial.mediq.question_dedupe import SYMPTOMS_FALLBACK

    return SYMPTOMS_FALLBACK
