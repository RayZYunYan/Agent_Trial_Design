import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from smart_trial.mediq import MediQConfig, MediQSessionState, option_text_for_letter, run_basic_expert_turn
from smart_trial.mediq.prompts import build_global_clinical_task_block
from smart_trial.models.model_client import ModelClient


def _format_age_for_intro(age: Any) -> str:
    s = str(age).strip()
    if s.isdigit():
        return s
    m = re.search(r"(\d{1,3})", s)
    return m.group(1) if m else s


def _chief_complaint_is_full_narrative(chief: str) -> bool:
    """True when first context line already states age/demographics (avoid repeating in intro)."""
    if not chief or len(chief) < 24:
        return False
    low = chief.lower()
    if "year-old" in low or "year old" in low or re.match(r"^\s*a\s+\d{1,3}\b", low):
        return True
    if re.search(r"^\s*\d{1,3}\s*[- ]?\s*(year|yr)\b", chief, re.I):
        return True
    return False


def load_arm_config(arm_id: str, arms_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML arm config by arm id (e.g. A1a -> stage1_A1a.yaml)."""
    stage_map = {
        "A1a": "stage1_A1a",
        "A1b": "stage1_A1b",
        "A1c": "stage1_A1c",
        "A2a": "stage2_A2a",
        "A2b": "stage2_A2b",
        "A2c": "stage2_A2c",
    }
    filename = stage_map.get(arm_id, arm_id)
    if arms_dir is None:
        arms_dir = Path(__file__).resolve().parent.parent / "config" / "arms"
    filepath = arms_dir / f"{filename}.yaml"
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DoctorAgent:
    """Chat doctor with optional MediQ BasicExpert layer + SMART arm injection."""

    BASE_SYSTEM_PROMPT = """You are an experienced primary care physician conducting a text-based visit with a simulated patient.

Your tasks:
1. Ask focused questions to gather history and symptoms.
2. Reason toward a working diagnosis as information allows.
3. Eventually provide a clear assessment and next-step advice.

Rules:
- Ask exactly one clear question at a time (unless the current stage arm says otherwise).
- Use professional but plain English the patient can understand.
- Do not dump long lectures or multiple unrelated questions in one turn.
- The patient scenario and chief complaint are in English: you MUST speak and write only in English in every message."""

    CONCLUSION_MARKER = "[DIAGNOSIS]"

    FINAL_TURN_INSTRUCTION = (
        "IMPORTANT: This is your final turn of the visit. Do not ask any more "
        "questions. Deliver your final assessment now, following the "
        "'Delivering Your Final Assessment' format: put the [DIAGNOSIS] line first, "
        "then your reasoning and management plan."
    )

    BASELINE_FINAL_TURN_INSTRUCTION = (
        "IMPORTANT: This is your final turn of the visit. Do not ask any more "
        "questions. Deliver your final assessment now. Include a line with "
        "[DIAGNOSIS] followed by your conclusion and next-step advice."
    )

    BASELINE_ARM_CONFIG: Dict[str, Any] = {
        "arm_id": "baseline",
        "stage": 0,
        "name": "Baseline",
        "system_prompt_injection": "",
    }

    def __init__(
        self,
        model_client: ModelClient,
        initial_arm_config: Dict[str, Any],
        *,
        case: Optional[Dict[str, Any]] = None,
        mediq_config: Optional[MediQConfig] = None,
    ):
        self.model = model_client
        self.current_arm = initial_arm_config
        self.case = case
        self.mediq_config = mediq_config or MediQConfig()
        self.conversation_history: List[Dict[str, str]] = []
        self.current_stage = int(initial_arm_config.get("stage", 1))
        self.turn_count = 0
        self._final_diagnosis: Optional[str] = None
        self._has_concluded = False
        self.mediq_state = MediQSessionState()
        self._last_mediq_meta: Optional[Dict[str, Any]] = None
        self._allow_finalize = True

    @property
    def mediq_enabled(self) -> bool:
        return bool(self.mediq_config.enabled and self.case is not None)

    def switch_arm(self, new_arm_config: Dict[str, Any]) -> None:
        self.current_arm = new_arm_config
        self.current_stage = int(new_arm_config.get("stage", self.current_stage))

    def set_allow_finalize(self, allow: bool) -> None:
        self._allow_finalize = allow

    def respond(
        self,
        patient_message: str,
        force_conclude: bool = False,
        retrieval_context: Optional[str] = None,
        *,
        allow_finalize: Optional[bool] = None,
    ) -> str:
        if allow_finalize is not None:
            self._allow_finalize = allow_finalize
        self.turn_count += 1
        if patient_message:
            self.conversation_history.append({"role": "user", "content": patient_message})

        if self.mediq_enabled:
            response = self._respond_mediq(
                force_conclude=force_conclude,
                retrieval_context=retrieval_context,
            )
        else:
            response = self._respond_legacy(
                force_conclude=force_conclude,
                retrieval_context=retrieval_context,
            )

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _respond_legacy(
        self,
        *,
        force_conclude: bool,
        retrieval_context: Optional[str],
    ) -> str:
        system_prompt = self._build_system_prompt(retrieval_context)
        if force_conclude or (self._allow_finalize and self._should_try_conclude_legacy()):
            final_hint = (
                self.BASELINE_FINAL_TURN_INSTRUCTION
                if self.current_stage == 0
                else self.FINAL_TURN_INSTRUCTION
            )
            system_prompt = f"{system_prompt}\n\n{final_hint}"
        response = self.model.chat(
            messages=self._conversation_for_api(),
            system_prompt=system_prompt,
        )
        self._check_conclusion(response)
        return response

    def _should_try_conclude_legacy(self) -> bool:
        return False

    def _respond_mediq(
        self,
        *,
        force_conclude: bool,
        retrieval_context: Optional[str],
    ) -> str:
        assert self.case is not None
        arm_injection = self.current_arm.get("system_prompt_injection", "") or ""

        if force_conclude:
            letter = self._resolve_letter_for_finalize()
            response = self._generate_final_assessment(
                letter,
                retrieval_context=retrieval_context,
            )
            self._record_mediq_shadow(letter, abstain=False, suppressed=False)
            return response

        _result, meta = run_basic_expert_turn(
            self.model,
            case=self.case,
            conversation_history=self.conversation_history,
            mediq_config=self.mediq_config,
            arm_system_injection=arm_injection,
            current_stage=self.current_stage,
            turn_index=self.turn_count,
        )
        self._last_mediq_meta = {
            "abstain": meta.abstain,
            "letter_choice": meta.letter_choice,
            "mediq_confidence": meta.mediq_confidence,
            "suppressed_answer": meta.suppressed_answer,
            "expert_class": meta.expert_class,
        }
        if meta.letter_choice:
            self.mediq_state.intermediate_choices.append(meta.letter_choice)
        self.mediq_state.turn_metas.append(meta)

        wants_to_finalize = (
            not meta.abstain
            and self.mediq_config.stage2_allow_mcq_finalize
            and self.current_stage in (0, 2)
        )

        if wants_to_finalize and self._allow_finalize:
            letter = meta.letter_choice
            self.mediq_state.final_letter_choice = letter
            response = self._generate_final_assessment(
                letter,
                retrieval_context=retrieval_context,
            )
            return response

        if wants_to_finalize and not self._allow_finalize:
            self._last_mediq_meta["finalize_blocked"] = "low_coverage"
            self._last_mediq_meta["suppressed_answer"] = True
            question = _result.atomic_question or generate_fallback_question()
            return question

        question = _result.atomic_question or "Could you tell me more about your symptoms?"
        return question

    def _resolve_letter_for_finalize(self) -> Optional[str]:
        if self.mediq_state.final_letter_choice:
            return self.mediq_state.final_letter_choice
        if self.mediq_state.intermediate_choices:
            return self.mediq_state.intermediate_choices[-1]
        return None

    def _record_mediq_shadow(
        self,
        letter: Optional[str],
        *,
        abstain: bool,
        suppressed: bool,
    ) -> None:
        if letter:
            self.mediq_state.intermediate_choices.append(letter)
            self.mediq_state.final_letter_choice = letter
        self._last_mediq_meta = {
            "abstain": abstain,
            "letter_choice": letter,
            "mediq_confidence": 1.0 if letter else 0.0,
            "suppressed_answer": suppressed,
            "expert_class": self.mediq_config.expert_class,
            "forced_finalize": True,
        }

    def _generate_final_assessment(
        self,
        letter: Optional[str],
        *,
        retrieval_context: Optional[str] = None,
    ) -> str:
        assert self.case is not None
        option_text = option_text_for_letter(self.case, letter)
        system_prompt = self._build_system_prompt(retrieval_context)
        final_hint = (
            self.BASELINE_FINAL_TURN_INSTRUCTION
            if self.current_stage == 0
            else self.FINAL_TURN_INSTRUCTION
        )
        mcq_hint = ""
        if letter and option_text:
            mcq_hint = (
                f"\n\nYour leading multiple-choice answer is {letter}: {option_text}. "
                f"The [DIAGNOSIS] line must state this diagnosis clearly."
            )
        elif letter:
            mcq_hint = f"\n\nYour leading multiple-choice answer is {letter}."

        system_prompt = f"{system_prompt}\n\n{final_hint}{mcq_hint}"

        if self.model.provider == "mock":
            diag = option_text or "mock diagnosis"
            response = (
                f"[DIAGNOSIS] {diag}\n"
                "Based on your symptoms, this is my working diagnosis. "
                "Please follow up if symptoms worsen."
            )
        else:
            response = self.model.chat(
                messages=self._conversation_for_api(),
                system_prompt=system_prompt,
            )

        self._check_conclusion(response)
        if letter:
            self.mediq_state.final_letter_choice = letter
        return response

    def _conversation_for_api(self) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]

    def _build_system_prompt(self, retrieval_context: Optional[str] = None) -> str:
        parts = [self.BASE_SYSTEM_PROMPT]
        if self.case is not None:
            parts.append(build_global_clinical_task_block(self.case))
        arm_instruction = self.current_arm.get("system_prompt_injection", "") or ""
        if arm_instruction:
            parts.append(arm_instruction)
        prompt = "\n\n".join(parts)
        if retrieval_context:
            prompt += f"\n\n{retrieval_context}"
        return prompt

    def _check_conclusion(self, response: str) -> None:
        if self.CONCLUSION_MARKER in response:
            self._has_concluded = True
            self._final_diagnosis = response

    def get_initial_message(self, case: Dict[str, Any]) -> str:
        chief_complaint = (case.get("chief_complaint") or "what brought you in today").strip()
        if not chief_complaint.endswith((".", "?", "!")):
            chief_run = chief_complaint + "."
        else:
            chief_run = chief_complaint

        if _chief_complaint_is_full_narrative(chief_complaint):
            initial_msg = (
                "Hello, I'm your doctor today. "
                f"{chief_run} "
                "I'd like to ask you a few questions to better understand your condition."
            )
        else:
            age = _format_age_for_intro(case.get("age", "unknown"))
            gender = case.get("gender", "unknown")
            initial_msg = (
                f"Hello, I'm your doctor today. I see you're a {age}-year-old {gender} "
                f"with a concern about {chief_run} "
                "I'd like to ask you a few questions to better understand your condition."
            )
        self.conversation_history.append({"role": "assistant", "content": initial_msg})
        return initial_msg

    def has_concluded(self) -> bool:
        return self._has_concluded

    def get_final_diagnosis(self) -> Optional[str]:
        return self._final_diagnosis

    def get_final_letter_choice(self) -> Optional[str]:
        return self.mediq_state.final_letter_choice

    def get_intermediate_choices(self) -> List[str]:
        return list(self.mediq_state.intermediate_choices)

    def get_last_mediq_meta(self) -> Optional[Dict[str, Any]]:
        return self._last_mediq_meta


def generate_fallback_question() -> str:
    return "Could you tell me more about any tests or treatments you have had for this problem?"
