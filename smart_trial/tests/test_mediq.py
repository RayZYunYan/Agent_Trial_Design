"""Tests for MediQ layer (self-contained under smart_trial)."""
import os

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

from smart_trial.core.doctor_agent import DoctorAgent
from smart_trial.core.judge import StageJudge
from smart_trial.mediq import MediQConfig
from smart_trial.mediq.bridge import build_interaction_history
from smart_trial.models.model_client import ModelClient


def _sample_case():
    return {
        "case_id": "medqa_0000",
        "chief_complaint": "A 21-year-old male with fever and joint pain.",
        "question": "Which antibiotic is most appropriate?",
        "options": {"A": "Gentamicin", "B": "Ciprofloxacin", "C": "Ceftriaxone", "D": "Trimethoprim"},
        "ground_truth_idx": "C",
        "ground_truth_answer": "Ceftriaxone",
        "red_flags": [],
    }


def test_build_interaction_history_pairs():
    history = [
        {"role": "assistant", "content": "Hello, how can I help?"},
        {"role": "assistant", "content": "When did the fever start?"},
        {"role": "user", "content": "Two days ago."},
    ]
    pairs = build_interaction_history(history)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "When did the fever start?"
    assert pairs[0]["answer"] == "Two days ago."


def test_mediq_disabled_uses_legacy_path():
    model = ModelClient(provider="mock", model_name="mock")
    doctor = DoctorAgent(
        model,
        {"arm_id": "A1a", "stage": 1, "system_prompt_injection": ""},
        case=_sample_case(),
        mediq_config=MediQConfig(enabled=False),
    )
    doctor.get_initial_message(_sample_case())
    msg, conf = doctor.respond("I have had fever.", force_conclude=False)
    assert "[MOCK]" in msg
    assert conf is None
    assert doctor.get_intermediate_choices() == []


def test_mediq_enabled_basic_expert_mock():
    model = ModelClient(provider="mock", model_name="mock")
    doctor = DoctorAgent(
        model,
        {"arm_id": "A1a", "stage": 1, "system_prompt_injection": "Ask focused questions."},
        case=_sample_case(),
        mediq_config=MediQConfig(enabled=True, stage1_suppress_answer=True),
    )
    doctor.get_initial_message(_sample_case())
    msg, _ = doctor.respond("Fever for two days.", force_conclude=False)
    assert "?" in msg
    meta = doctor.get_last_mediq_meta()
    assert meta is not None
    assert meta["letter_choice"] == "C"
    assert "C" in doctor.get_intermediate_choices()


def test_mediq_stage2_finalize_mock():
    model = ModelClient(provider="mock", model_name="mock")
    doctor = DoctorAgent(
        model,
        {"arm_id": "A2a", "stage": 2, "system_prompt_injection": ""},
        case=_sample_case(),
        mediq_config=MediQConfig(enabled=True),
    )
    doctor.conversation_history = [
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Hi."},
    ]
    # turn_index=2 → mock chooses answer path
    doctor.turn_count = 1
    msg, conf = doctor.respond("More details.", force_conclude=False)
    assert "[DIAGNOSIS]" in msg
    assert doctor.has_concluded()
    assert doctor.get_final_letter_choice() == "C"
    assert conf is not None


def test_mcq_correct_in_judge():
    judge = StageJudge(ModelClient(provider="mock", model_name="mock"))
    case = _sample_case()
    outcome = judge.evaluate_outcome(
        final_diagnosis="[DIAGNOSIS] Ceftriaxone",
        case=case,
        conversation_history=[],
        R2={"confidence_level": "high", "final_confidence": 0.9},
        mcq_letter_choice="C",
    )
    assert outcome["mcq_correct"] is True

    wrong = judge.evaluate_outcome(
        final_diagnosis="[DIAGNOSIS] x",
        case=case,
        conversation_history=[],
        R2={"confidence_level": "low", "final_confidence": 0.4},
        mcq_letter_choice="A",
    )
    assert wrong["mcq_correct"] is False
