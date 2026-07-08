"""Tests for doctor question deduplication."""
import os

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

from smart_trial.core.doctor_agent import DoctorAgent
from smart_trial.mediq import MediQConfig
from smart_trial.mediq.question_dedupe import (
    CARE_DETAIL_FALLBACK,
    SYMPTOMS_FALLBACK,
    count_normalized_matches,
    exceeds_repeat_limit,
    extract_doctor_questions,
    is_uninformative_patient_reply,
    normalize_question,
    patient_rejected_question,
    pick_fallback_question,
    should_block_question,
)
from smart_trial.models.model_client import ModelClient


def _sample_case():
    return {
        "case_id": "dedupe_test",
        "chief_complaint": "Fever and joint pain.",
        "question": "Which antibiotic?",
        "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
        "atomic_facts": ["Patient complains of fever.", "Culture of joint fluid shows a bacteria."],
    }


def test_normalize_strips_retrieval_query():
    raw = "[RETRIEVAL QUERY: foo]\n\nHave you had any fever?"
    assert normalize_question(raw) == "have you had any fever"


def test_exceeds_repeat_limit_after_two():
    prior = [
        "Could you tell me more about your symptoms?",
        "Could you tell me more about your symptoms?",
    ]
    assert exceeds_repeat_limit(
        "Could you tell me more about your symptoms?",
        prior,
        max_occurrences=2,
    )


def test_extract_doctor_questions_skips_diagnosis():
    hist = [
        {"role": "assistant", "content": "Hello."},
        {"role": "assistant", "content": "Any fever?"},
        {"role": "assistant", "content": "[DIAGNOSIS] Gonorrhea"},
    ]
    qs = extract_doctor_questions(hist)
    assert len(qs) == 1
    assert "fever" in qs[0].lower()


def test_doctor_regenerates_on_third_duplicate():
    model = ModelClient(provider="mock", model_name="mock")
    doctor = DoctorAgent(
        model,
        {"arm_id": "A1a", "stage": 1, "system_prompt_injection": ""},
        case=_sample_case(),
        mediq_config=MediQConfig(enabled=True, max_question_occurrences=2),
    )
    dup = SYMPTOMS_FALLBACK
    doctor.conversation_history = [
        {"role": "assistant", "content": dup},
        {"role": "user", "content": "Fever."},
        {"role": "assistant", "content": dup},
        {"role": "user", "content": "Still fever."},
    ]
    doctor.turn_count = 3
    out = doctor._ensure_unique_doctor_question(dup, "")
    assert normalize_question(out) != normalize_question(dup)
    assert "?" in out


def test_count_normalized_matches():
    prior = ["Any joint pain?", "ANY joint pain??"]
    assert count_normalized_matches("Any joint pain", prior) == 2


def test_uninformative_patient_reply():
    assert is_uninformative_patient_reply("I don't know. Why does that matter?")
    assert not is_uninformative_patient_reply("I have had fever for three days.")


def test_patient_rejection_blocks_repeat_even_once():
    hist = [
        {"role": "assistant", "content": "When did your knee pain start?"},
        {"role": "user", "content": "I don't know. Why does that matter?"},
    ]
    q = "When did your knee pain start?"
    assert patient_rejected_question(q, hist)
    assert should_block_question(q, hist, max_occurrences=2)


def test_fallback_pool_avoids_third_repeat():
    hist = [
        {"role": "assistant", "content": CARE_DETAIL_FALLBACK},
        {"role": "user", "content": "Nothing else."},
        {"role": "assistant", "content": CARE_DETAIL_FALLBACK},
        {"role": "user", "content": "Still nothing."},
    ]
    picked = pick_fallback_question(hist, max_occurrences=2, turn_index=5)
    assert picked is not None
    assert normalize_question(picked) != normalize_question(CARE_DETAIL_FALLBACK)


def test_doctor_skips_question_after_patient_says_unknown():
    model = ModelClient(provider="mock", model_name="mock")
    doctor = DoctorAgent(
        model,
        {"arm_id": "A1a", "stage": 1, "system_prompt_injection": ""},
        case=_sample_case(),
        mediq_config=MediQConfig(enabled=True, max_question_occurrences=2),
    )
    q = "When did your symptoms start?"
    doctor.conversation_history = [
        {"role": "assistant", "content": q},
        {"role": "user", "content": "I don't know."},
    ]
    doctor.turn_count = 2
    out = doctor._ensure_unique_doctor_question(q, "")
    assert normalize_question(out) != normalize_question(q)
