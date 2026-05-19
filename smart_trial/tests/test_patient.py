import yaml

from smart_trial.core.patient_agent import (
    PatientAgent,
    age_voice_band,
    build_patient_system_prompt,
    parent_age_for_child,
    parse_age_years,
)
from smart_trial.data.loader import load_cases_from_config
from smart_trial.models.model_client import ModelClient


def test_parse_age_years():
    assert parse_age_years("21 years old") == 21
    assert parse_age_years(16) == 16
    assert parse_age_years("unknown") is None


def test_age_voice_bands():
    assert age_voice_band(5) == "child_parent"
    assert age_voice_band(16) == "teen"
    assert age_voice_band(40) == "adult"
    assert age_voice_band(70) == "older_adult"


def test_parent_age_for_child_in_range():
    assert 30 <= parent_age_for_child(5) <= 45
    assert parent_age_for_child(5) == 33
    assert parent_age_for_child(12) == 40


def test_child_case_uses_parent_voice():
    prompt = build_patient_system_prompt({"age": "5 years old", "gender": "female"})
    assert "parent or guardian" in prompt
    assert "daughter" in prompt


def test_teen_case_uses_teen_voice():
    prompt = build_patient_system_prompt({"age": 16, "gender": "male"})
    assert "teenager" in prompt


def test_patient_responds_with_atomic_facts():
    client = ModelClient(provider="mock", model_name="mock", temperature=0.3)
    case = {
        "case_id": "test_patient_001",
        "chief_complaint": "chest pain for two hours",
        "atomic_facts": [
            "The pain started two hours ago.",
            "It is worse when walking upstairs.",
            "I take aspirin daily.",
        ],
    }
    patient = PatientAgent(client, case)
    answer = patient.respond("When did the pain start?")
    assert isinstance(answer, str) and len(answer) > 0
    assert len(patient.conversation_history) == 2


def test_patient_loads_from_config_cases():
    cfg_path = __import__("pathlib").Path(__file__).resolve().parent.parent / "config" / "trial_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cases = load_cases_from_config(cfg)
    assert cases
    client = ModelClient(provider="mock", model_name="mock")
    patient = PatientAgent(client, cases[0])
    reply = patient.respond("Can you describe your main symptom?")
    assert isinstance(reply, str)
    assert len(reply) > 0
