import yaml

from smart_trial.core.patient_agent import (
    PatientAgent,
    age_voice_band,
    build_patient_system_prompt,
    communication_cues,
    parent_age_for_child,
    parse_age_years,
)
from smart_trial.core.persona import PatientPersona, load_persona_from_id
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


def test_prompt_uses_case_background_communication_cues():
    case = {
        "age": "21 years old",
        "gender": "male",
        "full_record": "A 21-year-old sexually active male has pain during urination.",
    }
    prompt = build_patient_system_prompt(case)
    assert "sexual-health details" in prompt
    assert "stereotypes" in prompt


def test_psychiatric_background_can_sound_guarded():
    cues = communication_cues(
        {
            "age": "43",
            "gender": "male",
            "full_record": "When probed further, the patient accuses the physician of being angry and hostile.",
        }
    )
    assert any("guarded" in cue for cue in cues)


def test_work_background_is_detected_without_guessing():
    cues = communication_cues(
        {
            "age": "37 years old",
            "gender": "male",
            "full_record": "He works as a logger in Washington.",
        }
    )
    assert any("work or school" in cue for cue in cues)


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


# ---------------------------------------------------------------------------
# PatientPersona tests
# ---------------------------------------------------------------------------

def test_persona_neutral_renders_empty_block():
    p = PatientPersona()
    assert p.render_persona_block() == ""


def test_persona_anxious_renders_instruction():
    p = PatientPersona(personality="anxious")
    block = p.render_persona_block()
    assert "anxious" in block.lower() or "worried" in block.lower() or "reassurance" in block.lower()


def test_persona_recall_low_renders_instruction():
    p = PatientPersona(recall="low")
    assert "uncertain" in p.render_persona_block().lower() or "hedging" in p.render_persona_block().lower()


def test_jargon_detects_specialist_term_for_interactive_level():
    p = PatientPersona(jargon_comprehension="interactive")
    assert p.detects_jargon("do you have any syncope?") == "syncope"


def test_jargon_skips_for_critical_level():
    p = PatientPersona(jargon_comprehension="critical")
    assert p.detects_jargon("do you have any syncope?") is None


def test_jargon_functional_detects_common_medical_term():
    p = PatientPersona(jargon_comprehension="functional")
    assert p.detects_jargon("what is the onset of your symptoms?") == "onset"


def test_jargon_interactive_does_not_trigger_on_common_term():
    p = PatientPersona(jargon_comprehension="interactive")
    assert p.detects_jargon("what is the onset of your symptoms?") is None


def test_clarification_prompt_contains_term():
    p = PatientPersona()
    prompt = p.clarification_prompt("syncope")
    assert "syncope" in prompt


def test_persona_from_dict_roundtrip():
    p = PatientPersona(personality="reserved", vocabulary_register="functional", recall="low")
    p2 = PatientPersona.from_dict(p.to_dict())
    assert p2.personality == "reserved"
    assert p2.vocabulary_register == "functional"
    assert p2.recall == "low"


def test_persona_yaml_preset_loads():
    p = load_persona_from_id("difficult")
    assert p.personality == "distrustful"
    assert p.jargon_comprehension == "functional"


def test_patient_agent_accepts_persona():
    from smart_trial.models.model_client import ModelClient
    client = ModelClient(provider="mock", model_name="mock")
    case = {"case_id": "test_001", "atomic_facts": ["The pain started two hours ago."]}
    persona = PatientPersona(personality="anxious", jargon_comprehension="interactive")
    agent = PatientAgent(client, case, persona=persona)
    assert agent.persona.personality == "anxious"


def test_jargon_short_circuit_in_agent():
    from smart_trial.models.model_client import ModelClient
    client = ModelClient(provider="mock", model_name="mock")
    case = {"case_id": "test_002", "atomic_facts": ["Patient has chest pain."]}
    persona = PatientPersona(jargon_comprehension="functional")
    agent = PatientAgent(client, case, persona=persona)
    answer = agent.respond("What is the onset of your symptoms?")
    assert isinstance(answer, str) and len(answer) > 0
