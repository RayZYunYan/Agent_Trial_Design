import os

# Force the mock provider so these tests do not require a real Groq API key.
# The orchestrator honors this env var in _make_client.
os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

import yaml

from smart_trial.core.orchestrator import TrialOrchestrator
from smart_trial.data.loader import load_cases_from_config


def _orch_and_cases():
    orch = TrialOrchestrator()
    with open(orch.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cases = load_cases_from_config(cfg)
    return orch, cases


def test_run_encounter_mock_smoke():
    """Default config (both personas off) -> baseline behavior preserved."""
    orch, cases = _orch_and_cases()
    assert cases
    traj = orch.run_encounter(cases[0], seed=42)
    assert traj["stage1_arm"] in ("A1a", "A1b", "A1c")
    assert traj["stage2_arm"] in ("A2a", "A2b", "A2c")
    assert traj["stage3_arm"] in ("A3a", "A3b", "A3c")
    assert traj.get("outcome") is not None
    # JSONL persona fields are present-but-null when both modes are off.
    assert traj.get("literacy_persona") is None
    assert traj.get("trust_persona") is None
    # Outcome surfaces the concealed-topics fields even with no persona attached.
    outcome = traj["outcome"]
    assert outcome.get("concealed_topics_configured") == []
    assert outcome.get("concealed_topics_surfaced") == []


def test_run_encounter_with_both_personas_fixed():
    """persona.mode: fixed literacy_F AND trust_persona.mode: fixed wary_mistruster."""
    orch, cases = _orch_and_cases()
    orch.config["persona"] = {"mode": "fixed", "fixed_id": "literacy_F"}
    orch.config["trust_persona"] = {"mode": "fixed", "fixed_id": "wary_mistruster"}

    traj = orch.run_encounter(cases[0], seed=42)

    # Both persona blocks land in the JSONL.
    assert traj["literacy_persona"] is not None
    assert traj["literacy_persona"]["persona_id"] == "literacy_F"
    assert traj["literacy_persona"]["nutbeam_level"] == "functional"
    assert traj["literacy_persona"]["axes"] == {
        "vocabulary_register": "F",
        "jargon_comprehension": "F",
        "anatomical_localization": "F",
    }

    assert traj["trust_persona"] is not None
    assert traj["trust_persona"]["persona_id"] == "wary_mistruster"
    assert traj["trust_persona"]["axes"]["trust_in_clinician"] == "low"
    assert traj["trust_persona"]["axes"]["reaction_to_insensitive"] == "withdrawal"

    # Outcome carries the equity-gap measurement target.
    outcome = traj["outcome"]
    assert outcome["concealed_topics_configured"] == ["alcohol", "drug_use", "mental_health"]
    assert set(outcome["concealed_topics_surfaced"]).issubset(
        set(outcome["concealed_topics_configured"])
    )

    # Pipeline still produces a valid SMART trajectory.
    assert traj["stage1_arm"] in ("A1a", "A1b", "A1c")
    assert traj["stage2_arm"] in ("A2a", "A2b", "A2c")
    assert traj["stage3_arm"] in ("A3a", "A3b", "A3c")


def test_run_encounter_demographic_literacy():
    """persona.mode: demographic auto-resolves to literacy_F or literacy_I."""
    orch, cases = _orch_and_cases()
    orch.config["persona"] = {"mode": "demographic"}
    orch.config["trust_persona"] = {"mode": "off"}

    traj = orch.run_encounter(cases[0], seed=42)
    assert traj["literacy_persona"] is not None
    # v1 demographic rule never returns literacy_C.
    assert traj["literacy_persona"]["persona_id"] in ("literacy_F", "literacy_I")
    assert traj["trust_persona"] is None
