import os

# Force the mock provider so these tests do not require a real Groq API key.
# The orchestrator honors this env var in _make_client.
os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

import yaml

from smart_trial.core.orchestrator import TrialOrchestrator
from smart_trial.data.loader import load_cases_from_config


def test_run_encounter_mock_smoke():
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
