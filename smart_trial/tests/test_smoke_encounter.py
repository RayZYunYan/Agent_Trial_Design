import yaml

from smart_trial.core.orchestrator import TrialOrchestrator
from smart_trial.data.loader import load_cases_from_config


def test_run_encounter_mock_smoke():
    orch = TrialOrchestrator()
    with open(orch.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cases = load_cases_from_config(cfg)
    assert cases
    traj = orch.run_encounter(cases[0], seed=42)
    assert traj["stage1_arm"] in ("A1a", "A1b", "A1c")
    assert traj["stage2_arm"] in ("A2a", "A2b", "A2c")
    assert traj["stage3_arm"] in ("A3a", "A3b", "A3c")
    assert traj.get("outcome") is not None
