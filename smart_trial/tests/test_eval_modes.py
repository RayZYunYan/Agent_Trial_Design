import os

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

import yaml

from smart_trial.core.orchestrator import TrialOrchestrator
from smart_trial.data.loader import filter_cases_by_ids, load_cases_from_config
from smart_trial.eval.case_lists import BENCHMARK_CASE_IDS, path_id_for


def _eval_config(name: str) -> str:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    return str(root / "config" / "eval" / name)


def test_baseline_encounter_mock():
    orch = TrialOrchestrator(config_path=_eval_config("config_baseline.yaml"))
    with open(orch.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cases = load_cases_from_config(cfg)
    case = filter_cases_by_ids(cases, BENCHMARK_CASE_IDS[:1])[0]
    traj = orch.run_encounter(case, seed=42)
    assert traj["run_mode"] == "baseline"
    assert traj["stage1_arm"] is None
    assert traj["R1"] is None
    assert traj.get("outcome") is not None


def test_grid_encounter_forced_path_mock():
    orch = TrialOrchestrator(config_path=_eval_config("config_grid.yaml"))
    with open(orch.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cases = load_cases_from_config(cfg)
    case = filter_cases_by_ids(cases, BENCHMARK_CASE_IDS[:1])[0]
    traj = orch.run_encounter(
        case,
        seed=42,
        forced_a1="A1a",
        forced_a2="A2b",
        path_id=path_id_for("A1a", "A2b"),
        run_mode="smart_grid",
    )
    assert traj["run_mode"] == "smart_grid"
    assert traj["stage1_arm"] == "A1a"
    assert traj["stage2_arm"] == "A2b"
    assert traj["path_id"] == "A1a_A2b"
    assert traj.get("R2") is not None


def test_fix_persona_same_across_paths():
    orch = TrialOrchestrator(config_path=_eval_config("config_grid.yaml"))
    with open(orch.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cases = load_cases_from_config(cfg)
    case = filter_cases_by_ids(cases, BENCHMARK_CASE_IDS[:1])[0]
    t1 = orch.run_encounter(case, seed=42, forced_a1="A1a", forced_a2="A2a", run_mode="smart_grid")
    t2 = orch.run_encounter(case, seed=42, forced_a1="A1c", forced_a2="A2c", run_mode="smart_grid")
    assert t1["persona"] == t2["persona"]
