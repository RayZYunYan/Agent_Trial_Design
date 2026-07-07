import os
import tempfile
from pathlib import Path

import yaml

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

from smart_trial.core.coverage import CoverageSession, parse_coverage_config
from smart_trial.core.judge import StageJudge
from smart_trial.core.orchestrator import TrialOrchestrator, _parse_rounds_config
from smart_trial.data.loader import load_cases_from_config
from smart_trial.models.model_client import ModelClient


def _write_config(tmpdir: Path, trial_overrides: dict) -> Path:
    base = {
        "trial": {
            "stage1_turns": 10,
            "stage2_turns": 2,
            "R2_high_confidence_threshold": 0.7,
            "coverage": {
                "enabled": True,
                "check_every": 5,
                "threshold": 0.50,
                "stage1_min_turns": 2,
                "stage2_min_turns": 2,
            },
            "rounds": {"mode": "fixed"},
        },
        "data": {
            "source": "local",
            "local": {"path": "data/all_dev_good.jsonl", "max_cases": 1},
        },
        "models": {
            "patient_simulator": {"provider": "mock", "model_name": "mock", "temperature": 0.3},
            "doctor_agent": {"provider": "mock", "model_name": "mock", "temperature": 0.5},
            "judge": {"provider": "mock", "model_name": "mock", "temperature": 0.1},
        },
        "randomization": {"seed": 42},
        "persona": {"mode": "off"},
        "rag": {"enabled": False},
        "mediq": {"enabled": True},
        "logging": {"output_dir": str(tmpdir / "out")},
    }
    base["trial"].update(trial_overrides)
    cfg_path = tmpdir / "trial.yaml"
    cfg_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return cfg_path


def test_parse_coverage_config_defaults():
    cfg = parse_coverage_config(
        {
            "stage1_turns": 4,
            "stage2_turns": 6,
            "rounds": {"stage1": {"min_turns": 5}, "stage2": {"min_turns": 3}},
        }
    )
    assert cfg["enabled"] is True
    assert cfg["check_every"] == 5
    assert cfg["threshold"] == 0.50
    assert cfg["stage1_min_turns"] == 5
    assert cfg["stage2_min_turns"] == 3


def test_coverage_session_monotonic():
    judge = StageJudge(ModelClient("mock", "mock"))
    case = {
        "case_id": "t",
        "atomic_facts": ["f1", "f2", "f3", "f4"],
        "chief_complaint": "fever",
    }
    session = CoverageSession(judge, case, threshold=0.5, check_every=1)
    hist = [
        {"role": "assistant", "content": "When did fever start?"},
        {"role": "user", "content": "Two days ago."},
    ]
    session.probe_if_due(1, hist)
    first = session.coverage_count
    session.probe_if_due(2, hist + [{"role": "assistant", "content": "Any cough?"}])
    assert session.coverage_count >= first


def test_adaptive_stage1_stops_on_coverage():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {
                "stage1_turns": 10,
                "coverage": {"check_every": 5, "threshold": 0.40, "stage1_min_turns": 2},
                "rounds": {"mode": "adaptive", "stage1": {"min_turns": 2}},
            },
        )
        orch = TrialOrchestrator(str(cfg_path))
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cases = load_cases_from_config(cfg)
        traj = orch.run_encounter(cases[0], seed=42)

        stage1_turns = [t for t in traj["trajectory"] if t.get("stage") == 1]
        assert traj["rounds_mode"] == "adaptive"
        assert traj["stage1_turns_used"] == 5
        assert traj.get("stage1_early_stop") == "fact_coverage"
        assert len(stage1_turns) == 5
        assert "final_coverage_rate" in traj


def test_fixed_stage1_runs_full_budget():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {"stage1_turns": 3, "rounds": {"mode": "fixed"}},
        )
        orch = TrialOrchestrator(str(cfg_path))
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cases = load_cases_from_config(cfg)
        traj = orch.run_encounter(cases[0], seed=42)

        stage1_turns = [t for t in traj["trajectory"] if t.get("stage") == 1]
        assert traj["rounds_mode"] == "fixed"
        assert traj["stage1_turns_used"] == 3
        assert "stage1_early_stop" not in traj
        assert len(stage1_turns) == 3


def test_parse_rounds_config_defaults_fixed():
    cfg = _parse_rounds_config({"stage1_turns": 4, "stage2_turns": 6})
    assert cfg["mode"] == "fixed"
    assert cfg["adaptive"] is False
    assert cfg["stage1_max"] == 4
