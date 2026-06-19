import os
import tempfile
from pathlib import Path

import yaml

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

from smart_trial.core.orchestrator import (
    TrialOrchestrator,
    _parse_rounds_config,
    _r1_allows_early_stop,
)
from smart_trial.data.loader import load_cases_from_config


def _write_config(tmpdir: Path, trial_overrides: dict) -> Path:
    base = {
        "trial": {
            "stage1_turns": 4,
            "stage2_turns": 2,
            "R1_responder_threshold": 6,
            "R2_high_confidence_threshold": 0.7,
            "early_diagnosis_confidence": 0.80,
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
        "logging": {"output_dir": str(tmpdir / "out")},
    }
    base["trial"].update(trial_overrides)
    cfg_path = tmpdir / "trial.yaml"
    cfg_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return cfg_path


def test_parse_rounds_config_defaults_fixed():
    cfg = _parse_rounds_config({"stage1_turns": 4, "stage2_turns": 6})
    assert cfg["mode"] == "fixed"
    assert cfg["adaptive"] is False
    assert cfg["stage1_max"] == 4
    assert cfg["early_diagnosis_confidence"] == 0.80


def test_r1_allows_early_stop_respects_red_flags_min():
    r1 = {"responder": True, "red_flags": 0}
    assert _r1_allows_early_stop(r1, red_flags_min=1) is False
    assert _r1_allows_early_stop(r1, red_flags_min=None) is True
    r1_ok = {"responder": True, "red_flags": 2}
    assert _r1_allows_early_stop(r1_ok, red_flags_min=1) is True


def test_adaptive_stage1_stops_early():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = _write_config(
            Path(tmp),
            {
                "rounds": {
                    "mode": "adaptive",
                    "stage1": {"min_turns": 2, "check_every": 1, "red_flags_min": 1},
                    "stage2": {"min_turns": 99, "consecutive_high": 99, "auto_conclude": False},
                },
            },
        )
        orch = TrialOrchestrator(str(cfg_path))
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cases = load_cases_from_config(cfg)
        traj = orch.run_encounter(cases[0], seed=42)

        stage1_turns = [t for t in traj["trajectory"] if t.get("stage") == 1]
        assert traj["rounds_mode"] == "adaptive"
        assert traj["stage1_turns_used"] == 2
        assert traj.get("stage1_early_stop") == "r1_responder"
        assert len(stage1_turns) == 2


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
