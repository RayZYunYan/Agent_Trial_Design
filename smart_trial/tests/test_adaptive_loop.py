import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

import yaml

from smart_trial.core.orchestrator import TrialOrchestrator
from smart_trial.data.loader import filter_cases_by_ids, load_cases_from_config
from smart_trial.eval.adaptive_loop import run_adaptive_loop
from smart_trial.eval.case_lists import GRID_PATHS, path_id_for
from smart_trial.tests.test_summary_metrics import _enc


def _write_minimal_grid(path: Path, case_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid in case_ids:
            for a1, a2 in GRID_PATHS:
                pid = path_id_for(a1, a2)
                row = _enc(cid, "smart_grid", pid, a1, a2, a2 == "A2a")
                f.write(json.dumps(row) + "\n")


def _eval_config(name: str) -> str:
    root = Path(__file__).resolve().parent.parent
    return str(root / "config" / "eval" / name)


def test_closed_loop_adaptive_mock():
    tmp_path = Path("smart_trial/outputs/eval/_test_adaptive_tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    root = tmp_path
    grid_path = root / "grid" / "grid_encounters.jsonl"
    adaptive_dir = root / "adaptive"
    case_ids = ["medqa_0100", "medqa_0101"]

    _write_minimal_grid(grid_path, case_ids)

    cfg = {
        "run": {
            "mode": "smart_adaptive_loop",
            "refit_every_n": 1,
            "burn_in": 0,
            "policy_temperature": 1.0,
            "initial_q_from": str(grid_path),
        },
        "eval": {"fix_persona_per_case": True, "case_ids": case_ids},
        "trial": {
            "stage1_turns": 4,
            "stage2_turns": 6,
            "R1_responder_threshold": 6,
            "R2_high_confidence_threshold": 0.7,
        },
        "data": {
            "source": "local",
            "local": {"path": "data/all_dev_good.jsonl", "max_cases": None},
        },
        "models": {
            "patient_simulator": {"provider": "mock", "model_name": "mock", "temperature": 0.3},
            "doctor_agent": {"provider": "mock", "model_name": "mock", "temperature": 0.5},
            "judge": {"provider": "mock", "model_name": "mock", "temperature": 0.1},
        },
        "randomization": {"seed": 42},
        "persona": {"mode": "per_case_seed"},
        "rag": {"enabled": False},
        "logging": {
            "output_dir": str(adaptive_dir),
            "aggregate_filename": "adaptive_encounters.jsonl",
        },
    }
    cfg_path = tmp_path / "config_adaptive_test.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    orch = TrialOrchestrator(config_path=str(cfg_path))
    with open(orch.config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    cases = load_cases_from_config(loaded)
    run_cases = filter_cases_by_ids(cases, case_ids)

    ran = run_adaptive_loop(orch, run_cases, seed=42, resume=False)
    assert ran == len(case_ids)

    agg = adaptive_dir / "adaptive_encounters.jsonl"
    lines = agg.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(case_ids)

    first = json.loads(lines[0])
    assert first["run_mode"] == "smart_adaptive_loop"
    assert first.get("assignment_mode") == "policy_biased_random"
    assert first.get("stage1_propensity") is not None
    assert first.get("refit_generation") is not None

    # Resume should skip both
    ran2 = run_adaptive_loop(orch, run_cases, seed=42, resume=True)
    assert ran2 == 0

    shutil.rmtree(tmp_path, ignore_errors=True)
