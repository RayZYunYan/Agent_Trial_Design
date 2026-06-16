import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

from smart_trial.eval.summary_metrics import build_summary


def _enc(case_id: str, run_mode: str, path_id: str | None, a1, a2, correct: bool):
    return {
        "encounter_id": f"enc_{case_id}_{path_id or 'base'}",
        "run_mode": run_mode,
        "path_id": path_id,
        "forced_a1": a1,
        "forced_a2": a2,
        "case_id": case_id,
        "case_category": "Other",
        "chief_complaint": "test",
        "seed": 42,
        "persona": {},
        "stage1_arm": a1,
        "R1": {
            "OPQRST": 1, "red_flags": 0, "past_medical_history": 0,
            "medications_allergies": 0, "social_family_history": 0,
            "total": 5, "responder": False, "reasoning": "t",
        },
        "stage2_arm": a2,
        "stage2_pool": "non-responder",
        "R2": {
            "final_confidence": 0.5, "avg_confidence": 0.5,
            "confidence_level": "low", "confidence_scores": [],
            "r2_source": "mock",
        },
        "outcome": {"diag_correct": correct},
        "total_turns": 8,
        "trajectory": [
            {"turn": 1, "stage": 1, "arm": a1 or "baseline", "doctor": "d", "patient": "p"},
            {"turn": 5, "stage": 2, "arm": a2 or "baseline", "doctor": "d2", "patient": "p2"},
        ],
    }


def test_build_summary():
    tmp_path = Path("smart_trial/outputs/eval/_test_summary_tmp")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    case_ids = ["medqa_0000", "medqa_0001"]
    baseline_path = tmp_path / "baseline_encounters.jsonl"
    grid_path = tmp_path / "grid_encounters.jsonl"

    with open(baseline_path, "w", encoding="utf-8") as f:
        for cid in case_ids:
            f.write(json.dumps(_enc(cid, "baseline", None, None, None, True)) + "\n")

    from smart_trial.eval.case_lists import GRID_PATHS, path_id_for

    with open(grid_path, "w", encoding="utf-8") as f:
        for cid in case_ids:
            for a1, a2 in GRID_PATHS:
                pid = path_id_for(a1, a2)
                f.write(json.dumps(_enc(cid, "smart_grid", pid, a1, a2, a2 == "A2a")) + "\n")

    summary = build_summary(baseline_path, grid_path, case_ids=case_ids)
    baseline_row = summary[summary["metric"] == "baseline"].iloc[0]
    assert baseline_row["avg_correctness"] == 1.0

    a2a_rows = summary[(summary["metric"] == "static_path") & (summary["path_id"] == "A1a_A2a")]
    assert float(a2a_rows.iloc[0]["avg_correctness"]) == 1.0

    adaptive_row = summary[summary["metric"] == "qlearning_adaptive"].iloc[0]
    assert adaptive_row["n"] == 2
    assert 0.0 <= float(adaptive_row["avg_correctness"]) <= 1.0

    shutil.rmtree(tmp_path, ignore_errors=True)
