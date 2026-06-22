import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("SMART_TRIAL_USE_MOCK", "1")

from smart_trial.eval.case_lists import CASE_CATEGORIES, GRID_PATHS, path_id_for
from smart_trial.eval.summary_metrics import build_summary, write_category_jsonls
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger


def _enc(
    case_id: str,
    run_mode: str,
    path_id: str | None,
    a1,
    a2,
    correct: bool,
    *,
    case_category: str = "Other",
):
    return {
        "encounter_id": f"enc_{case_id}_{path_id or 'base'}",
        "run_mode": run_mode,
        "path_id": path_id,
        "forced_a1": a1,
        "forced_a2": a2,
        "case_id": case_id,
        "case_category": case_category,
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
    baseline_path = tmp_path / "baseline" / "baseline_encounters.jsonl"
    grid_path = tmp_path / "grid" / "grid_encounters.jsonl"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.parent.mkdir(parents=True, exist_ok=True)

    case_ids = ["medqa_0000", "medqa_0001"]

    with open(baseline_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_enc("medqa_0000", "baseline", None, None, None, True, case_category="Other")) + "\n")
        f.write(json.dumps(_enc("medqa_0001", "baseline", None, None, None, False, case_category="Cardiology")) + "\n")

    with open(grid_path, "w", encoding="utf-8") as f:
        for cid, cat in [("medqa_0000", "Other"), ("medqa_0001", "Cardiology")]:
            for a1, a2 in GRID_PATHS:
                pid = path_id_for(a1, a2)
                f.write(json.dumps(_enc(cid, "smart_grid", pid, a1, a2, a2 == "A2a", case_category=cat)) + "\n")

    summary = build_summary(
        baseline_path,
        grid_path,
        case_ids=case_ids,
        write_category_jsonl=True,
    )

    baseline_row = summary[summary["metric"] == "baseline"].iloc[0]
    assert baseline_row["avg_correctness"] == 0.5

    a2a_rows = summary[(summary["metric"] == "static_path") & (summary["path_id"] == "A1a_A2a")]
    assert float(a2a_rows.iloc[0]["avg_correctness"]) == 1.0

    adaptive_row = summary[summary["metric"] == "qlearning_adaptive"].iloc[0]
    assert adaptive_row["n"] == 2
    assert 0.0 <= float(adaptive_row["avg_correctness"]) <= 1.0

    cat_baseline = summary[summary["metric"] == "category_baseline"]
    assert len(cat_baseline) == len(CASE_CATEGORIES)
    other_baseline = cat_baseline[cat_baseline["case_category"] == "Other"].iloc[0]
    cardio_baseline = cat_baseline[cat_baseline["case_category"] == "Cardiology"].iloc[0]
    assert other_baseline["n"] == 1
    assert cardio_baseline["n"] == 1
    assert float(other_baseline["avg_correctness"]) == 1.0
    assert float(cardio_baseline["avg_correctness"]) == 0.0

    cat_static = summary[summary["metric"] == "category_static_path"]
    assert len(cat_static) == len(GRID_PATHS) * len(CASE_CATEGORIES)
    a2a_other = cat_static[
        (cat_static["path_id"] == "A1a_A2a") & (cat_static["case_category"] == "Other")
    ].iloc[0]
    assert a2a_other["n"] == 1
    assert float(a2a_other["avg_correctness"]) == 1.0

    cat_adaptive = summary[summary["metric"] == "category_adaptive"]
    assert len(cat_adaptive) == len(CASE_CATEGORIES)

    assert len(TrajectoryLogger.load_aggregate(str(baseline_path.parent / "by_category" / "Other.jsonl"))) == 1
    assert len(TrajectoryLogger.load_aggregate(str(grid_path.parent / "by_category" / "Other.jsonl"))) == len(GRID_PATHS)

    assert summary.shape[0] == 99 + 1 + len(CASE_CATEGORIES)  # offline rows + closed-loop block

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_write_category_jsonls_separate_dirs():
    tmp_path = Path("smart_trial/outputs/eval/_test_summary_jsonl")
    shutil.rmtree(tmp_path, ignore_errors=True)
    baseline_dir = tmp_path / "baseline" / "by_category"
    grid_dir = tmp_path / "grid" / "by_category"

    baseline_rows = [_enc("medqa_0000", "baseline", None, None, None, True, case_category="Other")]
    grid_rows = [
        _enc("medqa_0000", "smart_grid", path_id_for(a1, a2), a1, a2, True, case_category="Other")
        for a1, a2 in GRID_PATHS
    ]

    write_category_jsonls(baseline_rows, baseline_dir)
    write_category_jsonls(grid_rows, grid_dir)

    assert len(TrajectoryLogger.load_aggregate(str(baseline_dir / "Other.jsonl"))) == 1
    assert len(TrajectoryLogger.load_aggregate(str(grid_dir / "Other.jsonl"))) == len(GRID_PATHS)

    shutil.rmtree(tmp_path, ignore_errors=True)
