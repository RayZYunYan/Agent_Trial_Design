"""End-to-end pilot: 5 baseline + 5 grid (API + real judge) + summary + format checks."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from smart_trial.eval.case_lists import PILOT_CASE_IDS
from smart_trial.eval.summary_metrics import build_summary
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger

PILOT_ROOT = PROJECT_ROOT / "smart_trial" / "outputs" / "eval" / "pilot"
SMOKE_ROOT = PROJECT_ROOT / "smart_trial" / "outputs" / "eval" / "smoke"
BASELINE_AGG = PILOT_ROOT / "baseline" / "baseline_encounters.jsonl"
GRID_AGG = PILOT_ROOT / "grid" / "grid_encounters.jsonl"
SUMMARY_CSV = PILOT_ROOT / "summary_metrics.csv"

REQUIRED_ENCOUNTER_KEYS = {
    "encounter_id", "run_mode", "case_id", "seed", "persona",
    "trajectory", "outcome", "timestamp_start", "timestamp_end",
}

BASELINE_MODE = "baseline"
GRID_MODE = "smart_grid"
EXPECTED_BASELINE = len(PILOT_CASE_IDS)
EXPECTED_GRID = len(PILOT_CASE_IDS)


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _validate_encounter(enc: Dict[str, Any], mode: str) -> List[str]:
    errors: List[str] = []
    for key in REQUIRED_ENCOUNTER_KEYS:
        if key not in enc:
            errors.append(f"missing key: {key}")
    if enc.get("run_mode") != mode:
        errors.append(f"run_mode={enc.get('run_mode')!r} expected {mode!r}")
    if not isinstance(enc.get("trajectory"), list) or len(enc["trajectory"]) == 0:
        errors.append("trajectory empty")
    else:
        t0 = enc["trajectory"][0]
        if "doctor" not in t0 or "patient" not in t0:
            errors.append("trajectory missing doctor/patient")
    outcome = enc.get("outcome")
    if not isinstance(outcome, dict) or "diag_correct" not in outcome:
        errors.append("outcome.diag_correct missing")
    if mode == "smart_grid":
        for k in ("path_id", "forced_a1", "forced_a2"):
            if not enc.get(k):
                errors.append(f"missing grid field: {k}")
    if mode == "baseline":
        if enc.get("stage1_arm") is not None:
            errors.append("baseline should have stage1_arm=null")
    return errors


def _print_encounter_sample(path: Path, mode: str, n: int = 1) -> None:
    rows = TrajectoryLogger.load_aggregate(str(path))
    rows = [r for r in rows if r.get("run_mode") == mode]
    print(f"\n--- Sample {mode} ({path.name}, n={len(rows)}) ---")
    for enc in rows[:n]:
        print(json.dumps({
            "encounter_id": enc.get("encounter_id"),
            "case_id": enc.get("case_id"),
            "run_mode": enc.get("run_mode"),
            "path_id": enc.get("path_id"),
            "forced_a1": enc.get("forced_a1"),
            "forced_a2": enc.get("forced_a2"),
            "total_turns": enc.get("total_turns"),
            "diag_correct": (enc.get("outcome") or {}).get("diag_correct"),
            "trajectory_turns": len(enc.get("trajectory") or []),
            "persona_keys": list((enc.get("persona") or {}).keys())[:4],
        }, indent=2))


def validate_all() -> int:
    errors: List[str] = []
    if not BASELINE_AGG.is_file():
        errors.append(f"missing {BASELINE_AGG}")
    if not GRID_AGG.is_file():
        errors.append(f"missing {GRID_AGG}")

    baseline_rows = TrajectoryLogger.load_aggregate(str(BASELINE_AGG)) if BASELINE_AGG.is_file() else []
    grid_rows = TrajectoryLogger.load_aggregate(str(GRID_AGG)) if GRID_AGG.is_file() else []

    bl = [r for r in baseline_rows if r.get("run_mode") == "baseline"]
    gr = [r for r in grid_rows if r.get("run_mode") == "smart_grid"]

    print(f"\n=== Counts: baseline={len(bl)} grid={len(gr)} (expected {EXPECTED_BASELINE}/{EXPECTED_GRID}) ===")
    if len(bl) != EXPECTED_BASELINE:
        errors.append(f"expected {EXPECTED_BASELINE} baseline rows, got {len(bl)}")
    if len(gr) != EXPECTED_GRID:
        errors.append(f"expected {EXPECTED_GRID} grid rows, got {len(gr)}")

    for enc in bl:
        errors.extend(_validate_encounter(enc, "baseline"))
    for enc in gr:
        errors.extend(_validate_encounter(enc, "smart_grid"))

    if gr:
        by_case = {}
        for enc in gr:
            by_case.setdefault(enc["case_id"], []).append(enc.get("persona"))
        for cid, personas in by_case.items():
            if len(personas) > 1 and personas[0] != personas[1]:
                errors.append(f"persona mismatch for {cid} across grid paths")

    if not SUMMARY_CSV.is_file():
        errors.append(f"missing {SUMMARY_CSV}")
    else:
        import pandas as pd
        summary = pd.read_csv(SUMMARY_CSV)
        print(f"\n=== summary_metrics.csv ===\n{summary.to_string(index=False)}")
        for metric in ("baseline", "static_path", "qlearning_adaptive"):
            if metric not in summary["metric"].values:
                errors.append(f"summary missing metric: {metric}")
        adapt = summary[summary["metric"] == "qlearning_adaptive"]
        if not adapt.empty and pd.notna(adapt.iloc[0].get("error")):
            print(f"Note: adaptive issue — {adapt.iloc[0]['error']}")

    _print_encounter_sample(BASELINE_AGG, "baseline")
    _print_encounter_sample(GRID_AGG, "smart_grid")

    if errors:
        print("\n=== VALIDATION FAILED ===")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\n=== PILOT PIPELINE OK ===")
    return 0


def _wipe_test_outputs() -> None:
    for path in (PILOT_ROOT, SMOKE_ROOT, PROJECT_ROOT / "smart_trial" / "outputs" / "eval" / "_test_summary_tmp"):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"  removed {path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    use_api = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GROQ_API_KEY"))
    if not use_api:
        print("No ANTHROPIC_API_KEY or GROQ_API_KEY — using full mock (SMART_TRIAL_USE_MOCK=1)")
    else:
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("Using Anthropic API (claude-sonnet-4-6) for patient/doctor/judge")
        else:
            print("Using Groq API for patient/doctor/judge")

    print("\n=== Clearing prior test outputs (pilot/, smoke/) ===")
    _wipe_test_outputs()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("SMART_TRIAL_MOCK_JUDGE", None)
    env.pop("SMART_TRIAL_USE_MOCK", None)
    if not use_api:
        env["SMART_TRIAL_USE_MOCK"] = "1"

    py = sys.executable
    _run([
        py, "-m", "smart_trial.run_eval",
        "--config", "smart_trial/config/eval/config_baseline_pilot.yaml",
        "--no-resume",
    ], env)
    _run([
        py, "-m", "smart_trial.run_eval",
        "--config", "smart_trial/config/eval/config_grid_pilot.yaml",
        "--no-resume",
    ], env)

    summary = build_summary(
        BASELINE_AGG,
        GRID_AGG,
        case_ids=list(PILOT_CASE_IDS),
        write_category_jsonl=True,
    )
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    return validate_all()


if __name__ == "__main__":
    raise SystemExit(main())
