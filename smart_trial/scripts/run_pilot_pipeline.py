"""End-to-end pilot: 3 baseline + 3 grid (Groq + mock judge) + summary + format checks."""
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

from smart_trial.eval.summary_metrics import build_summary
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger

PILOT_ROOT = PROJECT_ROOT / "smart_trial" / "outputs" / "eval" / "pilot"
BASELINE_AGG = PILOT_ROOT / "baseline" / "baseline_encounters.jsonl"
GRID_AGG = PILOT_ROOT / "grid" / "grid_encounters.jsonl"
SUMMARY_CSV = PILOT_ROOT / "summary_metrics.csv"

REQUIRED_ENCOUNTER_KEYS = {
    "encounter_id", "run_mode", "case_id", "seed", "persona",
    "trajectory", "outcome", "timestamp_start", "timestamp_end",
}

BASELINE_MODE = "baseline"
GRID_MODE = "smart_grid"


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

    print(f"\n=== Counts: baseline={len(bl)} grid={len(gr)} ===")
    if len(bl) != 3:
        errors.append(f"expected 3 baseline rows, got {len(bl)}")
    if len(gr) != 3:
        errors.append(f"expected 3 grid rows, got {len(gr)}")

    for enc in bl:
        errors.extend(_validate_encounter(enc, "baseline"))
    for enc in gr:
        errors.extend(_validate_encounter(enc, "smart_grid"))

    # Same persona per case across grid paths (trivial with 1 path; check fields exist)
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
        # Pilot with 1 grid path may not fit Q-learning (single Y class) — allow error column
        adapt = summary[summary["metric"] == "qlearning_adaptive"]
        if not adapt.empty and pd.notna(adapt.iloc[0].get("error")):
            print(f"Note: adaptive skipped — {adapt.iloc[0]['error']}")

    _print_encounter_sample(BASELINE_AGG, "baseline")
    _print_encounter_sample(GRID_AGG, "smart_grid")

    if errors:
        print("\n=== VALIDATION FAILED ===")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\n=== PILOT PIPELINE OK ===")
    return 0


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    use_groq = bool(os.environ.get("GROQ_API_KEY"))
    if not use_groq:
        print("GROQ_API_KEY not set — using full mock (SMART_TRIAL_USE_MOCK=1)")
    else:
        print("Using Groq for patient/doctor; judge mocked (SMART_TRIAL_MOCK_JUDGE=1)")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if use_groq:
        env["SMART_TRIAL_MOCK_JUDGE"] = "1"
    else:
        env["SMART_TRIAL_USE_MOCK"] = "1"

    # Fresh pilot output
    shutil.rmtree(PILOT_ROOT, ignore_errors=True)

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

    summary = build_summary(BASELINE_AGG, GRID_AGG, case_ids=["medqa_0000", "medqa_0001", "medqa_0002"])
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    return validate_all()


if __name__ == "__main__":
    raise SystemExit(main())
