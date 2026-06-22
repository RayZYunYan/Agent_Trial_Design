"""Claude smoke: 1 baseline + 1×9 grid with real judge (requires ANTHROPIC_API_KEY)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from smart_trial.eval.output_utils import validate_grid_aggregate
from smart_trial.eval.summary_metrics import build_summary
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger

SMOKE_ROOT = PROJECT_ROOT / "smart_trial" / "outputs" / "eval" / "smoke_claude"
BASELINE_AGG = SMOKE_ROOT / "baseline" / "baseline_encounters.jsonl"
GRID_AGG = SMOKE_ROOT / "grid" / "grid_encounters.jsonl"
SUMMARY_CSV = SMOKE_ROOT / "summary_metrics.csv"
CASE_ID = "medqa_0000"

BASELINE_CONFIG = "smart_trial/config/eval/config_smoke_claude_baseline.yaml"
GRID_CONFIG = "smart_trial/config/eval/config_smoke_claude_grid.yaml"


def _wipe_dir(path: Path) -> None:
    if not path.exists():
        return
    for f in path.rglob("*.jsonl"):
        try:
            f.unlink()
        except OSError as exc:
            print(f"  warning: could not delete {f}: {exc}")
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        print(f"  note: {path.relative_to(PROJECT_ROOT)} still exists (close open files in IDE)")
    else:
        print(f"  removed {path.relative_to(PROJECT_ROOT)}")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env", file=sys.stderr)
        return 1

    print("=== Clearing smoke_claude output ===")
    _wipe_dir(SMOKE_ROOT)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("SMART_TRIAL_MOCK_JUDGE", None)
    env.pop("SMART_TRIAL_USE_MOCK", None)
    print("\nMode: Claude — patient/judge claude-haiku-4-5, doctor claude-sonnet-4-6")

    py = sys.executable
    _run([py, "-m", "smart_trial.run_eval", "--config", BASELINE_CONFIG, "--no-resume"], env)
    _run([py, "-m", "smart_trial.run_eval", "--config", GRID_CONFIG, "--no-resume"], env)

    summary = build_summary(
        BASELINE_AGG,
        GRID_AGG,
        case_ids=[CASE_ID],
        write_category_jsonl=True,
    )
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    baseline_cat = BASELINE_AGG.parent / "by_category" / "Other.jsonl"
    grid_cat = GRID_AGG.parent / "by_category" / "Other.jsonl"

    print("\n=== Output files ===")
    for p in [BASELINE_AGG, GRID_AGG, baseline_cat, grid_cat, SUMMARY_CSV]:
        n = 0
        if p.suffix == ".jsonl" and p.is_file():
            n = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
        print(f"  {p.relative_to(PROJECT_ROOT)}  ({n} lines)" if n else f"  {p.relative_to(PROJECT_ROOT)}")

    baseline_rows = TrajectoryLogger.load_aggregate(str(BASELINE_AGG))
    print(f"\n=== Baseline: {len(baseline_rows)} row(s), case={baseline_rows[0]['case_id'] if baseline_rows else '?'} ===")
    if baseline_rows:
        out = baseline_rows[0].get("outcome") or {}
        print(f"  diag_correct={out.get('diag_correct')} reasoning={out.get('reasoning', '')[:80]}")

    grid_report = validate_grid_aggregate(GRID_AGG, expected_cases=[CASE_ID], expected_paths_per_case=9)
    print("\n=== Grid validation ===")
    print(json.dumps(grid_report, indent=2))

    print("\n=== summary_metrics.csv (rows with data) ===")
    print(summary[summary["n"].fillna(0) > 0].to_string(index=False))

    cat_rows = len(summary[summary["metric"].str.startswith("category_")])
    expected_summary_rows = 11 + 8 + 72 + 8
    category_ok = (
        baseline_cat.is_file()
        and grid_cat.is_file()
        and len(TrajectoryLogger.load_aggregate(str(baseline_cat))) == 1
        and len(TrajectoryLogger.load_aggregate(str(grid_cat))) == 9
        and cat_rows == 88
        and len(summary) == expected_summary_rows
    )

    judge_real = True
    if baseline_rows:
        reasoning = str((baseline_rows[0].get("outcome") or {}).get("reasoning", ""))
        r1 = baseline_rows[0].get("R1") or {}
        if "mock judge" in reasoning.lower() or r1.get("reasoning") == "mock judge default":
            judge_real = False
            print("\nWARNING: outcome looks like mock judge — check env vars")

    if grid_report["ok"] and len(baseline_rows) == 1 and category_ok and judge_real:
        print("\n=== CLAUDE SMOKE OK ===")
        return 0

    print("\n=== CLAUDE SMOKE FAILED ===")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
