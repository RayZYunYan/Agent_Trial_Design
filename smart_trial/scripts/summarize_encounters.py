"""
Read encounter JSONL logs and print a concise summary for pilot review.

Usage (from repo root):
  python -m smart_trial.scripts.summarize_encounters
  python -m smart_trial.scripts.summarize_encounters --output-dir smart_trial/outputs/encounters
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger


def summarize(output_dir: str, responder_threshold: int = 6) -> None:
    encounters = TrajectoryLogger.load_all_encounters(output_dir)
    if not encounters:
        print(f"No encounters found in {output_dir}")
        return

    real = [e for e in encounters if not _is_mock_encounter(e)]
    if real:
        print(f"Note: {len(encounters) - len(real)} mock-only encounter(s) excluded from stats.")
        encounters = real

    print(f"\n{'=' * 60}")
    print(f"SMART Trial Summary - {len(encounters)} encounter(s)")
    print(f"{'=' * 60}")

    s1_arms = Counter(e.get("stage1_arm", "?") for e in encounters)
    s2_arms = Counter(e.get("stage2_arm", "?") for e in encounters)
    s3_arms = Counter(e.get("stage3_arm", "?") for e in encounters)
    print(f"\nStage 1 arm distribution: {dict(s1_arms)}")
    print(f"Stage 2 arm distribution: {dict(s2_arms)}")
    print(f"Stage 3 arm distribution: {dict(s3_arms)}")

    categories = Counter(e.get("case_category", "?") for e in encounters)
    print(f"\nCase categories: {dict(categories)}")

    r1_scores = []
    for e in encounters:
        r1 = e.get("R1") or {}
        if r1.get("total") is not None:
            r1_scores.append(int(r1["total"]))
    if r1_scores:
        responder_rate = sum(1 for s in r1_scores if s >= responder_threshold) / len(r1_scores)
        print(
            f"\nR1 scores: mean={sum(r1_scores) / len(r1_scores):.1f}, "
            f"responder_rate={responder_rate:.1%} (threshold>={responder_threshold})"
        )

    r2_levels = Counter((e.get("R2") or {}).get("confidence_level", "?") for e in encounters)
    if any(k != "?" for k in r2_levels):
        print(f"R2 confidence levels: {dict(r2_levels)}")

    outcomes = [e.get("outcome") for e in encounters if e.get("outcome")]
    if outcomes:
        n_correct = sum(1 for o in outcomes if o.get("diag_correct"))
        n_dangerous = sum(1 for o in outcomes if o.get("dangerous_advice"))
        n_rf_miss = sum(1 for o in outcomes if o.get("red_flag_miss"))
        print("\nOutcomes:")
        print(f"  Diagnostic accuracy: {n_correct}/{len(outcomes)} = {n_correct / len(outcomes):.1%}")
        print(f"  Dangerous advice: {n_dangerous}/{len(outcomes)}")
        print(f"  Red flag miss: {n_rf_miss}/{len(outcomes)}")

    trajectories = Counter(
        f"{e.get('stage1_arm', '?')}->{e.get('stage2_arm', '?')}->{e.get('stage3_arm', '?')}"
        for e in encounters
    )
    print("\nTop trajectories:")
    for traj, count in trajectories.most_common(5):
        print(f"  {traj}: {count}x")

    avg_turns = sum(e.get("total_turns", 0) for e in encounters) / len(encounters)
    print(f"\nAvg turns used: {avg_turns:.1f}")


def _is_mock_encounter(encounter: dict) -> bool:
    traj = encounter.get("trajectory") or []
    if not traj:
        return False
    for turn in traj[:3]:
        doc = turn.get("doctor", "")
        if "[MOCK]" in doc:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SMART encounter JSONL logs")
    parser.add_argument(
        "--output-dir",
        default="smart_trial/outputs/encounters",
        help="Directory containing per-case *.jsonl files",
    )
    parser.add_argument(
        "--r1-threshold",
        type=int,
        default=6,
        help="R1 responder threshold (out of 10)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    summarize(str(out), responder_threshold=args.r1_threshold)


if __name__ == "__main__":
    main()
