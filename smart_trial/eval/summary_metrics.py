"""Compute average correctness: baseline, static paths, Q-learning adaptive."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_trial.eval.case_lists import BENCHMARK_CASE_IDS, GRID_PATHS, path_id_for
from smart_trial.q_learning import load as ql_load
from smart_trial.q_learning import q_learning
from smart_trial.q_learning.features import build_H1, build_H2
from smart_trial.q_learning.config import STAGE1_ARMS, STAGE2_ARMS
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger


def _diag_correct(enc: Dict[str, Any]) -> Optional[bool]:
    outcome = enc.get("outcome")
    if not isinstance(outcome, dict):
        return None
    val = outcome.get("diag_correct")
    if val is None:
        return None
    return bool(val)


def _mean_correctness(encounters: List[Dict[str, Any]]) -> float:
    vals = [_diag_correct(e) for e in encounters]
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _filter_benchmark(encounters: List[Dict[str, Any]], case_ids: List[str]) -> List[Dict[str, Any]]:
    allowed = set(case_ids)
    return [e for e in encounters if e.get("case_id") in allowed]


def compute_baseline_metrics(
    baseline_path: Path,
    case_ids: List[str],
) -> Dict[str, Any]:
    rows = _filter_benchmark(TrajectoryLogger.load_aggregate(str(baseline_path)), case_ids)
    rows = [r for r in rows if r.get("run_mode") == "baseline"]
    return {
        "metric": "baseline",
        "strategy": "baseline",
        "n": len(rows),
        "avg_correctness": _mean_correctness(rows),
    }


def compute_grid_path_metrics(
    grid_path: Path,
    case_ids: List[str],
) -> List[Dict[str, Any]]:
    rows = _filter_benchmark(TrajectoryLogger.load_aggregate(str(grid_path)), case_ids)
    rows = [r for r in rows if r.get("run_mode") == "smart_grid"]
    out: List[Dict[str, Any]] = []
    for a1, a2 in GRID_PATHS:
        pid = path_id_for(a1, a2)
        subset = [
            r for r in rows
            if r.get("path_id") == pid
            or (r.get("forced_a1") == a1 and r.get("forced_a2") == a2)
        ]
        out.append({
            "metric": "static_path",
            "strategy": f"{a1}->{a2}",
            "path_id": pid,
            "forced_a1": a1,
            "forced_a2": a2,
            "n": len(subset),
            "avg_correctness": _mean_correctness(subset),
        })
    return out


def _policy_for_case(result: q_learning.QLearningResult, case_df: pd.DataFrame) -> tuple[str, str]:
    """Per-case π̂: argmax stage-1, then stage-2 at H2(pi1) using observed R1 under pi1."""
    h1 = build_H1(case_df.iloc[[0]])
    q1_vals = {a: float(result.stage1.predict_for_arm(h1, a)[0]) for a in STAGE1_ARMS}
    pi1 = max(q1_vals, key=q1_vals.get)

    a1_rows = case_df[case_df["A1"] == pi1]
    ref_row = a1_rows.iloc[[0]] if not a1_rows.empty else case_df.iloc[[0]]
    h2 = build_H2(ref_row, a1_override=pi1)
    q2_vals = {a: float(result.stage2.predict_for_arm(h2, a)[0]) for a in STAGE2_ARMS}
    pi2 = max(q2_vals, key=q2_vals.get)
    return pi1, pi2


def compute_adaptive_metrics(
    grid_path: Path,
    case_ids: List[str],
) -> Dict[str, Any]:
    encounters = _filter_benchmark(TrajectoryLogger.load_aggregate(str(grid_path)), case_ids)
    encounters = [e for e in encounters if e.get("run_mode") == "smart_grid"]
    df = ql_load.to_dataframe(encounters)
    if df.empty:
        return {
            "metric": "qlearning_adaptive",
            "strategy": "pi_hat",
            "n": 0,
            "avg_correctness": float("nan"),
            "error": "no grid rows for benchmark cohort",
        }

    result = None
    try:
        result = q_learning.fit(df)
    except ValueError as exc:
        return {
            "metric": "qlearning_adaptive",
            "strategy": "pi_hat",
            "n": 0,
            "avg_correctness": float("nan"),
            "V_hat_in_sample": float("nan"),
            "cases_missing_path": len(case_ids),
            "error": str(exc),
        }

    by_case_path: Dict[tuple, bool] = {}
    for enc in encounters:
        key = (enc["case_id"], enc.get("stage1_arm"), enc.get("stage2_arm"))
        y = _diag_correct(enc)
        if y is not None:
            by_case_path[key] = y

    hits: List[bool] = []
    missing = 0
    for cid in case_ids:
        case_df = df[df["case_id"] == cid]
        if case_df.empty:
            missing += 1
            continue
        pi1, pi2 = _policy_for_case(result, case_df)
        key = (cid, pi1, pi2)
        if key in by_case_path:
            hits.append(by_case_path[key])
        else:
            missing += 1

    return {
        "metric": "qlearning_adaptive",
        "strategy": "pi_hat",
        "n": len(hits),
        "avg_correctness": float(sum(hits) / len(hits)) if hits else float("nan"),
        "V_hat_in_sample": result.value,
        "cases_missing_path": missing,
    }


def build_summary(
    baseline_path: Optional[Path],
    grid_path: Path,
    case_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    cohort = case_ids or BENCHMARK_CASE_IDS
    rows: List[Dict[str, Any]] = []

    if baseline_path and baseline_path.is_file():
        rows.append(compute_baseline_metrics(baseline_path, cohort))
    else:
        rows.append({
            "metric": "baseline",
            "strategy": "baseline",
            "n": 0,
            "avg_correctness": float("nan"),
            "error": "baseline file missing",
        })

    rows.extend(compute_grid_path_metrics(grid_path, cohort))
    rows.append(compute_adaptive_metrics(grid_path, cohort))
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark eval correctness metrics")
    parser.add_argument(
        "--grid",
        type=Path,
        required=True,
        help="Path to grid_encounters.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline_encounters.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("smart_trial/outputs/eval/summary_metrics.csv"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON summary path",
    )
    args = parser.parse_args(argv)

    summary = build_summary(args.baseline, args.grid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote {args.out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": summary[summary["metric"] == "baseline"].to_dict(orient="records"),
            "static_paths": summary[summary["metric"] == "static_path"].to_dict(orient="records"),
            "adaptive": summary[summary["metric"] == "qlearning_adaptive"].to_dict(orient="records"),
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
