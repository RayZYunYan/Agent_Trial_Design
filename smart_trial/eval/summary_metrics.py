"""Compute average correctness: baseline, static paths, Q-learning adaptive."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_trial.eval.case_lists import (
    BENCHMARK_CASE_IDS,
    CASE_CATEGORIES,
    GRID_PATHS,
    path_id_for,
)
from smart_trial.q_learning import load as ql_load
from smart_trial.q_learning import q_learning
from smart_trial.q_learning.features import build_H1, build_H2
from smart_trial.q_learning.config import STAGE1_ARMS, STAGE2_ARMS
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger

SUMMARY_COLUMNS = [
    "metric",
    "strategy",
    "path_id",
    "forced_a1",
    "forced_a2",
    "case_category",
    "n",
    "avg_correctness",
    "V_hat_in_sample",
    "cases_missing_path",
    "error",
]


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


def _normalize_category(category: Optional[str]) -> str:
    if category in CASE_CATEGORIES:
        return category
    return "Other"


def _filter_benchmark(encounters: List[Dict[str, Any]], case_ids: List[str]) -> List[Dict[str, Any]]:
    allowed = set(case_ids)
    return [e for e in encounters if e.get("case_id") in allowed]


def _load_baseline_rows(baseline_path: Optional[Path], case_ids: List[str]) -> List[Dict[str, Any]]:
    if not baseline_path or not baseline_path.is_file():
        return []
    rows = _filter_benchmark(TrajectoryLogger.load_aggregate(str(baseline_path)), case_ids)
    return [r for r in rows if r.get("run_mode") == "baseline"]


def _load_grid_rows(grid_path: Path, case_ids: List[str]) -> List[Dict[str, Any]]:
    rows = _filter_benchmark(TrajectoryLogger.load_aggregate(str(grid_path)), case_ids)
    return [r for r in rows if r.get("run_mode") == "smart_grid"]


def _case_category_map(encounters: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for enc in encounters:
        cid = enc.get("case_id")
        if cid and cid not in mapping:
            mapping[cid] = _normalize_category(enc.get("case_category"))
    return mapping


def _rows_for_path(rows: List[Dict[str, Any]], a1: str, a2: str) -> List[Dict[str, Any]]:
    pid = path_id_for(a1, a2)
    return [
        r for r in rows
        if r.get("path_id") == pid
        or (r.get("forced_a1") == a1 and r.get("forced_a2") == a2)
    ]


def compute_baseline_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "metric": "baseline",
        "strategy": "baseline",
        "n": len(rows),
        "avg_correctness": _mean_correctness(rows),
    }


def compute_baseline_by_category(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cat in CASE_CATEGORIES:
        subset = [r for r in rows if _normalize_category(r.get("case_category")) == cat]
        out.append({
            "metric": "category_baseline",
            "strategy": "baseline",
            "case_category": cat,
            "n": len(subset),
            "avg_correctness": _mean_correctness(subset),
        })
    return out


def compute_grid_path_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a1, a2 in GRID_PATHS:
        pid = path_id_for(a1, a2)
        subset = _rows_for_path(rows, a1, a2)
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


def compute_grid_path_by_category(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a1, a2 in GRID_PATHS:
        pid = path_id_for(a1, a2)
        path_rows = _rows_for_path(rows, a1, a2)
        for cat in CASE_CATEGORIES:
            subset = [r for r in path_rows if _normalize_category(r.get("case_category")) == cat]
            out.append({
                "metric": "category_static_path",
                "strategy": f"{a1}->{a2}",
                "path_id": pid,
                "forced_a1": a1,
                "forced_a2": a2,
                "case_category": cat,
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


def _build_by_case_path(encounters: List[Dict[str, Any]]) -> Dict[tuple, bool]:
    by_case_path: Dict[tuple, bool] = {}
    for enc in encounters:
        key = (enc["case_id"], enc.get("stage1_arm"), enc.get("stage2_arm"))
        y = _diag_correct(enc)
        if y is not None:
            by_case_path[key] = y
    return by_case_path


def _adaptive_hits_for_cases(
    result: q_learning.QLearningResult,
    df: pd.DataFrame,
    case_ids: List[str],
    by_case_path: Dict[tuple, bool],
) -> Tuple[List[bool], int]:
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
    return hits, missing


def _fit_adaptive(grid_rows: List[Dict[str, Any]]) -> Tuple[
    Optional[q_learning.QLearningResult],
    pd.DataFrame,
    Optional[str],
]:
    df = ql_load.to_dataframe(grid_rows)
    if df.empty:
        return None, df, "no grid rows for benchmark cohort"
    try:
        return q_learning.fit(df), df, None
    except ValueError as exc:
        return None, df, str(exc)


def compute_adaptive_metrics(
    grid_rows: List[Dict[str, Any]],
    case_ids: List[str],
    *,
    result: Optional[q_learning.QLearningResult] = None,
    df: Optional[pd.DataFrame] = None,
    fit_error: Optional[str] = None,
) -> Dict[str, Any]:
    if result is None and fit_error is None:
        result, df, fit_error = _fit_adaptive(grid_rows)
    if fit_error:
        row: Dict[str, Any] = {
            "metric": "qlearning_adaptive",
            "strategy": "pi_hat",
            "n": 0,
            "avg_correctness": float("nan"),
            "error": fit_error,
        }
        if result is None and fit_error != "no grid rows for benchmark cohort":
            row["V_hat_in_sample"] = float("nan")
            row["cases_missing_path"] = len(case_ids)
        return row

    by_case_path = _build_by_case_path(grid_rows)
    hits, missing = _adaptive_hits_for_cases(result, df, case_ids, by_case_path)

    return {
        "metric": "qlearning_adaptive",
        "strategy": "pi_hat",
        "n": len(hits),
        "avg_correctness": float(sum(hits) / len(hits)) if hits else float("nan"),
        "V_hat_in_sample": result.value,
        "cases_missing_path": missing,
    }


def compute_adaptive_by_category(
    grid_rows: List[Dict[str, Any]],
    case_ids: List[str],
    *,
    result: Optional[q_learning.QLearningResult] = None,
    df: Optional[pd.DataFrame] = None,
    fit_error: Optional[str] = None,
    case_cat_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    case_cat_map = case_cat_map or _case_category_map(grid_rows)
    out: List[Dict[str, Any]] = []

    if fit_error or result is None or df is None or df.empty:
        err = fit_error or "adaptive fit unavailable"
        for cat in CASE_CATEGORIES:
            cat_cases = [cid for cid in case_ids if case_cat_map.get(cid) == cat]
            out.append({
                "metric": "category_adaptive",
                "strategy": "pi_hat",
                "case_category": cat,
                "n": 0,
                "avg_correctness": float("nan"),
                "cases_missing_path": len(cat_cases),
                "error": err,
            })
        return out

    by_case_path = _build_by_case_path(grid_rows)
    for cat in CASE_CATEGORIES:
        cat_cases = [cid for cid in case_ids if case_cat_map.get(cid) == cat]
        hits, missing = _adaptive_hits_for_cases(result, df, cat_cases, by_case_path)
        out.append({
            "metric": "category_adaptive",
            "strategy": "pi_hat",
            "case_category": cat,
            "n": len(hits),
            "avg_correctness": float(sum(hits) / len(hits)) if hits else float("nan"),
            "cases_missing_path": missing,
        })
    return out


def _load_closed_loop_rows(
    adaptive_path: Optional[Path],
    case_ids: List[str],
) -> List[Dict[str, Any]]:
    if not adaptive_path or not adaptive_path.is_file():
        return []
    rows = _filter_benchmark(TrajectoryLogger.load_aggregate(str(adaptive_path)), case_ids)
    return [r for r in rows if r.get("run_mode") == "smart_adaptive_loop"]


def compute_closed_loop_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "metric": "closed_loop_adaptive",
            "strategy": "pi_hat_online",
            "n": 0,
            "avg_correctness": float("nan"),
            "error": "adaptive file missing or empty",
        }
    return {
        "metric": "closed_loop_adaptive",
        "strategy": "pi_hat_online",
        "n": len(rows),
        "avg_correctness": _mean_correctness(rows),
    }


def compute_closed_loop_by_category(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cat in CASE_CATEGORIES:
        subset = [r for r in rows if _normalize_category(r.get("case_category")) == cat]
        out.append({
            "metric": "category_closed_loop_adaptive",
            "strategy": "pi_hat_online",
            "case_category": cat,
            "n": len(subset),
            "avg_correctness": _mean_correctness(subset),
        })
    return out


def write_category_jsonls(
    encounters: List[Dict[str, Any]],
    out_dir: Path,
    *,
    clear: bool = True,
) -> List[Path]:
    """Split encounters into one JSONL per case_category (skip empty categories)."""
    if clear and out_dir.is_dir():
        for path in out_dir.glob("*.jsonl"):
            path.unlink()

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for enc in encounters:
        by_cat[_normalize_category(enc.get("case_category"))].append(enc)

    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in CASE_CATEGORIES:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        path = out_dir / f"{cat}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for enc in rows:
                f.write(json.dumps(enc, ensure_ascii=False) + "\n")
        written.append(path)
    return written


def build_summary(
    baseline_path: Optional[Path],
    grid_path: Path,
    case_ids: Optional[List[str]] = None,
    *,
    adaptive_path: Optional[Path] = None,
    closed_loop_case_ids: Optional[List[str]] = None,
    write_category_jsonl: bool = False,
) -> pd.DataFrame:
    cohort = case_ids or BENCHMARK_CASE_IDS
    closed_cohort = closed_loop_case_ids or cohort
    baseline_rows = _load_baseline_rows(baseline_path, cohort)
    grid_rows = _load_grid_rows(grid_path, cohort)

    rows: List[Dict[str, Any]] = []

    if baseline_path and baseline_path.is_file():
        rows.append(compute_baseline_metrics(baseline_rows))
    else:
        rows.append({
            "metric": "baseline",
            "strategy": "baseline",
            "n": 0,
            "avg_correctness": float("nan"),
            "error": "baseline file missing",
        })

    rows.extend(compute_baseline_by_category(baseline_rows))
    rows.extend(compute_grid_path_metrics(grid_rows))
    rows.extend(compute_grid_path_by_category(grid_rows))

    result, df, fit_error = _fit_adaptive(grid_rows)
    case_cat_map = _case_category_map(baseline_rows + grid_rows)
    rows.append(
        compute_adaptive_metrics(
            grid_rows,
            cohort,
            result=result,
            df=df,
            fit_error=fit_error,
        )
    )
    rows.extend(
        compute_adaptive_by_category(
            grid_rows,
            cohort,
            result=result,
            df=df,
            fit_error=fit_error,
            case_cat_map=case_cat_map,
        )
    )

    closed_rows = _load_closed_loop_rows(adaptive_path, closed_cohort)
    rows.append(compute_closed_loop_metrics(closed_rows))
    rows.extend(compute_closed_loop_by_category(closed_rows))

    if write_category_jsonl:
        if baseline_path and baseline_path.is_file():
            write_category_jsonls(baseline_rows, baseline_path.parent / "by_category")
        if grid_path.is_file():
            write_category_jsonls(grid_rows, grid_path.parent / "by_category")
        if adaptive_path and adaptive_path.is_file() and closed_rows:
            write_category_jsonls(closed_rows, adaptive_path.parent / "by_category")

    summary = pd.DataFrame(rows)
    return summary.reindex(columns=SUMMARY_COLUMNS)


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
        "--adaptive",
        type=Path,
        default=None,
        help="Path to adaptive_encounters.jsonl (closed-loop Phase 2)",
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
    parser.add_argument(
        "--no-category-jsonl",
        action="store_true",
        help="Skip writing baseline/grid by_category/*.jsonl splits",
    )
    args = parser.parse_args(argv)

    from smart_trial.eval.case_lists import CLOSED_LOOP_CASE_IDS

    summary = build_summary(
        args.baseline,
        args.grid,
        adaptive_path=args.adaptive,
        closed_loop_case_ids=CLOSED_LOOP_CASE_IDS if args.adaptive else None,
        write_category_jsonl=not args.no_category_jsonl,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote {args.out}")

    if not args.no_category_jsonl:
        for label, agg in [("baseline", args.baseline), ("grid", args.grid)]:
            if agg and agg.is_file():
                cat_dir = agg.parent / "by_category"
                files = list(cat_dir.glob("*.jsonl")) if cat_dir.is_dir() else []
                if files:
                    print(f"Wrote {len(files)} {label} category JSONL(s) under {cat_dir}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": summary[summary["metric"] == "baseline"].to_dict(orient="records"),
            "static_paths": summary[summary["metric"] == "static_path"].to_dict(orient="records"),
            "adaptive": summary[summary["metric"] == "qlearning_adaptive"].to_dict(orient="records"),
            "closed_loop_adaptive": summary[
                summary["metric"] == "closed_loop_adaptive"
            ].to_dict(orient="records"),
            "by_category": {
                "baseline": summary[summary["metric"] == "category_baseline"].to_dict(orient="records"),
                "static_paths": summary[summary["metric"] == "category_static_path"].to_dict(orient="records"),
                "adaptive": summary[summary["metric"] == "category_adaptive"].to_dict(orient="records"),
                "closed_loop_adaptive": summary[
                    summary["metric"] == "category_closed_loop_adaptive"
                ].to_dict(orient="records"),
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
