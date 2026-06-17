"""Helpers for eval output directories."""
from __future__ import annotations

from pathlib import Path


def clear_eval_output_dir(output_dir: Path) -> int:
    """Remove all *.jsonl under an eval output directory. Returns count removed."""
    if not output_dir.is_dir():
        return 0
    removed = 0
    for path in output_dir.glob("*.jsonl"):
        path.unlink()
        removed += 1
    return removed


def validate_grid_aggregate(
    aggregate_path: Path,
    *,
    expected_cases: list[str] | None = None,
    expected_paths_per_case: int = 9,
) -> dict:
    """Check grid_encounters.jsonl: unique (case_id, path_id) counts."""
    from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger

    rows = [r for r in TrajectoryLogger.load_aggregate(str(aggregate_path)) if r.get("run_mode") == "smart_grid"]
    by_case: dict[str, list[str]] = {}
    duplicates: list[tuple[str, str, int]] = []
    pair_counts: dict[tuple[str, str], int] = {}

    for r in rows:
        cid = str(r.get("case_id", ""))
        pid = str(r.get("path_id", ""))
        if not cid or not pid:
            continue
        by_case.setdefault(cid, []).append(pid)
        key = (cid, pid)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    for (cid, pid), n in pair_counts.items():
        if n > 1:
            duplicates.append((cid, pid, n))

    case_ids = expected_cases or sorted(by_case.keys())
    report = {
        "total_rows": len(rows),
        "expected_total": len(case_ids) * expected_paths_per_case,
        "cases": {},
        "duplicates": duplicates,
        "ok": True,
        "errors": [],
    }

    for cid in case_ids:
        paths = by_case.get(cid, [])
        unique = sorted(set(paths))
        report["cases"][cid] = {
            "rows": len(paths),
            "unique_paths": len(unique),
            "path_ids": unique,
        }
        if len(unique) != expected_paths_per_case:
            report["ok"] = False
            report["errors"].append(
                f"{cid}: expected {expected_paths_per_case} unique paths, got {len(unique)} (rows={len(paths)})"
            )
        if len(paths) != len(unique):
            report["ok"] = False
            report["errors"].append(f"{cid}: duplicate path_id in aggregate")

    if duplicates:
        report["ok"] = False
        report["errors"].append(f"duplicate (case_id, path_id) pairs: {duplicates}")

    if report["total_rows"] != report["expected_total"]:
        report["ok"] = False
        report["errors"].append(
            f"total rows {report['total_rows']} != expected {report['expected_total']}"
        )

    return report
