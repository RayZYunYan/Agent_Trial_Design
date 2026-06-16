"""Resume helpers for eval batch runs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set, Tuple

from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger


def load_completed_baseline_keys(aggregate_path: str | Path) -> Set[str]:
    """Return case_ids already present in a baseline aggregate file."""
    done: Set[str] = set()
    for enc in TrajectoryLogger.load_aggregate(str(aggregate_path)):
        if enc.get("run_mode") == "baseline" and enc.get("case_id"):
            done.add(str(enc["case_id"]))
    return done


def load_completed_grid_keys(aggregate_path: str | Path) -> Set[Tuple[str, str]]:
    """Return (case_id, path_id) pairs already present in a grid aggregate file."""
    done: Set[Tuple[str, str]] = set()
    for enc in TrajectoryLogger.load_aggregate(str(aggregate_path)):
        if enc.get("run_mode") != "smart_grid":
            continue
        case_id = enc.get("case_id")
        path_id = enc.get("path_id")
        if case_id and path_id:
            done.add((str(case_id), str(path_id)))
    return done


def count_completed(aggregate_path: str | Path, run_mode: str) -> int:
    return sum(
        1
        for enc in TrajectoryLogger.load_aggregate(str(aggregate_path))
        if enc.get("run_mode") == run_mode
    )
