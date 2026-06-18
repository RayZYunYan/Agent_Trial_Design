"""Batch runner for benchmark eval: baseline, grid, and closed-loop adaptive."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SMART_TRIAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SMART_TRIAL_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from dotenv import load_dotenv

from smart_trial.core.orchestrator import TrialOrchestrator, resolve_path
from smart_trial.data.loader import (
    apply_red_flag_cache,
    filter_cases_by_ids,
    load_cases_from_config,
    load_red_flag_cache,
)
from smart_trial.eval.adaptive_loop import resolve_closed_loop_case_ids, run_adaptive_loop
from smart_trial.eval.case_lists import BENCHMARK_CASE_IDS, GRID_PATHS, path_id_for
from smart_trial.eval.output_utils import clear_eval_output_dir
from smart_trial.eval.resume import load_completed_baseline_keys, load_completed_grid_keys


def _resolve_case_ids(cfg: dict) -> list[str]:
    eval_cfg = cfg.get("eval") or {}
    explicit = eval_cfg.get("case_ids")
    if explicit:
        return list(explicit)
    return list(BENCHMARK_CASE_IDS)


def _resolve_grid_paths(cfg: dict) -> list[tuple[str, str]]:
    """Full 3×3 grid unless eval.grid_paths lists explicit [a1, a2] pairs."""
    eval_cfg = cfg.get("eval") or {}
    raw = eval_cfg.get("grid_paths")
    if not raw:
        return list(GRID_PATHS)
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
    return out or list(GRID_PATHS)


def _aggregate_path(orch: TrialOrchestrator) -> Path:
    name = orch.config.get("logging", {}).get("aggregate_filename")
    if not name:
        raise SystemExit("logging.aggregate_filename is required for eval batch runs.")
    return orch._output_dir / name


def run_baseline(orch: TrialOrchestrator, cases: list[dict], seed: int, resume: bool) -> int:
    agg = _aggregate_path(orch)
    done = load_completed_baseline_keys(agg) if resume else set()
    ran = 0
    for case in cases:
        cid = case["case_id"]
        if cid in done:
            print(f"[skip] baseline already done: {cid}")
            continue
        orch.run_encounter(case, seed=seed)
        ran += 1
    return ran


def run_grid(
    orch: TrialOrchestrator,
    cases: list[dict],
    seed: int,
    resume: bool,
    grid_paths: list[tuple[str, str]],
) -> int:
    agg = _aggregate_path(orch)
    done = load_completed_grid_keys(agg) if resume else set()
    ran = 0
    for case in cases:
        cid = case["case_id"]
        for a1, a2 in grid_paths:
            pid = path_id_for(a1, a2)
            if (cid, pid) in done:
                print(f"[skip] grid already done: {cid} {pid}")
                continue
            orch.run_encounter(
                case,
                seed=seed,
                forced_a1=a1,
                forced_a2=a2,
                path_id=pid,
                run_mode="smart_grid",
            )
            ran += 1
    return ran


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run benchmark eval batch (baseline or grid)")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip (case_id) or (case_id, path_id) already in aggregate JSONL",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override randomization seed")
    args = parser.parse_args(argv)

    orch = TrialOrchestrator(config_path=args.config)
    with open(orch.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cases = load_cases_from_config(cfg)
    cache_path = (cfg.get("data") or {}).get("red_flag_cache")
    if cache_path:
        resolved = resolve_path(str(cache_path))
        if resolved.is_file():
            apply_red_flag_cache(cases, load_red_flag_cache(str(resolved)))

    case_ids = (
        resolve_closed_loop_case_ids(cfg)
        if mode == "smart_adaptive_loop"
        else _resolve_case_ids(cfg)
    )
    run_cases = filter_cases_by_ids(cases, case_ids)
    if len(run_cases) != len(case_ids):
        missing = set(case_ids) - {c["case_id"] for c in run_cases}
        print(f"Warning: missing {len(missing)} case_ids: {sorted(missing)[:5]}...", file=sys.stderr)

    seed = args.seed if args.seed is not None else int(cfg.get("randomization", {}).get("seed", 42))
    mode = (cfg.get("run") or {}).get("mode", "smart_random")

    orch._output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        n_removed = clear_eval_output_dir(orch._output_dir)
        print(f"Cleared {n_removed} *.jsonl file(s) in {orch._output_dir}")

    print(f"Mode={mode} | cases={len(run_cases)} | seed={seed} | resume={args.resume}")
    print(f"Output dir: {orch._output_dir}")

    grid_paths = _resolve_grid_paths(cfg)

    if mode == "baseline":
        ran = run_baseline(orch, run_cases, seed, args.resume)
    elif mode == "smart_grid":
        print(f"Grid paths ({len(grid_paths)}): {[path_id_for(a, b) for a, b in grid_paths]}")
        ran = run_grid(orch, run_cases, seed, args.resume, grid_paths)
    elif mode == "smart_adaptive_loop":
        run_cfg = cfg.get("run") or {}
        print(
            f"Closed-loop adaptive | refit_every_n={run_cfg.get('refit_every_n', 5)} "
            f"| burn_in={run_cfg.get('burn_in', 0)} "
            f"| initial_q_from={run_cfg.get('initial_q_from')}"
        )
        ran = run_adaptive_loop(orch, run_cases, seed, args.resume)
    else:
        print(
            f"run_eval supports baseline, smart_grid, smart_adaptive_loop; got {mode!r}",
            file=sys.stderr,
        )
        return 1

    print(f"Finished. New encounters this run: {ran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
