"""CLI: run one or more SMART encounters (see also repo-root `run_encounter.py`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    from smart_trial.core.orchestrator import TrialOrchestrator
    from smart_trial.data.loader import (
        apply_red_flag_cache,
        get_case_by_id,
        load_cases_from_config,
        load_red_flag_cache,
        pick_diverse_cases,
    )
    from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger

    parser = argparse.ArgumentParser(description="Run SMART trial encounter(s)")
    parser.add_argument("--config", default=None, help="Path to trial_config.yaml")
    parser.add_argument("--case_id", default=None, help="Run a single case by case_id")
    parser.add_argument("--seed", type=int, default=None, help="Override randomization seed")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Run only the first n cases in file order (default: all cases in the dataset)",
    )
    parser.add_argument(
        "--diverse",
        type=int,
        default=None,
        metavar="K",
        help="Run K cases with distinct case_category values (overrides default all/n)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if case_id is already in the output JSONL",
    )
    parser.add_argument(
        "--red-flag-cache",
        default=None,
        help="Optional JSON path mapping case_id -> red_flags (overrides config data.red_flag_cache)",
    )
    args = parser.parse_args(argv)

    orch = TrialOrchestrator(config_path=args.config)
    with open(orch.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cases = load_cases_from_config(cfg)
    if args.red_flag_cache:
        cache = load_red_flag_cache(args.red_flag_cache)
        apply_red_flag_cache(cases, cache)

    if args.case_id:
        case = get_case_by_id(cases, args.case_id)
        if case is None:
            raise SystemExit(f"case_id not found in loaded cases: {args.case_id}")
        logged = TrajectoryLogger.load_logged_case_ids(str(orch.output_path))
        if case["case_id"] in logged and not args.force:
            print(f"Skip {case['case_id']} (already in {orch.output_path}); use --force to re-run.")
            return
        orch.run_encounter(case, seed=args.seed)
        return

    if args.diverse is not None:
        run_cases = pick_diverse_cases(cases, n=max(1, args.diverse))
        if not run_cases:
            raise SystemExit("No cases available for --diverse run.")
    elif args.n is not None:
        run_cases = cases[: max(1, args.n)]
    else:
        run_cases = cases

    logged = set() if args.force else TrajectoryLogger.load_logged_case_ids(str(orch.output_path))
    skipped = 0
    ran = 0
    total = len(run_cases)

    print(f"Output: {orch.output_path}")
    print(f"Cases to process: {total} (dataset has {len(cases)}; skip if already logged)\n")

    for idx, case in enumerate(run_cases, start=1):
        case_id = case["case_id"]
        if case_id in logged:
            skipped += 1
            print(f"[{idx}/{total}] Skip {case_id} (already logged)")
            continue
        print(f"[{idx}/{total}] Running {case_id} ...")
        orch.run_encounter(case, seed=args.seed)
        logged.add(case_id)
        ran += 1

    print(f"\nDone. Ran {ran}, skipped {skipped}, output: {orch.output_path}")


if __name__ == "__main__":
    main()
