"""CLI: run one or more SMART encounters (see also repo-root `run_encounter.py`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from smart_trial.core.orchestrator import TrialOrchestrator
    from smart_trial.data.loader import apply_red_flag_cache, get_case_by_id, load_cases_from_config, load_red_flag_cache

    parser = argparse.ArgumentParser(description="Run SMART trial encounter(s)")
    parser.add_argument("--config", default=None, help="Path to trial_config.yaml")
    parser.add_argument("--case_id", default=None, help="Run a single case by case_id")
    parser.add_argument("--seed", type=int, default=None, help="Override randomization seed")
    parser.add_argument("--n", type=int, default=1, help="Run first n cases (ignored if --case_id set)")
    parser.add_argument(
        "--red-flag-cache",
        default=None,
        help="Optional JSON dict path mapping case_id -> red_flags list",
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
        orch.run_encounter(case, seed=args.seed)
        return

    n = max(1, args.n)
    for case in cases[:n]:
        orch.run_encounter(case, seed=args.seed)


if __name__ == "__main__":
    main()
