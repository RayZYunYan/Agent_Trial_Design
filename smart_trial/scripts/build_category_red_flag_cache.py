"""
Build a deterministic red-flag cache from case category defaults (no API calls).

Usage (from repo root):
  python -m smart_trial.scripts.build_category_red_flag_cache
  python -m smart_trial.scripts.build_category_red_flag_cache --max-cases 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_trial.data.loader import load_cases_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to trial_config.yaml")
    parser.add_argument(
        "--out",
        default="smart_trial/data/red_flag_cache.json",
        help="Output JSON path (case_id -> list of red flag strings)",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "smart_trial" / "config" / "trial_config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cases = load_cases_from_config(cfg)
    if args.max_cases:
        cases = cases[: args.max_cases]

    cache = {case["case_id"]: list(case.get("red_flags") or []) for case in cases}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(cache)} entries to {out_path}")


if __name__ == "__main__":
    main()
