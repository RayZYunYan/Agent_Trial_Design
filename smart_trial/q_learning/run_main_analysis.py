"""End-to-end Q-learning analysis entry point.

Usage:
    python -m smart_trial.q_learning.run_main_analysis \
        [--encounters smart_trial/outputs/encounters] \
        [--out smart_trial/q_learning/outputs] \
        [--n-boot 200]

Produces under --out:
    rules.csv                 every-row optimal DTR per π̂
    rule_summary.csv          subgroup rule summary (plan §5 deliverable 1)
    coefficients.csv          fitted linear coefficients (plan appendix)
    static_vs_adaptive.csv    plan §5 deliverable 2
    value_ci.csv              point + percentile + m-of-n CIs for V(π̂)
    interaction_plot.png      plan §5 deliverable 3 (one example)
    value_comparison.png      adaptive vs top-5 static
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import bootstrap, figures, load, q_learning, rules
from .config import DEFAULT_ENCOUNTERS_DIR, DEFAULT_RESULTS_DIR


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encounters", type=Path, default=DEFAULT_ENCOUNTERS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    df = load.load(args.encounters)
    if df.empty:
        print(f"No encounters found under {args.encounters}", file=sys.stderr)
        return 1
    print(f"Loaded {len(df)} encounters from {args.encounters}")

    result = q_learning.fit(df)
    print(f"Estimated V(π̂) = {result.value:.4f}")

    # ---- Tables ----
    result.rules.to_csv(args.out / "rules.csv", index=False)
    rules.summarise_rules(result).to_csv(args.out / "rule_summary.csv", index=False)
    rules.coefficient_table(result).to_csv(args.out / "coefficients.csv", index=False)

    static_table = rules.adaptive_vs_static_value(df, result.value)
    static_table.to_csv(args.out / "static_vs_adaptive.csv", index=False)

    # ---- Inference ----
    point_pct, ci_pct, _ = bootstrap.percentile_ci(
        df, bootstrap.optimal_dtr_value, n_boot=args.n_boot, seed=args.seed,
    )
    point_mn, ci_mn, _ = bootstrap.m_out_of_n_ci(
        df, bootstrap.optimal_dtr_value, n_boot=args.n_boot, seed=args.seed,
    )
    pd.DataFrame([
        {"method": "percentile", "point": point_pct, "lo": ci_pct[0], "hi": ci_pct[1]},
        {"method": "m_out_of_n", "point": point_mn, "lo": ci_mn[0], "hi": ci_mn[1]},
    ]).to_csv(args.out / "value_ci.csv", index=False)

    print(json.dumps({
        "n": int(len(df)),
        "V_hat": result.value,
        "ci_percentile": list(ci_pct),
        "ci_m_out_of_n": list(ci_mn),
    }, indent=2))

    # ---- Figures ----
    figures.interaction_plot(df, result, out_path=args.out / "interaction_plot.png")
    figures.value_comparison_bar(
        static_table, result.value, out_path=args.out / "value_comparison.png",
    )

    print(f"Wrote results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
