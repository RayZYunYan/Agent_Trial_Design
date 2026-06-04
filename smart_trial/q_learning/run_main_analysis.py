"""End-to-end Q-learning analysis entry point.

Usage:
    python -m smart_trial.q_learning.run_main_analysis \
        [--encounters smart_trial/outputs/encounters.jsonl] \
        [--out smart_trial/q_learning/outputs] \
        [--n-boot 200]

Produces under --out:
    rules.csv, rule_summary.csv, coefficients.csv
    static_vs_adaptive.csv, value_ci.csv, diagnostics.csv, arm_cells.csv
    qa_agreement.csv (Q-learning vs A-learning)
    interaction_plot.png, value_comparison.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import a_learning, bootstrap, consistency, diagnostics, figures, load, q_learning, rules
from .config import DEFAULT_ENCOUNTERS_DIR, DEFAULT_ENCOUNTERS_FILE, DEFAULT_RESULTS_DIR


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--encounters",
        type=Path,
        default=None,
        help="Encounters directory or single .jsonl file (default: auto-detect)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    enc_source = args.encounters
    if enc_source is None:
        enc_source = DEFAULT_ENCOUNTERS_FILE if DEFAULT_ENCOUNTERS_FILE.is_file() else DEFAULT_ENCOUNTERS_DIR

    df = load.load(enc_source)
    if df.empty:
        print(f"No encounters found at {enc_source}", file=sys.stderr)
        return 1
    print(f"Loaded {len(df)} encounters from {enc_source}")

    result = q_learning.fit(df)
    value_cf, _ = q_learning.cross_fitted_value(df, seed=args.seed)
    result.value_crossfit = value_cf
    print(f"Estimated V(pi_hat) in-sample = {result.value:.4f}")
    print(f"Estimated V(pi_hat) cross-fitted = {value_cf:.4f}")

    # ---- Tables ----
    result.rules.to_csv(args.out / "rules.csv", index=False)
    rules.summarise_rules(result).to_csv(args.out / "rule_summary.csv", index=False)
    rules.coefficient_table(result).to_csv(args.out / "coefficients.csv", index=False)

    static_table = rules.adaptive_vs_static_value(df, value_cf, fitted=result)
    static_table.to_csv(args.out / "static_vs_adaptive.csv", index=False)

    diagnostics.sample_size_report(df, result).to_csv(args.out / "diagnostics.csv", index=False)
    diagnostics.arm_cell_table(df).to_csv(args.out / "arm_cells.csv", index=False)

    # ---- Q vs A agreement ----
    try:
        a_res = a_learning.fit(df)
        cmp_df = consistency.compare_rules(result.rules, a_res.rules)
        consistency.agreement_summary(result.rules, a_res.rules).to_csv(
            args.out / "qa_agreement_summary.csv",
        )
        cmp_df.to_csv(args.out / "qa_agreement.csv", index=False)
        print(f"Q/A agree_all = {cmp_df['agree_all'].mean():.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"A-learning skipped: {exc}", file=sys.stderr)

    # ---- Inference ----
    point_mn, ci_mn, _ = bootstrap.m_out_of_n_ci(
        df, bootstrap.optimal_dtr_value, n_boot=args.n_boot, seed=args.seed,
    )
    _, ci_cms, _ = bootstrap.cms_projection_ci(
        df, bootstrap.optimal_dtr_value, n_boot=args.n_boot, seed=args.seed + 1,
    )
    point_cf, ci_cf, _ = bootstrap.percentile_ci(
        df, bootstrap.cross_fitted_dtr_value, n_boot=args.n_boot, seed=args.seed + 2,
    )
    pd.DataFrame([
        {"method": "in_sample", "point": result.value, "lo": float("nan"), "hi": float("nan")},
        {"method": "cross_fitted", "point": point_cf, "lo": ci_cf[0], "hi": ci_cf[1]},
        {"method": "m_out_of_n", "point": point_mn, "lo": ci_mn[0], "hi": ci_mn[1]},
        {"method": "cms_projection", "point": result.value, "lo": ci_cms[0], "hi": ci_cms[1]},
    ]).to_csv(args.out / "value_ci.csv", index=False)

    print(json.dumps({
        "n": int(len(df)),
        "V_hat_in_sample": result.value,
        "V_hat_crossfit": value_cf,
        "ci_crossfit": list(ci_cf),
        "ci_m_out_of_n": list(ci_mn),
        "ci_cms_projection": list(ci_cms),
    }, indent=2))

    figures.interaction_plot(df, result, out_path=args.out / "interaction_plot.png")
    figures.value_comparison_bar(
        static_table, value_cf, out_path=args.out / "value_comparison.png",
    )

    print(f"Wrote results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
