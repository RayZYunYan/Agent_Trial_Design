"""Result figures for the Q-learning analysis — plan §5.

This file ships one working example (interaction_plot) so the intern has a
template to copy when adding the others. matplotlib is imported lazily so
test collection doesn't break in headless environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .features import build_H2
from .q_learning import QLearningResult


def interaction_plot(
    df: pd.DataFrame,
    result: QLearningResult,
    fixed_a1: str = "A1b",
    out_path: Optional[Path] = None,
):
    """Plan §5 punchline figure: holding A1 fixed, how does the *best* Stage-2 arm
    change as R1 sweeps from 0 to 10?

    We sweep R1_total across the observed range, hold all other H2 features at
    their sample means (or modes for binary), and plot Q̂2(H2, a) for each
    Stage-2 arm. The crossing of curves is the interaction the paper hangs on.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H2 = build_H2(df)
    # Build a synthetic state grid.
    grid_n = 21
    r1_grid = np.linspace(0, 10, grid_n)
    base = H2.mean(numeric_only=True)
    rows = []
    for r1 in r1_grid:
        row = base.copy()
        row["R1_total"] = r1
        row["R1_responder"] = 1 if r1 >= 6 else 0
        # Force the fixed A1.
        row["A1_a"] = int(fixed_a1 == "A1a")
        row["A1_b"] = int(fixed_a1 == "A1b")
        row["A1_c"] = int(fixed_a1 == "A1c")
        rows.append(row)
    grid = pd.DataFrame(rows)
    grid = grid.reindex(columns=H2.columns, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in result.stage2.arm_levels:
        q_vals = result.stage2.predict_for_arm(grid, arm)
        ax.plot(r1_grid, q_vals, marker="o", label=arm)
    ax.set_xlabel("R1 total (0–10)")
    ax.set_ylabel("Estimated Q̂2(H2, A2)")
    ax.set_title(f"Stage-2 Q-function vs R1  |  A1 fixed at {fixed_a1}")
    ax.axvline(6, linestyle="--", alpha=0.4, label="responder threshold")
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig


def value_comparison_bar(
    static_vs_adaptive: pd.DataFrame,
    adaptive_value: float,
    top_k: int = 5,
    out_path: Optional[Path] = None,
):
    """Bar chart of adaptive DTR vs top-k static strategies — plan §5 second figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = static_vs_adaptive.head(top_k)
    labels = ["adaptive π̂"] + list(top["strategy"])
    values = [adaptive_value] + list(top["value"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#cc3333"] + ["#3366cc"] * len(top)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Estimated value (P(diag_correct))")
    ax.set_title("Adaptive DTR vs best static strategies")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig
