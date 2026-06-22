"""Sample-size and design-matrix diagnostics for Q-learning fits."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .q_learning import QLearningResult


def sample_size_report(df: pd.DataFrame, result: Optional[QLearningResult] = None) -> pd.DataFrame:
    """Summarise n, parameter counts, arm cells, and warnings."""
    n = len(df)
    rows = [{"metric": "n_encounters", "value": n}]

    if result is not None:
        for stage_name, reg in [
            ("stage1", result.stage1),
            ("stage2", result.stage2),
        ]:
            n_params = reg.n_parameters
            rows.append({"metric": f"{stage_name}_n_design_cols", "value": n_params})
            rows.append({
                "metric": f"{stage_name}_n_over_p",
                "value": round(n / max(n_params, 1), 3),
            })

    arm_cell = df.groupby(["A1", "A2"], dropna=False).size().reset_index(name="count")
    rows.append({"metric": "n_arm_cells", "value": len(arm_cell)})
    rows.append({"metric": "min_cell_count", "value": int(arm_cell["count"].min()) if len(arm_cell) else 0})
    rows.append({"metric": "mean_Y", "value": round(float(df["Y"].mean()), 4)})

    if result is not None and n < result.stage2.n_parameters:
        rows.append({"metric": "warning_high_dimensional", "value": 1})
    else:
        rows.append({"metric": "warning_high_dimensional", "value": 0})

    return pd.DataFrame(rows)


def arm_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of observed (A1, A2) with outcome rate."""
    g = df.groupby(["A1", "A2"], dropna=False).agg(
        n=("Y", "size"),
        mean_Y=("Y", "mean"),
    ).reset_index()
    return g.sort_values("n", ascending=False)
