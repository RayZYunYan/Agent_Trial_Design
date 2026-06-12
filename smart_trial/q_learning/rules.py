"""Extract human-readable optimal DTR rules from fitted Q-learning result.

Two ways the intern will want to read the rules out:
  1. summarise_rules(): "what does π̂ recommend on the observed data?"
     — grouped breakdown by (case_category, R1_responder, R2_level).
  2. coefficient_table(): raw fitted coefficients per stage, for the paper appendix.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .q_learning import QLearningResult


def summarise_rules(result: QLearningResult) -> pd.DataFrame:
    """For each subgroup, what did π̂ pick and what was the observed value?"""
    rules = result.rules.copy()
    grouped = rules.groupby(
        ["case_category", "R1_responder"], dropna=False
    ).agg(
        n=("encounter_id", "size"),
        modal_pi1=("pi1", _mode_str),
        modal_pi2=("pi2", _mode_str),
        Qhat_at_pi=("Qhat_at_pi", "mean"),
    ).reset_index()
    return grouped


def _mode_str(s: pd.Series) -> str:
    vals = s.dropna()
    if vals.empty:
        return ""
    return str(vals.value_counts().idxmax())


def _unwrap_linear(model):
    """Get a (coef, intercept) tuple if the model is linear-shaped, else None.

    Handles both bare sklearn estimators and Pipeline(..., linear) wrappers.
    """
    inner = model
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "coef_"):
                inner = step
                break
    if not hasattr(inner, "coef_"):
        return None
    coef = np.atleast_1d(inner.coef_).ravel()
    intercept = float(getattr(inner, "intercept_", 0.0))
    return coef, intercept


def coefficient_table(result: QLearningResult) -> pd.DataFrame:
    """Concatenate per-stage linear-model coefficients (if linear)."""
    rows = []
    for stage_name, reg in [("stage1", result.stage1), ("stage2", result.stage2)]:
        unwrapped = _unwrap_linear(reg.model)
        if unwrapped is None:
            continue
        coef, intercept = unwrapped
        for col, c in zip(reg._columns, coef):
            rows.append({"stage": stage_name, "term": col, "coef": float(c)})
        rows.append({"stage": stage_name, "term": "(intercept)", "coef": intercept})
    return pd.DataFrame(rows)


def adaptive_vs_static_value(
    df: pd.DataFrame,
    adaptive_value: float,
    fitted: Optional["QLearningResult"] = None,
) -> pd.DataFrame:
    """Compare adaptive DTR value vs every feasible static (a1,a2) — plan §5."""
    from itertools import product

    from .q_learning import QLearningResult, value_of_fixed_strategy
    from .config import STAGE1_ARMS, STAGE2_ARMS

    rows = []
    for a1, a2 in product(STAGE1_ARMS, STAGE2_ARMS):
        v = value_of_fixed_strategy(df, a1, a2, fitted=fitted)
        rows.append({
            "strategy": f"{a1}->{a2}",
            "value": v,
            "gap_vs_adaptive": adaptive_value - v if pd.notna(v) else float("nan"),
        })
    out = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    return out
