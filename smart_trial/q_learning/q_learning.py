"""Backward induction Q-learning for the two-stage SMART.

Stage 2 fits a logistic Q on binary Y; stage 1 regresses the stage-2
pseudo-outcome. Counterfactual H2(a1) is used when evaluating stage-1 arms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_RIDGE_ALPHA, REFERENCE_ARMS, STAGE1_ARMS, STAGE2_ARMS
from .features import build_H1, build_H2
from .models import StageRegressor
from .pools import r1_pool_for


def _argmax_over_pool(
    q_by_arm: Dict[str, np.ndarray],
    pools: List[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(pools)
    best_arm = np.empty(n, dtype=object)
    best_q = np.full(n, -np.inf)
    for i, pool in enumerate(pools):
        for a in pool:
            if a not in q_by_arm:
                continue
            qi = q_by_arm[a][i]
            if qi > best_q[i]:
                best_q[i] = qi
                best_arm[i] = a
    return best_arm, best_q


def _stage2_optimal(
    q2: StageRegressor,
    df: pd.DataFrame,
    *,
    a1: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Best stage-2 arm and value at counterfactual a1 or observed path."""
    H2 = build_H2(df, a1_override=a1)
    pools = [r1_pool_for(row) for _, row in df.iterrows()]
    q_by_arm = {a: q2.predict_for_arm(H2, a) for a in q2.arm_levels}
    return _argmax_over_pool(q_by_arm, pools)


def _best_stage1_via_q2(q2: StageRegressor, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Counterfactual stage-1 decision: max over (a1, a2) via Q2 at H2(a1)."""
    n = len(df)
    best_arm = np.empty(n, dtype=object)
    best_q = np.full(n, -np.inf)
    for a1 in STAGE1_ARMS:
        _, q_vals = _stage2_optimal(q2, df, a1=a1)
        for i in range(n):
            if q_vals[i] > best_q[i]:
                best_q[i] = q_vals[i]
                best_arm[i] = a1
    return best_arm, best_q


@dataclass
class QLearningResult:
    stage2: StageRegressor
    stage1: StageRegressor
    rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    value: float = float("nan")
    value_crossfit: float = float("nan")
    pseudo_y1: np.ndarray = field(default_factory=lambda: np.array([]))
    pseudo_y0: np.ndarray = field(default_factory=lambda: np.array([]))


def fit(
    df: pd.DataFrame,
    *,
    stage2_model: Optional[StageRegressor] = None,
    stage1_model: Optional[StageRegressor] = None,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    use_logistic_stage2: bool = True,
) -> QLearningResult:
    """Run backward induction with counterfactual decisions at stage 1."""
    if df.empty:
        raise ValueError("Empty trajectory DataFrame; nothing to fit.")

    H1 = build_H1(df)
    H2 = build_H2(df)
    y = df["Y"].to_numpy(dtype=float)

    q2 = stage2_model or StageRegressor(
        arm_levels=list(STAGE2_ARMS),
        reference_arm=REFERENCE_ARMS["stage2"],
        binary_outcome=use_logistic_stage2,
        ridge_alpha=ridge_alpha,
    )
    q2.fit(H2, df["A2"], y)

    # Stage 2 decision (observed H2) + regression target for stage 1.
    best_a2, pseudo_y1 = _stage2_optimal(q2, df)

    # Stage 1: counterfactual argmax via Q2 at H2(a1).
    best_a1, pseudo_y0 = _best_stage1_via_q2(q2, df)

    q1 = stage1_model or StageRegressor(
        arm_levels=STAGE1_ARMS,
        reference_arm=REFERENCE_ARMS["stage1"],
        ridge_alpha=ridge_alpha,
    )
    q1.fit(H1, df["A1"], pseudo_y1)

    rules = pd.DataFrame({
        "encounter_id": df["encounter_id"].values,
        "case_id": df["case_id"].values,
        "case_category": df["case_category"].values,
        "R1_responder": df["R1_responder"].values,
        # R2 is post-treatment in the two-stage design: descriptive only.
        "R2_level": df["R2_level"].values,
        "observed_A1": df["A1"].values,
        "observed_A2": df["A2"].values,
        "pi1": best_a1,
        "pi2": best_a2,
        "Qhat_at_pi": pseudo_y0,
    })

    return QLearningResult(
        stage2=q2,
        stage1=q1,
        rules=rules,
        value=float(np.mean(pseudo_y0)),
        pseudo_y1=pseudo_y1,
        pseudo_y0=pseudo_y0,
    )


def value_of_fixed_strategy(
    df: pd.DataFrame,
    a1: str,
    a2: str,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    use_logistic_stage2: bool = True,
    fitted: Optional[QLearningResult] = None,
) -> float:
    """Plug-in value of static (a1, a2) via Q2 at counterfactual H2(a1)."""
    if df.empty:
        return float("nan")

    q2 = fitted.stage2 if fitted is not None else StageRegressor(
        arm_levels=list(STAGE2_ARMS),
        reference_arm=REFERENCE_ARMS["stage2"],
        binary_outcome=use_logistic_stage2,
        ridge_alpha=ridge_alpha,
    )
    if fitted is None:
        H2 = build_H2(df)
        q2.fit(H2, df["A2"], df["Y"].to_numpy(dtype=float))

    H2_cf = build_H2(df, a1_override=a1)
    q2_pred = q2.predict_for_arm(H2_cf, a2)
    feasible: List[float] = []
    for i, row in df.reset_index(drop=True).iterrows():
        if a2 in r1_pool_for(row):
            feasible.append(float(q2_pred[i]))
    if not feasible:
        return float("nan")
    return float(np.mean(feasible))


def cross_fitted_value(
    df: pd.DataFrame,
    n_folds: int = 5,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    use_logistic_stage2: bool = True,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """K-fold cross-fitted plug-in value of π̂ (reduces in-sample optimism)."""
    n = len(df)
    if n < 2:
        res = fit(df, ridge_alpha=ridge_alpha, use_logistic_stage2=use_logistic_stage2)
        return res.value, res.pseudo_y0

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_ids = np.array_split(indices, min(n_folds, n))
    oof_values = np.full(n, np.nan)

    for fold in fold_ids:
        test_idx = fold
        train_idx = np.setdiff1d(indices, test_idx)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        res = fit(train_df, ridge_alpha=ridge_alpha, use_logistic_stage2=use_logistic_stage2)
        # Evaluate trained Q2 policy on test rows.
        _, test_vals = _best_stage1_via_q2(res.stage2, test_df)
        for j, orig_i in enumerate(test_idx):
            oof_values[orig_i] = test_vals[j]

    return float(np.nanmean(oof_values)), oof_values
