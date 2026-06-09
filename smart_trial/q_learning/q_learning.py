"""Backward induction Q-learning for the three-stage SMART.

Uses counterfactual H3(H2, a2) when selecting stage-2 arms and counterfactual
H2/H3 when evaluating stage-1 arms. Stage 3 uses logistic Q on binary Y.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_RIDGE_ALPHA, REFERENCE_ARMS, STAGE1_ARMS, STAGE3_ARMS
from .features import build_H1, build_H2, build_H3
from .models import StageRegressor
from .pools import r1_pool_for, r2_pool_for


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


def _stage3_optimal(
    q3: StageRegressor,
    df: pd.DataFrame,
    *,
    a1: Optional[str] = None,
    a2: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Best stage-3 arm and value at counterfactual (a1, a2) or observed path."""
    H3 = build_H3(df, a1_override=a1, a2_override=a2)
    pools = [r2_pool_for(row) for _, row in df.iterrows()]
    q_by_arm = {a: q3.predict_for_arm(H3, a) for a in q3.arm_levels}
    return _argmax_over_pool(q_by_arm, pools)


def _best_stage2_via_q3(q3: StageRegressor, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Counterfactual stage-2 decision using fitted Q3."""
    from .config import STAGE2_ARMS

    n = len(df)
    best_arm = np.empty(n, dtype=object)
    best_q = np.full(n, -np.inf)
    for a2 in STAGE2_ARMS:
        _, q_vals = _stage3_optimal(q3, df, a2=a2)
        for i in range(n):
            if a2 not in r1_pool_for(df.iloc[i]):
                continue
            if q_vals[i] > best_q[i]:
                best_q[i] = q_vals[i]
                best_arm[i] = a2
    return best_arm, best_q


def _best_stage1_via_q3(q3: StageRegressor, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Counterfactual stage-1 decision: max over a1, a2, a3 via Q3."""
    from .config import STAGE2_ARMS

    n = len(df)
    best_arm = np.empty(n, dtype=object)
    best_q = np.full(n, -np.inf)
    for a1 in STAGE1_ARMS:
        for a2 in STAGE2_ARMS:
            _, q_vals = _stage3_optimal(q3, df, a1=a1, a2=a2)
            for i in range(n):
                if a2 not in r1_pool_for(df.iloc[i]):
                    continue
                if q_vals[i] > best_q[i]:
                    best_q[i] = q_vals[i]
                    best_arm[i] = a1
    return best_arm, best_q


@dataclass
class QLearningResult:
    stage3: StageRegressor
    stage2: StageRegressor
    stage1: StageRegressor
    rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    value: float = float("nan")
    value_crossfit: float = float("nan")
    pseudo_y2: np.ndarray = field(default_factory=lambda: np.array([]))
    pseudo_y1: np.ndarray = field(default_factory=lambda: np.array([]))
    pseudo_y0: np.ndarray = field(default_factory=lambda: np.array([]))


def fit(
    df: pd.DataFrame,
    *,
    stage3_model: Optional[StageRegressor] = None,
    stage2_model: Optional[StageRegressor] = None,
    stage1_model: Optional[StageRegressor] = None,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    use_logistic_stage3: bool = True,
) -> QLearningResult:
    """Run backward induction with counterfactual decisions at stages 1–2."""
    if df.empty:
        raise ValueError("Empty trajectory DataFrame; nothing to fit.")

    H1 = build_H1(df)
    H2 = build_H2(df)
    H3 = build_H3(df)
    y = df["Y"].to_numpy(dtype=float)

    q3 = stage3_model or StageRegressor(
        arm_levels=list(STAGE3_ARMS),
        reference_arm=REFERENCE_ARMS["stage3"],
        binary_outcome=use_logistic_stage3,
        ridge_alpha=ridge_alpha,
    )
    q3.fit(H3, df["A3"], y)

    # Stage 3 decision (observed H3).
    r2_pools = [r2_pool_for(row) for _, row in df.iterrows()]
    q3_by_arm_obs = {a: q3.predict_for_arm(H3, a) for a in q3.arm_levels}
    best_a3, pseudo_y2 = _argmax_over_pool(q3_by_arm_obs, r2_pools)

    # Stage 2: counterfactual argmax + regression target from observed H3 max.
    best_a2, pseudo_y1_cf = _best_stage2_via_q3(q3, df)

    q2 = stage2_model or StageRegressor(
        arm_levels=["A2a", "A2b", "A2c"],
        reference_arm=REFERENCE_ARMS["stage2"],
        ridge_alpha=ridge_alpha,
    )
    q2.fit(H2, df["A2"], pseudo_y2)

    # Stage 1: counterfactual argmax via Q3; regression target from stage-2 CF values.
    best_a1, pseudo_y0 = _best_stage1_via_q3(q3, df)

    q1 = stage1_model or StageRegressor(
        arm_levels=STAGE1_ARMS,
        reference_arm=REFERENCE_ARMS["stage1"],
        ridge_alpha=ridge_alpha,
    )
    q1.fit(H1, df["A1"], pseudo_y1_cf)

    rules = pd.DataFrame({
        "encounter_id": df["encounter_id"].values,
        "case_id": df["case_id"].values,
        "case_category": df["case_category"].values,
        "R1_responder": df["R1_responder"].values,
        "R2_level": df["R2_level"].values,
        "observed_A1": df["A1"].values,
        "observed_A2": df["A2"].values,
        "observed_A3": df["A3"].values,
        "pi1": best_a1,
        "pi2": best_a2,
        "pi3": best_a3,
        "Qhat_at_pi": pseudo_y0,
    })

    return QLearningResult(
        stage3=q3,
        stage2=q2,
        stage1=q1,
        rules=rules,
        value=float(np.mean(pseudo_y0)),
        pseudo_y2=pseudo_y2,
        pseudo_y1=pseudo_y1_cf,
        pseudo_y0=pseudo_y0,
    )


def value_of_fixed_strategy(
    df: pd.DataFrame,
    a1: str,
    a2: str,
    a3: str,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    use_logistic_stage3: bool = True,
    fitted: Optional[QLearningResult] = None,
) -> float:
    """Plug-in value of static (a1,a2,a3) via Q3 at counterfactual H3(a1,a2)."""
    if df.empty:
        return float("nan")

    q3 = fitted.stage3 if fitted is not None else StageRegressor(
        arm_levels=list(STAGE3_ARMS),
        reference_arm=REFERENCE_ARMS["stage3"],
        binary_outcome=use_logistic_stage3,
        ridge_alpha=ridge_alpha,
    )
    if fitted is None:
        H3 = build_H3(df)
        q3.fit(H3, df["A3"], df["Y"].to_numpy(dtype=float))

    H3_cf = build_H3(df, a1_override=a1, a2_override=a2)
    q3_pred = q3.predict_for_arm(H3_cf, a3)
    feasible: List[float] = []
    for i, row in df.reset_index(drop=True).iterrows():
        if a2 in r1_pool_for(row) and a3 in r2_pool_for(row):
            feasible.append(float(q3_pred[i]))
    if not feasible:
        return float("nan")
    return float(np.mean(feasible))


def cross_fitted_value(
    df: pd.DataFrame,
    n_folds: int = 5,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    use_logistic_stage3: bool = True,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """K-fold cross-fitted plug-in value of π̂ (reduces in-sample optimism)."""
    n = len(df)
    if n < 2:
        res = fit(df, ridge_alpha=ridge_alpha, use_logistic_stage3=use_logistic_stage3)
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
        res = fit(train_df, ridge_alpha=ridge_alpha, use_logistic_stage3=use_logistic_stage3)
        # Evaluate trained Q3 policy on test rows.
        _, test_vals = _best_stage1_via_q3(res.stage3, test_df)
        for j, orig_i in enumerate(test_idx):
            oof_values[orig_i] = test_vals[j]

    return float(np.nanmean(oof_values)), oof_values
