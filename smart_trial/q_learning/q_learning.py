"""Backward induction Q-learning for the three-stage SMART.

Plan §2:
  Stage 3:  fit Q̂3(H3, A3) on observed Y, restricted argmax over R2-pool.
  Stage 2:  fit Q̂2(H2, A2) on Ỹ2 = max_{a3 in R2-pool} Q̂3,
            restricted argmax over R1-pool.
  Stage 1:  fit Q̂1(H1, A1) on Ỹ1 = max_{a2 in R1-pool} Q̂2,
            argmax over {A1a, A1b, A1c}.

The estimated optimal DTR π̂ = (π̂1, π̂2, π̂3) is a function of the observed
history; its value on the data is the row-mean of Ỹ0 = max_{a1} Q̂1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import STAGE1_ARMS, STAGE2_POOLS, STAGE3_POOLS
from .features import build_H1, build_H2, build_H3
from .models import StageRegressor


def _r1_pool_for(row: pd.Series) -> List[str]:
    return STAGE2_POOLS["responder"] if bool(row["R1_responder"]) else STAGE2_POOLS["non-responder"]


def _r2_pool_for(row: pd.Series) -> List[str]:
    return STAGE3_POOLS.get(str(row.get("R2_level", "low")), STAGE3_POOLS["low"])


def _argmax_over_pool(q_by_arm: Dict[str, np.ndarray], pools: List[List[str]]):
    """Row-wise argmax restricted to that row's pool. Returns (best_arm[n], best_q[n])."""
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


@dataclass
class QLearningResult:
    stage3: StageRegressor
    stage2: StageRegressor
    stage1: StageRegressor
    rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    value: float = float("nan")
    pseudo_y2: np.ndarray = field(default_factory=lambda: np.array([]))
    pseudo_y1: np.ndarray = field(default_factory=lambda: np.array([]))
    pseudo_y0: np.ndarray = field(default_factory=lambda: np.array([]))


def fit(
    df: pd.DataFrame,
    *,
    stage3_model: Optional[StageRegressor] = None,
    stage2_model: Optional[StageRegressor] = None,
    stage1_model: Optional[StageRegressor] = None,
) -> QLearningResult:
    """Run the three backward-induction steps. Returns fitted regressors + rules."""
    if df.empty:
        raise ValueError("Empty trajectory DataFrame; nothing to fit.")

    H1 = build_H1(df)
    H2 = build_H2(df)
    H3 = build_H3(df)
    y = df["Y"].to_numpy(dtype=float)

    # ---- Stage 3 ----
    q3 = stage3_model or StageRegressor(arm_levels=["A3a", "A3b", "A3c"])
    q3.fit(H3, df["A3"], y)

    r2_pools = [_r2_pool_for(row) for _, row in df.iterrows()]
    q3_by_arm = {a: q3.predict_for_arm(H3, a) for a in q3.arm_levels}
    best_a3, pseudo_y2 = _argmax_over_pool(q3_by_arm, r2_pools)

    # ---- Stage 2 ----
    q2 = stage2_model or StageRegressor(arm_levels=["A2a", "A2b", "A2c"])
    q2.fit(H2, df["A2"], pseudo_y2)

    r1_pools = [_r1_pool_for(row) for _, row in df.iterrows()]
    q2_by_arm = {a: q2.predict_for_arm(H2, a) for a in q2.arm_levels}
    best_a2, pseudo_y1 = _argmax_over_pool(q2_by_arm, r1_pools)

    # ---- Stage 1 ----
    q1 = stage1_model or StageRegressor(arm_levels=STAGE1_ARMS)
    q1.fit(H1, df["A1"], pseudo_y1)

    s1_pools = [STAGE1_ARMS] * len(df)
    q1_by_arm = {a: q1.predict_for_arm(H1, a) for a in q1.arm_levels}
    best_a1, pseudo_y0 = _argmax_over_pool(q1_by_arm, s1_pools)

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
        pseudo_y1=pseudo_y1,
        pseudo_y0=pseudo_y0,
    )


def value_of_fixed_strategy(df: pd.DataFrame, a1: str, a2: str, a3: str) -> float:
    """Plug-in value of a single fixed (a1,a2,a3) strategy: fit Q3 then evaluate.

    Used for §5's adaptive-vs-static comparison. Strategy must be feasible for
    every row's pool; rows where it isn't are skipped.
    """
    if df.empty:
        return float("nan")
    H3 = build_H3(df)
    q3 = StageRegressor(arm_levels=["A3a", "A3b", "A3c"])
    q3.fit(H3, df["A3"], df["Y"].to_numpy(dtype=float))
    q3_pred = q3.predict_for_arm(H3, a3)
    feasible = []
    for i, row in df.reset_index(drop=True).iterrows():
        if a2 in _r1_pool_for(row) and a3 in _r2_pool_for(row):
            feasible.append(q3_pred[i])
    if not feasible:
        return float("nan")
    return float(np.mean(feasible))
