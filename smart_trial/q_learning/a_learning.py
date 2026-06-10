"""A-learning (advantage learning) estimator — plan §3.

Doubly-robust blip estimation with known SMART propensities. Decision rules
use estimated blip contrasts; backward pseudo-outcomes mirror Q-learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import DEFAULT_RIDGE_ALPHA, REFERENCE_ARMS, STAGE1_ARMS, STAGE2_ARMS
from .features import build_H1, build_H2
from .pools import r1_pool_for


STAGE1_PROPENSITY = {a: 1.0 / len(STAGE1_ARMS) for a in STAGE1_ARMS}


def stage2_propensity(responder: bool, arm: str) -> float:
    from .config import STAGE2_POOLS

    pool = STAGE2_POOLS["responder" if responder else "non-responder"]
    return (1.0 / len(pool)) if arm in pool else 0.0


def _propensity_vector(df: pd.DataFrame, arms: pd.Series, prop_fn: Callable) -> np.ndarray:
    out = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        out[i] = prop_fn(row, arms.iloc[i])
    return out


def _stage2_prop(row: pd.Series, arm: str) -> float:
    return stage2_propensity(bool(row["R1_responder"]), arm)


def _fit_blip_stage(
    state: pd.DataFrame,
    arms: pd.Series,
    pseudo_y: np.ndarray,
    arm_levels: List[str],
    reference_arm: str,
    prop_fn: Callable,
    df: pd.DataFrame,
    ridge_alpha: float,
) -> tuple[np.ndarray, dict]:
    """Robins-style blip regression: G ~ H*β + Σ (I(A=a)-π_a)*H*ψ_a."""
    H = state.astype(float).to_numpy()
    n, p = H.shape
    y = np.asarray(pseudo_y, dtype=float)

    # Baseline ν(H) from reference-arm subjects.
    ref_mask = (arms == reference_arm).to_numpy()
    if ref_mask.sum() >= max(p, 3):
        nu_coef = np.linalg.lstsq(H[ref_mask], y[ref_mask], rcond=None)[0]
    else:
        nu_coef = np.linalg.lstsq(H, y, rcond=None)[0]
    nu = H @ nu_coef

    blip_coefs: dict = {reference_arm: np.zeros(p)}
    for arm in arm_levels:
        if arm == reference_arm:
            continue
        pi = np.array([prop_fn(df.iloc[i], arm) for i in range(n)])
        ia = (arms == arm).astype(float).to_numpy()
        z = ia - pi
        resid = y - nu
        mask = np.abs(z) > 1e-6
        if mask.sum() > p:
            X_w = H[mask] * z[mask].reshape(-1, 1)
            target = resid[mask]
            model = Ridge(alpha=ridge_alpha, solver="svd")
            model.fit(X_w, target)
            blip_coefs[arm] = model.coef_.ravel()
        else:
            blip_coefs[arm] = np.zeros(p)

    return nu_coef, blip_coefs


def _blip_values(state: pd.DataFrame, nu_coef: np.ndarray, blip_coefs: dict) -> dict:
    H = state.astype(float).to_numpy()
    nu = H @ nu_coef
    return {arm: nu + H @ psi for arm, psi in blip_coefs.items()}


def _argmax_blips(
    blip_vals: dict,
    arm_levels: List[str],
    pools: List[List[str]],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pools)
    best_arm = np.empty(n, dtype=object)
    best_v = np.full(n, -np.inf)
    for i, pool in enumerate(pools):
        for a in pool:
            if a not in blip_vals:
                continue
            v = blip_vals[a][i]
            if v > best_v[i]:
                best_v[i] = v
                best_arm[i] = a
    return best_arm, best_v


@dataclass
class ALearningResult:
    rules: pd.DataFrame
    value: float
    blip_stage1: dict
    blip_stage2: dict


def fit(
    df: pd.DataFrame,
    *,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> ALearningResult:
    """A-learning backward induction with linear blip models."""
    if df.empty:
        raise ValueError("Empty trajectory DataFrame; nothing to fit.")

    H1 = build_H1(df)
    H2 = build_H2(df)
    y = df["Y"].to_numpy(dtype=float)

    ref2 = REFERENCE_ARMS["stage2"]
    nu2, blip2 = _fit_blip_stage(
        H2, df["A2"], y, list(STAGE2_ARMS), ref2, _stage2_prop, df, ridge_alpha,
    )
    blip2_vals = _blip_values(H2, nu2, blip2)
    r1_pools = [r1_pool_for(row) for _, row in df.iterrows()]
    pi2, pseudo_y1 = _argmax_blips(blip2_vals, list(STAGE2_ARMS), r1_pools)

    ref1 = REFERENCE_ARMS["stage1"]
    prop1 = lambda row, arm: STAGE1_PROPENSITY.get(arm, 0.0)  # noqa: E731
    nu1, blip1 = _fit_blip_stage(
        H1, df["A1"], pseudo_y1, STAGE1_ARMS, ref1, prop1, df, ridge_alpha,
    )
    blip1_vals = _blip_values(H1, nu1, blip1)
    s1_pools = [STAGE1_ARMS] * len(df)
    pi1, pseudo_y0 = _argmax_blips(blip1_vals, STAGE1_ARMS, s1_pools)

    rules = pd.DataFrame({
        "encounter_id": df["encounter_id"].values,
        "case_id": df["case_id"].values,
        "case_category": df["case_category"].values,
        "R1_responder": df["R1_responder"].values,
        "R2_level": df["R2_level"].values,
        "pi1": pi1,
        "pi2": pi2,
        "Avalue_at_pi": pseudo_y0,
    })

    return ALearningResult(
        rules=rules,
        value=float(np.mean(pseudo_y0)),
        blip_stage1=blip1,
        blip_stage2=blip2,
    )
