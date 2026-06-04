"""Bootstrap inference for DTR value estimates.

Plan §4:
- value_of_fixed_strategy / Stage-3 contrasts: standard non-parametric bootstrap OK.
- value of estimated optimal DTR (contains max): non-regular under H0 at non-unique
  argmax. Use m-out-of-n bootstrap or Chakraborty-Murphy-Strecher (CMS, 2010) projection.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd


def percentile_ci(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Standard non-parametric bootstrap percentile CI.

    Use for regular targets: Stage-3 Q-values, value of a *fixed* DTR, observed-mean Y.
    Do NOT use for the value of the estimated *optimal* DTR — use m_out_of_n_ci instead.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return float("nan"), (float("nan"), float("nan")), np.array([])
    point = float(statistic(df))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = float(statistic(df.iloc[idx].reset_index(drop=True)))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, (float(lo), float(hi)), draws


def m_out_of_n_ci(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 500,
    m: Optional[int] = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """m-out-of-n bootstrap for non-regular parameters (value of estimated optimal DTR).

    Default m = floor(n^{0.75}) per Chakraborty, Murphy & Strecher (2010). The
    resampled estimator distribution, after the n→m correction, is consistent
    even at points where the limiting distribution of √n(V̂−V) is non-Gaussian
    (degenerate ties under H0 at non-unique optimal arm).

    Caveat: pure m-out-of-n is known to be conservative. CMS Section 4 proposes
    a smoothed / projected variant that tightens coverage — left as a TODO when
    sample sizes get large enough to matter.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return float("nan"), (float("nan"), float("nan")), np.array([])
    if m is None:
        m = max(2, int(np.floor(n ** 0.75)))
    point = float(statistic(df))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=m)
        draws[b] = float(statistic(df.iloc[idx].reset_index(drop=True)))
    # CMS scaling: re-center boot dist by point, scale by √(m/n).
    scaled = point + (draws - point) * np.sqrt(m / n)
    lo, hi = np.quantile(scaled, [alpha / 2, 1 - alpha / 2])
    return point, (float(lo), float(hi)), draws


def optimal_dtr_value(df: pd.DataFrame) -> float:
    """Statistic callable: fit Q-learning on df, return mean of pseudo_y0."""
    from .q_learning import fit
    return float(fit(df).value)


def cross_fitted_dtr_value(df: pd.DataFrame) -> float:
    """Statistic callable: K-fold cross-fitted value of π̂."""
    from .q_learning import cross_fitted_value
    val, _ = cross_fitted_value(df)
    return float(val)


def cms_projection_ci(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 500,
    m: Optional[int] = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """CMS projection bootstrap: 2 V_full - V_mn resamples (Chakraborty et al. 2010)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n == 0:
        return float("nan"), (float("nan"), float("nan")), np.array([])
    if m is None:
        m = max(2, int(np.floor(n ** 0.75)))
    point = float(statistic(df))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=m)
        draws[b] = float(statistic(df.iloc[idx].reset_index(drop=True)))
    projected = 2.0 * point - draws
    lo, hi = np.quantile(projected, [alpha / 2, 1 - alpha / 2])
    return point, (float(lo), float(hi)), projected
