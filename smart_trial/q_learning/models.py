"""Thin regressor wrappers used by the Q-learning backward induction.

Default model: linear regression with arm-main-effects + arm×state interactions.
Swap in GradientBoostingRegressor (or any sklearn-compatible regressor) for the
§7 robustness pass without changing the call sites in q_learning.py.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@contextmanager
def _suppress_numerics():
    """Suppress harmless numpy/sklearn underflow chatter when n<p.

    Ridge with SVD still returns finite, well-defined coefficients in
    rank-deficient cases; the matmul warnings are noise from intermediate
    products. We hide them locally so pipeline stdout stays readable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            yield


def _arm_dummies(arms: pd.Series, levels: Sequence[str]) -> pd.DataFrame:
    """One-hot encode arm; full encoding (no drop) so we can read coefficients per arm."""
    dummies = pd.DataFrame(
        {f"arm_{lvl}": (arms == lvl).astype(int) for lvl in levels},
        index=arms.index,
    )
    return dummies


def _interactions(state: pd.DataFrame, arm_dummies: pd.DataFrame) -> pd.DataFrame:
    """All pairwise arm × state interactions (state columns multiplied by each arm dummy)."""
    blocks = []
    for arm_col in arm_dummies.columns:
        a = arm_dummies[arm_col].to_numpy().reshape(-1, 1)
        inter = pd.DataFrame(
            state.to_numpy() * a,
            index=state.index,
            columns=[f"{arm_col}*{c}" for c in state.columns],
        )
        blocks.append(inter)
    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=state.index)


def build_design(
    state: pd.DataFrame,
    arms: pd.Series,
    arm_levels: Sequence[str],
    with_interactions: bool = True,
) -> pd.DataFrame:
    """Design matrix = [state | arm dummies | arm × state interactions]."""
    arm_dum = _arm_dummies(arms, arm_levels)
    if not with_interactions:
        return pd.concat([state, arm_dum], axis=1)
    inter = _interactions(state, arm_dum)
    return pd.concat([state, arm_dum, inter], axis=1)


class StageRegressor:
    """Fit Q̂(H, A) once; expose predict_for_arm(H, a) to evaluate any arm at any state."""

    def __init__(self, arm_levels: List[str], model=None, with_interactions: bool = True):
        self.arm_levels = list(arm_levels)
        self.with_interactions = with_interactions
        # Ridge with the SVD solver: stable under rank-deficient design (n<p
        # or constant columns), preserves linear/interpretable structure.
        # For the §7 robustness pass swap in GradientBoostingRegressor.
        self.model = model if model is not None else Ridge(alpha=1.0, solver="svd")
        self._columns: List[str] = []

    def fit(self, state: pd.DataFrame, arms: pd.Series, y: np.ndarray) -> "StageRegressor":
        X = build_design(state, arms, self.arm_levels, self.with_interactions)
        self._columns = list(X.columns)
        with _suppress_numerics():
            self.model.fit(X.to_numpy(), np.asarray(y, dtype=float))
        return self

    def predict_for_arm(self, state: pd.DataFrame, arm: str) -> np.ndarray:
        if arm not in self.arm_levels:
            raise ValueError(f"Arm {arm!r} not in trained levels {self.arm_levels}")
        fake_arms = pd.Series([arm] * len(state), index=state.index)
        X = build_design(state, fake_arms, self.arm_levels, self.with_interactions)
        X = X.reindex(columns=self._columns, fill_value=0)
        with _suppress_numerics():
            return self.model.predict(X.to_numpy())

    def predict(self, state: pd.DataFrame, arms: pd.Series) -> np.ndarray:
        X = build_design(state, arms, self.arm_levels, self.with_interactions)
        X = X.reindex(columns=self._columns, fill_value=0)
        with _suppress_numerics():
            return self.model.predict(X.to_numpy())
