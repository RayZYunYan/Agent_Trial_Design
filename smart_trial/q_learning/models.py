"""Thin regressor wrappers used by the Q-learning backward induction.

Default model: Ridge on reference-arm contrast design (state + arm×state for
non-reference arms). Stage 2 can use logistic regression for binary Y.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@contextmanager
def _suppress_numerics():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
            yield


def build_design(
    state: pd.DataFrame,
    arms: pd.Series,
    arm_levels: Sequence[str],
    reference_arm: str,
    with_interactions: bool = True,
) -> pd.DataFrame:
    """Design = [state | (I(A=a)-ref contrasts via arm×state for a != ref)]."""
    parts = [state.reset_index(drop=True)]
    arms = arms.reset_index(drop=True)
    state = state.reset_index(drop=True)
    for arm in arm_levels:
        if arm == reference_arm:
            continue
        indicator = (arms == arm).astype(float)
        if with_interactions:
            inter = state.multiply(indicator, axis=0)
            inter = inter.add_prefix(f"arm_{arm}*")
            parts.append(inter)
        else:
            parts.append(pd.DataFrame({f"arm_{arm}": indicator}))
    return pd.concat(parts, axis=1)


class StageRegressor:
    """Fit Q̂(H, A); predict_for_arm evaluates any arm via contrast coding."""

    def __init__(
        self,
        arm_levels: List[str],
        reference_arm: Optional[str] = None,
        model=None,
        with_interactions: bool = True,
        *,
        binary_outcome: bool = False,
        ridge_alpha: float = 1.0,
    ):
        self.arm_levels = list(arm_levels)
        self.reference_arm = reference_arm or arm_levels[0]
        self.with_interactions = with_interactions
        self.binary_outcome = binary_outcome
        if model is not None:
            self.model = model
        elif binary_outcome:
            self.model = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0 / max(ridge_alpha, 1e-6),
                    solver="lbfgs",
                    max_iter=5000,
                )),
            ])
        else:
            self.model = Ridge(alpha=ridge_alpha, solver="svd")
        self._columns: List[str] = []

    def fit(self, state: pd.DataFrame, arms: pd.Series, y: np.ndarray) -> "StageRegressor":
        X = build_design(
            state, arms, self.arm_levels, self.reference_arm, self.with_interactions,
        )
        self._columns = list(X.columns)
        y_arr = np.asarray(y, dtype=float)
        with _suppress_numerics():
            if self.binary_outcome:
                self.model.fit(X.to_numpy(), y_arr.astype(int))
            else:
                self.model.fit(X.to_numpy(), y_arr)
        return self

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._columns, fill_value=0)
        with _suppress_numerics():
            if self.binary_outcome:
                return self.model.predict_proba(X.to_numpy())[:, 1]
            return self.model.predict(X.to_numpy())

    def predict_for_arm(self, state: pd.DataFrame, arm: str) -> np.ndarray:
        if arm not in self.arm_levels:
            raise ValueError(f"Arm {arm!r} not in trained levels {self.arm_levels}")
        fake_arms = pd.Series([arm] * len(state), index=state.index)
        X = build_design(
            state, fake_arms, self.arm_levels, self.reference_arm, self.with_interactions,
        )
        pred = self._predict_raw(X)
        return np.clip(pred, 0.0, 1.0)

    def predict(self, state: pd.DataFrame, arms: pd.Series) -> np.ndarray:
        X = build_design(
            state, arms, self.arm_levels, self.reference_arm, self.with_interactions,
        )
        pred = self._predict_raw(X)
        return np.clip(pred, 0.0, 1.0)

    @property
    def n_parameters(self) -> int:
        return len(self._columns) + (0 if self.binary_outcome else 1)
