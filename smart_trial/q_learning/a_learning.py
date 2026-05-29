"""A-learning (advantage learning) estimator — plan §3.

Why this exists:
  Q-learning relies on the outcome regression being correctly specified.
  A-learning targets only the *contrast* between arms (the optimal blip
  function), so when the propensity score is known — which it is here, since
  arm assignment is by design random within each pool — A-learning is
  unbiased even if the baseline outcome model is wrong.

When Q-learning and A-learning agree → conclusions are robust to model
misspecification. When they disagree → Q's outcome model is the suspect.

Status: STUB. Intern task in Week 3 of the plan. Outline below.

Reference: Schulte, Tsiatis, Laber, Davidian (2014) "Q- and A-learning methods
for estimating optimal dynamic treatment regimes." Statistical Science 29:640–661.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import STAGE1_ARMS, STAGE2_POOLS, STAGE3_POOLS


# Known design propensities. STAGE1: uniform over 3 arms.
# STAGE2: uniform over responder/non-responder pool.
# STAGE3: uniform over high/low pool.
STAGE1_PROPENSITY: Dict[str, float] = {a: 1.0 / len(STAGE1_ARMS) for a in STAGE1_ARMS}

def stage2_propensity(responder: bool, arm: str) -> float:
    pool = STAGE2_POOLS["responder" if responder else "non-responder"]
    return (1.0 / len(pool)) if arm in pool else 0.0

def stage3_propensity(confidence_level: str, arm: str) -> float:
    pool = STAGE3_POOLS.get(confidence_level, STAGE3_POOLS["low"])
    return (1.0 / len(pool)) if arm in pool else 0.0


@dataclass
class ALearningResult:
    """Placeholder result type. Populate fields as estimator is implemented."""
    rules: pd.DataFrame
    value: float


def fit(df: pd.DataFrame) -> ALearningResult:
    """A-learning fit. NOT IMPLEMENTED — see steps below.

    Algorithm (one stage shown, repeat back from t=3):

      1. Pick a parametric blip model μ_t(H_t; ψ_t) for the contrast vs reference arm.
         e.g. μ_t(H_t; ψ_t) = H_t^T ψ_t,a   per non-reference arm.
      2. Pick a parametric model ν_t(H_t; β_t) for the baseline (reference arm) outcome.
         e.g. ν_t(H_t; β_t) = H_t^T β_t.
      3. Pseudo-outcome at stage t:
             Ỹ_t = Y if t = T else max over feasible a of Q̂_{t+1}(H_{t+1}, a)
                                       — same recursion as Q-learning.
      4. Solve the estimating equation
             ∑_i [Ỹ_t - ν_t(H_t; β_t) - μ_t(H_t, A_t; ψ_t)] ·
                 [1{A_t=a} - π_t(a | H_t)] · ∂μ_t/∂ψ_t = 0
         simultaneously for (β_t, ψ_t).
         The propensity-weighted residual against (A_t indicator - propensity) is
         what gives the double-robustness — only one of the blip OR the baseline
         model needs to be correct (and propensity is known).
      5. Decision rule: π̂_t(H_t) = argmax_a μ_t(H_t, a; ψ̂_t).

    Practical Python approach:
      - Linearise the estimating equation into a weighted least-squares form
        (Robins 2004 trick) and solve with sklearn LinearRegression on the
        adjusted design matrix.
      - Re-use models.build_design for the H_t × arm interactions.

    Cross-checking: run on the same df as q_learning.fit() and compare the
    resulting decision rules row-by-row. Expect >90% agreement on data the
    Q-model fits well.
    """
    raise NotImplementedError(
        "A-learning estimator is intentionally a stub. "
        "See docstring for the algorithm; implement in Week 3 per analysis plan §3."
    )
