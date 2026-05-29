"""Oracle validation — plan §7 W2 task.

Build a synthetic environment where the truly optimal DTR is known by
construction, then run the full Q-learning pipeline on synthetic data of
increasing N. Plot recovery rate vs N. If the pipeline can't recover the
known optimum on clean synthetic data, *don't trust the real-data answer*.

Status: STUB. Two functions sketched:
  - generate_synthetic_trajectories(n, seed, true_dtr_spec): returns a
    DataFrame in the same schema as load.to_dataframe.
  - recovery_rate(n_values, n_replicates, true_dtr_spec): for each N, draw
    n_replicates synthetic datasets, fit Q-learning, check fraction of rows
    where π̂ matches the true rule.

Sketch of a clean generative model worth using:
  H1 ~ baseline covariates (category, literacy)
  A1 | H1 ~ Uniform(STAGE1_ARMS)
  R1 | H1, A1 ~ Bernoulli with rate depending on (category, A1)
  A2 | R1 ~ Uniform(pool_R1)
  R2 | H2, A2 ~ Bernoulli with rate depending on R1, category, A2
  A3 | R2 ~ Uniform(pool_R2)
  Y | H3, A3 ~ Bernoulli with rate where the optimal (a3 | H3) depends
                on R2 and category in a known closed form.

The "known optimal DTR" then falls out of the conditional success rates.
Sanity check by computing it analytically and comparing to what Q-learning
recovers as n grows. Expect monotone recovery → ~1 by n ≈ 1000.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def generate_synthetic_trajectories(
    n: int,
    seed: int = 0,
    true_dtr_spec: Dict | None = None,
) -> pd.DataFrame:
    """Return a synthetic DataFrame matching the schema of load.to_dataframe.

    NOT IMPLEMENTED. Required columns (see load.py REQUIRED_TOP_LEVEL plus the
    flat columns: A1, A2, A3, R1_*, R2_*, Y, case_category, etc.).

    Suggested defaults if true_dtr_spec is None:
        category effects only; A2a beats A2b by 0.15 for responders,
        A3a optimal in 'high' R2, A3c optimal in 'low' R2.
    """
    raise NotImplementedError(
        "Implement synthetic data generator. See module docstring for the suggested DGP."
    )


def recovery_rate(
    n_values: List[int],
    n_replicates: int = 50,
    seed: int = 0,
    true_dtr_spec: Dict | None = None,
) -> pd.DataFrame:
    """Run Q-learning on synthetic data, return per-N recovery rate of the optimal DTR.

    NOT IMPLEMENTED. Suggested output columns: ['n', 'replicate', 'agreement_rate',
    'value_gap']. agreement_rate = fraction of test-state rows where π̂ == π*;
    value_gap = E_H1[Q*(π*) - Q*(π̂)]. Plot agreement_rate vs n with error bars.
    """
    raise NotImplementedError(
        "Implement oracle recovery loop. See plan §7 — this is the highest-risk "
        "validation step; do it in parallel with the main analysis, not after."
    )
