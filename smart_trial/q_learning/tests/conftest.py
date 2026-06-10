"""Shared fixtures for q_learning tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Tiny synthetic encounter table that respects pool constraints."""
    rng = np.random.default_rng(0)
    n = 60
    cats = rng.choice(["Cardiology", "Pulmonology", "Other"], size=n)
    a1 = rng.choice(["A1a", "A1b", "A1c"], size=n)
    r1_total = rng.integers(0, 11, size=n)
    r1_resp = r1_total >= 6
    a2 = np.array([
        rng.choice(["A2a", "A2b", "A2c"]) if r else rng.choice(["A2a", "A2b"])
        for r in r1_resp
    ])
    r2_final = rng.uniform(0.3, 0.95, size=n)
    r2_level = np.where(r2_final >= 0.7, "high", "low")
    base = 0.4
    bonus_cat_a1c = 0.15 * ((cats == "Cardiology") & (a1 == "A1c"))
    bonus_a2 = 0.2 * (r1_resp & (a2 == "A2a"))
    p = np.clip(base + bonus_cat_a1c + bonus_a2, 0.05, 0.95)
    y = rng.binomial(1, p)
    return pd.DataFrame({
        "encounter_id": [f"enc_{i:03d}" for i in range(n)],
        "case_id": [f"case_{i}" for i in range(n)],
        "case_category": cats,
        "seed": 0,
        "chief_complaint": "x" * 20,
        "A1": a1,
        "R1_total": r1_total,
        "R1_responder": r1_resp,
        "R1_OPQRST": rng.integers(0, 3, size=n),
        "R1_red_flags": rng.integers(0, 3, size=n),
        "R1_pmh": rng.integers(0, 3, size=n),
        "R1_meds": rng.integers(0, 3, size=n),
        "R1_social": rng.integers(0, 3, size=n),
        "stage1_turns": 4,
        "A2": a2,
        "stage2_pool": np.where(r1_resp, "responder", "non-responder"),
        "R2_final_conf": r2_final,
        "R2_avg_conf": r2_final,
        "R2_level": r2_level,
        "stage2_turns": 6,
        "literacy_id": "literacy_I",
        "Y": y,
        "outcome_red_flag_miss": False,
        "outcome_dangerous": False,
    })
