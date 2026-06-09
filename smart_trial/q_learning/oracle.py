"""Oracle validation — synthetic DGP with known optimal DTR.

Generate data where arm success probabilities follow a closed form, then
measure how often Q-learning recovers the known rule as n grows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    CATEGORY_LEVELS,
    R1_RESPONDER_THRESHOLD,
    R2_HIGH_CONFIDENCE_THRESHOLD,
    STAGE1_ARMS,
    STAGE2_POOLS,
    STAGE3_POOLS,
)
from . import q_learning


DEFAULT_DTR_SPEC: Dict = {
    "stage1_rules": {"Cardiology": "A1c"},
    "stage1_default": "A1a",
    "stage2_responder_best": "A2a",
    "stage2_non_responder_best": "A2a",
    "stage3_high_best": "A3a",
    "stage3_low_best": "A3c",
    "base_prob": 0.35,
    "a1c_cardiology_bonus": 0.18,
    "a2a_responder_bonus": 0.12,
    "a3a_high_bonus": 0.20,
    "a3c_low_bonus": 0.20,
}


@dataclass
class OracleDTRSpec:
    stage1_rules: Dict[str, str]
    stage1_default: str
    stage2_responder_best: str
    stage2_non_responder_best: str
    stage3_high_best: str
    stage3_low_best: str
    base_prob: float
    a1c_cardiology_bonus: float
    a2a_responder_bonus: float
    a3a_high_bonus: float
    a3c_low_bonus: float

    @classmethod
    def from_dict(cls, d: Optional[Dict] = None) -> "OracleDTRSpec":
        merged = {**DEFAULT_DTR_SPEC, **(d or {})}
        return cls(**{k: merged[k] for k in cls.__dataclass_fields__})


def _true_pi1(category: str, spec: OracleDTRSpec) -> str:
    return spec.stage1_rules.get(category, spec.stage1_default)


def _true_pi2(responder: bool, spec: OracleDTRSpec) -> str:
    return spec.stage2_responder_best if responder else spec.stage2_non_responder_best


def _true_pi3(r2_level: str, spec: OracleDTRSpec) -> str:
    return spec.stage3_high_best if r2_level == "high" else spec.stage3_low_best


def success_prob(
    category: str,
    a1: str,
    responder: bool,
    a2: str,
    r2_level: str,
    a3: str,
    spec: OracleDTRSpec,
) -> float:
    p = spec.base_prob
    if category == "Cardiology" and a1 == "A1c":
        p += spec.a1c_cardiology_bonus
    if responder and a2 == "A2a":
        p += spec.a2a_responder_bonus
    if r2_level == "high" and a3 == "A3a":
        p += spec.a3a_high_bonus
    if r2_level == "low" and a3 == "A3c":
        p += spec.a3c_low_bonus
    return float(np.clip(p, 0.05, 0.95))


def generate_synthetic_trajectories(
    n: int,
    seed: int = 0,
    true_dtr_spec: Optional[Dict] = None,
) -> pd.DataFrame:
    """Return a synthetic DataFrame matching load.to_dataframe schema."""
    spec = OracleDTRSpec.from_dict(true_dtr_spec)
    rng = np.random.default_rng(seed)
    cats = rng.choice(CATEGORY_LEVELS, size=n)
    a1 = rng.choice(STAGE1_ARMS, size=n)

    r1_total = np.zeros(n, dtype=int)
    for i in range(n):
        base = rng.integers(2, 8)
        if a1[i] == "A1c":
            base += rng.integers(1, 4)
        r1_total[i] = int(np.clip(base, 0, 10))
    r1_resp = r1_total >= R1_RESPONDER_THRESHOLD

    a2 = np.empty(n, dtype=object)
    for i in range(n):
        pool = STAGE2_POOLS["responder" if r1_resp[i] else "non-responder"]
        a2[i] = rng.choice(pool)

    r2_final = np.clip(
        rng.normal(0.55, 0.15, size=n) + (a2 == "A2a") * 0.1,
        0.2,
        0.95,
    )
    r2_level = np.where(r2_final >= R2_HIGH_CONFIDENCE_THRESHOLD, "high", "low")

    a3 = np.empty(n, dtype=object)
    for i in range(n):
        pool = STAGE3_POOLS[str(r2_level[i])]
        a3[i] = rng.choice(pool)

    y = np.zeros(n, dtype=int)
    for i in range(n):
        p = success_prob(
            cats[i], a1[i], bool(r1_resp[i]), a2[i], str(r2_level[i]), a3[i], spec,
        )
        y[i] = int(rng.binomial(1, p))

    return pd.DataFrame({
        "encounter_id": [f"syn_{seed}_{i:05d}" for i in range(n)],
        "case_id": [f"syn_case_{i}" for i in range(n)],
        "case_category": cats,
        "seed": seed,
        "chief_complaint": ["synthetic complaint text"] * n,
        "A1": a1,
        "R1_total": r1_total,
        "R1_responder": r1_resp,
        "R1_OPQRST": rng.integers(0, 3, size=n),
        "R1_red_flags": rng.integers(0, 3, size=n),
        "R1_pmh": rng.integers(0, 3, size=n),
        "R1_meds": rng.integers(0, 3, size=n),
        "R1_social": rng.integers(0, 3, size=n),
        "stage1_turns": np.full(n, 4),
        "A2": a2,
        "stage2_pool": np.where(r1_resp, "responder", "non-responder"),
        "R2_final_conf": r2_final,
        "R2_avg_conf": r2_final,
        "R2_level": r2_level,
        "stage2_turns": np.full(n, 6),
        "A3": a3,
        "stage3_pool": r2_level,
        "stage3_turns": np.full(n, 3),
        "literacy_id": "literacy_I",
        "Y": y,
        "outcome_red_flag_miss": np.zeros(n, dtype=bool),
        "outcome_dangerous": np.zeros(n, dtype=bool),
    })


def true_optimal_rules(df: pd.DataFrame, spec: OracleDTRSpec) -> pd.DataFrame:
    """Per-row oracle DTR from the generative success-prob structure."""
    pi1, pi2, pi3 = [], [], []
    for _, row in df.iterrows():
        cat = row["case_category"]
        resp = bool(row["R1_responder"])
        r2 = str(row["R2_level"])
        pi1.append(_true_pi1(cat, spec))
        pi2.append(_true_pi2(resp, spec))
        pi3.append(_true_pi3(r2, spec))
    return pd.DataFrame({"pi1": pi1, "pi2": pi2, "pi3": pi3})


def _agreement_rate(estimated: pd.DataFrame, truth: pd.DataFrame) -> Dict[str, float]:
    merged = pd.concat([estimated.reset_index(drop=True), truth.reset_index(drop=True)], axis=1)
    merged.columns = ["ep1", "ep2", "ep3", "tp1", "tp2", "tp3"]
    return {
        "agree_pi1": float((merged["ep1"] == merged["tp1"]).mean()),
        "agree_pi2": float((merged["ep2"] == merged["tp2"]).mean()),
        "agree_pi3": float((merged["ep3"] == merged["tp3"]).mean()),
        "agree_all": float((merged[["ep1", "ep2", "ep3"]] == merged[["tp1", "tp2", "tp3"]].values).all(axis=1).mean()),
    }


def recovery_rate(
    n_values: List[int],
    n_replicates: int = 20,
    seed: int = 0,
    true_dtr_spec: Optional[Dict] = None,
) -> pd.DataFrame:
    """Fit Q-learning on synthetic data; report agreement with oracle DTR."""
    spec = OracleDTRSpec.from_dict(true_dtr_spec)
    rows: List[Dict] = []
    for n in n_values:
        for rep in range(n_replicates):
            rep_seed = seed + n * 1000 + rep
            df = generate_synthetic_trajectories(n, seed=rep_seed, true_dtr_spec=true_dtr_spec)
            truth = true_optimal_rules(df, spec)
            try:
                result = q_learning.fit(df)
                est = result.rules[["pi1", "pi2", "pi3"]]
                rates = _agreement_rate(est, truth)
                rows.append({
                    "n": n,
                    "replicate": rep,
                    "agreement_rate": rates["agree_all"],
                    "agree_pi1": rates["agree_pi1"],
                    "agree_pi2": rates["agree_pi2"],
                    "agree_pi3": rates["agree_pi3"],
                    "V_hat": result.value,
                    "value_gap": float(np.mean(df["Y"]) - result.value),
                })
            except Exception as exc:  # noqa: BLE001 — record failed replicates
                rows.append({
                    "n": n,
                    "replicate": rep,
                    "agreement_rate": float("nan"),
                    "agree_pi1": float("nan"),
                    "agree_pi2": float("nan"),
                    "agree_pi3": float("nan"),
                    "V_hat": float("nan"),
                    "value_gap": float("nan"),
                    "error": str(exc),
                })
    return pd.DataFrame(rows)
