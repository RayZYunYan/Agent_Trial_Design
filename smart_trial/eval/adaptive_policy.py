"""Closed-loop arm assignment via Q-learning policy with biased randomization."""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from smart_trial.q_learning import q_learning
from smart_trial.q_learning.config import STAGE1_ARMS
from smart_trial.q_learning.features import build_H1, build_H2
from smart_trial.q_learning.load import _literacy_from_persona_block
from smart_trial.q_learning.pools import r1_pool_for


def _stable_seed(global_seed: int, case_id: str, tag: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{case_id}:{tag}".encode()).hexdigest()
    return int(digest[:16], 16)


def _persona_literacy(persona: Optional[Dict[str, Any]]) -> Optional[str]:
    if not persona:
        return None
    return _literacy_from_persona_block(persona)


def _case_row_h1(case: Dict[str, Any], persona: Optional[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "case_category": case.get("case_category", "Other"),
            "chief_complaint": case.get("chief_complaint", ""),
            "literacy_id": _persona_literacy(persona),
        }]
    )


def _case_row_h2(
    case: Dict[str, Any],
    persona: Optional[Dict[str, Any]],
    *,
    stage1_arm: str,
    R1: Dict[str, Any],
    stage1_turns: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "case_category": case.get("case_category", "Other"),
            "chief_complaint": case.get("chief_complaint", ""),
            "literacy_id": _persona_literacy(persona),
            "A1": stage1_arm,
            "R1_total": R1.get("total"),
            "R1_responder": bool(R1.get("responder", False)),
            "R1_red_flags": R1.get("red_flags"),
            "R1_OPQRST": R1.get("OPQRST"),
            "R1_pmh": R1.get("past_medical_history"),
            "R1_meds": R1.get("medications_allergies"),
            "R1_social": R1.get("social_family_history"),
            "stage1_turns": stage1_turns,
        }]
    )


def biased_sample_arm(
    arms: List[str],
    q_by_arm: Dict[str, float],
    rng: random.Random,
    *,
    temperature: float = 1.0,
) -> Tuple[str, float]:
    """Sample arm with p(a) ∝ exp(Q(a)/temperature); return (arm, propensity)."""
    if not arms:
        raise ValueError("empty arm pool for biased sampling")
    if len(arms) == 1:
        return arms[0], 1.0

    scores = np.array([float(q_by_arm.get(a, 0.0)) for a in arms], dtype=float)
    temp = max(float(temperature), 1e-6)
    scaled = scores / temp
    scaled -= np.max(scaled)
    exp_s = np.exp(scaled)
    total = float(exp_s.sum())
    if total <= 0 or not np.isfinite(total):
        probs = np.full(len(arms), 1.0 / len(arms))
    else:
        probs = exp_s / total

    idx = rng.choices(range(len(arms)), weights=probs.tolist(), k=1)[0]
    return arms[idx], float(probs[idx])


class AdaptivePolicyAssigner:
    """Assign SMART arms using fitted Q models and softmax biased randomization."""

    def __init__(
        self,
        result: q_learning.QLearningResult,
        *,
        refit_generation: int = 0,
        temperature: float = 1.0,
    ) -> None:
        self.result = result
        self.refit_generation = refit_generation
        self.temperature = temperature

    def assign_stage1(
        self,
        case: Dict[str, Any],
        persona: Optional[Dict[str, Any]],
        seed: int,
    ) -> Tuple[str, float, Dict[str, float]]:
        h1 = build_H1(_case_row_h1(case, persona))
        q_vals = {
            a: float(self.result.stage1.predict_for_arm(h1, a)[0])
            for a in STAGE1_ARMS
        }
        rng = random.Random(_stable_seed(seed, case["case_id"], f"s1_g{self.refit_generation}"))
        arm, prop = biased_sample_arm(STAGE1_ARMS, q_vals, rng, temperature=self.temperature)
        return arm, prop, q_vals

    def assign_stage2(
        self,
        case: Dict[str, Any],
        persona: Optional[Dict[str, Any]],
        stage1_arm: str,
        R1: Dict[str, Any],
        stage1_turns: int,
        seed: int,
    ) -> Dict[str, Any]:
        row = _case_row_h2(
            case,
            persona,
            stage1_arm=stage1_arm,
            R1=R1,
            stage1_turns=stage1_turns,
        )
        h2 = build_H2(row, a1_override=stage1_arm)
        pool = r1_pool_for(row.iloc[0])
        q_vals = {
            a: float(self.result.stage2.predict_for_arm(h2, a)[0])
            for a in pool
        }
        rng = random.Random(_stable_seed(seed, case["case_id"], f"s2_g{self.refit_generation}"))
        arm, prop = biased_sample_arm(pool, q_vals, rng, temperature=self.temperature)
        pool_key = "responder" if R1.get("responder") else "non-responder"
        return {
            "arm": arm,
            "pool_used": pool_key,
            "pool": pool,
            "R1_total": R1.get("total", 0),
            "policy_assigned": True,
            "propensity": prop,
            "q_values": q_vals,
        }
