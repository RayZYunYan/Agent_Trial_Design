"""Fixed case cohorts and grid path definitions for eval runs."""
from __future__ import annotations

from itertools import product

from smart_trial.q_learning.config import STAGE1_ARMS, STAGE2_ARMS

BENCHMARK_CASE_IDS = [f"medqa_{i:04d}" for i in range(50)]
PILOT_CASE_IDS = [f"medqa_{i:04d}" for i in range(5)]
CLOSED_LOOP_CASE_IDS = [f"medqa_{i:04d}" for i in range(100, 150)]

# Loader `SPECIALTY_TO_CATEGORY` vocabulary (8 coarse clinical buckets).
CASE_CATEGORIES = [
    "Cardiology",
    "Neuro",
    "GI",
    "Pulm",
    "Infectious",
    "Pediatrics",
    "Psychiatry",
    "Other",
]

GRID_PATHS: list[tuple[str, str]] = list(product(STAGE1_ARMS, STAGE2_ARMS))


def path_id_for(a1: str, a2: str) -> str:
    return f"{a1}_{a2}"
