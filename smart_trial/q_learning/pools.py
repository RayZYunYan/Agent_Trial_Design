"""Shared pool helpers for SMART stage-2 randomization."""
from __future__ import annotations

from typing import List

import pandas as pd

from .config import STAGE1_ARMS, STAGE2_POOLS


def r1_pool_for(row: pd.Series) -> List[str]:
    return STAGE2_POOLS["responder"] if bool(row["R1_responder"]) else STAGE2_POOLS["non-responder"]


def stage1_pool() -> List[str]:
    return list(STAGE1_ARMS)
