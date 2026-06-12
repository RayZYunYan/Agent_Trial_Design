"""Hand-crafted interpretable features H1, H2.

Plan §1: stay interpretable on the first pass. Embedding features can later be
appended as an ablation without changing this API.

Each function returns a DataFrame with the same row order as the input.
The columns are the features the regression at that stage may use.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .config import CATEGORY_LEVELS, CATEGORY_MAP


def normalize_category(series: pd.Series) -> pd.Series:
    """Map raw log categories to the fixed CATEGORY_LEVELS vocabulary."""
    s = series.fillna("Other").astype(str)
    return s.map(lambda x: CATEGORY_MAP.get(x, CATEGORY_MAP.get(x.strip(), "Other")))


def _one_hot(series: pd.Series, levels: List[str], prefix: str) -> pd.DataFrame:
    s = normalize_category(series)
    s = s.where(s.isin(levels), other="Other")
    return pd.get_dummies(s, prefix=prefix).reindex(
        columns=[f"{prefix}_{lvl}" for lvl in levels], fill_value=0
    )


def build_H1(df: pd.DataFrame) -> pd.DataFrame:
    """State entering Stage 1: only baseline covariates available."""
    feats = pd.DataFrame(index=df.index)
    feats = pd.concat(
        [feats, _one_hot(df["case_category"], CATEGORY_LEVELS, "cat")],
        axis=1,
    )
    lit = df["literacy_id"].fillna("")
    feats["literacy_F"] = (lit == "literacy_F").astype(int)
    feats["literacy_I"] = (lit == "literacy_I").astype(int)
    # Drop one literacy level (F) to avoid dummy trap; I and C vs reference F.
    feats["literacy_C"] = (lit == "literacy_C").astype(int)
    feats["chief_complaint_len"] = df["chief_complaint"].fillna("").str.len()
    return feats


def build_H2(
    df: pd.DataFrame,
    *,
    a1_override: Optional[str] = None,
) -> pd.DataFrame:
    """State entering Stage 2: H1 + R1 features + Stage-1 arm + turn counts."""
    h1 = build_H1(df)
    a1 = df["A1"] if a1_override is None else pd.Series(a1_override, index=df.index)
    extras = pd.DataFrame(index=df.index)
    extras["R1_total"] = df["R1_total"].fillna(0).astype(float)
    extras["R1_responder"] = df["R1_responder"].astype(int)
    extras["R1_red_flags"] = df["R1_red_flags"].fillna(0).astype(float)
    extras["R1_OPQRST"] = df["R1_OPQRST"].fillna(0).astype(float)
    extras["R1_pmh"] = df["R1_pmh"].fillna(0).astype(float)
    extras["R1_meds"] = df["R1_meds"].fillna(0).astype(float)
    extras["R1_social"] = df["R1_social"].fillna(0).astype(float)
    extras["stage1_turns"] = df["stage1_turns"].fillna(0).astype(float)
    extras["A1_a"] = (a1 == "A1a").astype(int)
    extras["A1_b"] = (a1 == "A1b").astype(int)
    extras["A1_c"] = (a1 == "A1c").astype(int)
    return pd.concat([h1, extras], axis=1)


def build_all(df: pd.DataFrame) -> dict:
    """Return {'H1': ..., 'H2': ...} aligned to df.

    R2 is measured after A2 is chosen (post-treatment), so it is not part of
    any decision state in the two-stage design; it stays in the raw DataFrame
    for descriptive analysis only.
    """
    return {"H1": build_H1(df), "H2": build_H2(df)}
