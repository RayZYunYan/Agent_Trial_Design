"""Hand-crafted interpretable features H1, H2, H3.

Plan §1: stay interpretable on the first pass. Embedding features can later be
appended as an ablation without changing this API.

Each function returns a DataFrame with the same row order as the input.
The columns are the features the regression at that stage may use.
"""
from __future__ import annotations

from typing import List

import pandas as pd


CATEGORY_LEVELS = [
    "Cardiology",
    "Pulmonology",
    "Gastroenterology",
    "Neurology",
    "Infectious_Disease",
    "Endocrinology",
    "Other",
]


def _one_hot(series: pd.Series, levels: List[str], prefix: str) -> pd.DataFrame:
    s = series.fillna("Other").astype(str)
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
    feats["literacy_F"] = (df["literacy_id"] == "literacy_F").astype(int)
    feats["literacy_I"] = (df["literacy_id"] == "literacy_I").astype(int)
    feats["literacy_C"] = (df["literacy_id"] == "literacy_C").astype(int)
    feats["chief_complaint_len"] = df["chief_complaint"].fillna("").str.len()
    return feats


def build_H2(df: pd.DataFrame) -> pd.DataFrame:
    """State entering Stage 2: H1 + R1 features + Stage-1 arm + turn counts."""
    h1 = build_H1(df)
    extras = pd.DataFrame(index=df.index)
    extras["R1_total"] = df["R1_total"].fillna(0).astype(float)
    extras["R1_responder"] = df["R1_responder"].astype(int)
    extras["R1_red_flags"] = df["R1_red_flags"].fillna(0).astype(float)
    extras["R1_OPQRST"] = df["R1_OPQRST"].fillna(0).astype(float)
    extras["R1_pmh"] = df["R1_pmh"].fillna(0).astype(float)
    extras["R1_meds"] = df["R1_meds"].fillna(0).astype(float)
    extras["R1_social"] = df["R1_social"].fillna(0).astype(float)
    extras["stage1_turns"] = df["stage1_turns"].fillna(0).astype(float)
    extras["A1_a"] = (df["A1"] == "A1a").astype(int)
    extras["A1_b"] = (df["A1"] == "A1b").astype(int)
    extras["A1_c"] = (df["A1"] == "A1c").astype(int)
    return pd.concat([h1, extras], axis=1)


def build_H3(df: pd.DataFrame) -> pd.DataFrame:
    """State entering Stage 3: H2 + R2 + Stage-2 arm + Stage-2 turn count."""
    h2 = build_H2(df)
    extras = pd.DataFrame(index=df.index)
    extras["R2_final_conf"] = df["R2_final_conf"].fillna(0.0).astype(float)
    extras["R2_avg_conf"] = df["R2_avg_conf"].fillna(0.0).astype(float)
    extras["R2_high"] = (df["R2_level"] == "high").astype(int)
    extras["stage2_turns"] = df["stage2_turns"].fillna(0).astype(float)
    extras["A2_a"] = (df["A2"] == "A2a").astype(int)
    extras["A2_b"] = (df["A2"] == "A2b").astype(int)
    extras["A2_c"] = (df["A2"] == "A2c").astype(int)
    return pd.concat([h2, extras], axis=1)


def build_all(df: pd.DataFrame) -> dict:
    """Return {'H1': ..., 'H2': ..., 'H3': ...} aligned to df."""
    return {"H1": build_H1(df), "H2": build_H2(df), "H3": build_H3(df)}
