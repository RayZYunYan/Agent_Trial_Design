"""Q-learning vs A-learning agreement check — plan §3 sanity comparison."""
from __future__ import annotations

import pandas as pd


def compare_rules(q_rules: pd.DataFrame, a_rules: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row agreement on (π̂1, π̂2, π̂3). Requires same encounter_id order."""
    merged = q_rules[["encounter_id", "pi1", "pi2", "pi3"]].merge(
        a_rules[["encounter_id", "pi1", "pi2", "pi3"]],
        on="encounter_id",
        suffixes=("_q", "_a"),
    )
    merged["agree_pi1"] = merged["pi1_q"] == merged["pi1_a"]
    merged["agree_pi2"] = merged["pi2_q"] == merged["pi2_a"]
    merged["agree_pi3"] = merged["pi3_q"] == merged["pi3_a"]
    merged["agree_all"] = merged[["agree_pi1", "agree_pi2", "agree_pi3"]].all(axis=1)
    return merged


def agreement_summary(q_rules: pd.DataFrame, a_rules: pd.DataFrame) -> pd.Series:
    cmp_ = compare_rules(q_rules, a_rules)
    return pd.Series({
        "n": len(cmp_),
        "agree_pi1": cmp_["agree_pi1"].mean(),
        "agree_pi2": cmp_["agree_pi2"].mean(),
        "agree_pi3": cmp_["agree_pi3"].mean(),
        "agree_all": cmp_["agree_all"].mean(),
    })
