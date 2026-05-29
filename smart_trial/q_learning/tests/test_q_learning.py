import numpy as np

from smart_trial.q_learning import q_learning
from smart_trial.q_learning.config import STAGE2_POOLS, STAGE3_POOLS


def test_fit_returns_rules_aligned(toy_df):
    res = q_learning.fit(toy_df)
    assert len(res.rules) == len(toy_df)
    assert np.isfinite(res.value)
    # value is the row-mean of the optimal Q at H1
    assert abs(res.value - float(res.pseudo_y0.mean())) < 1e-9


def test_recommendations_respect_pool(toy_df):
    res = q_learning.fit(toy_df)
    for _, r in res.rules.iterrows():
        pool_s2 = STAGE2_POOLS["responder"] if r["R1_responder"] else STAGE2_POOLS["non-responder"]
        pool_s3 = STAGE3_POOLS.get(str(r["R2_level"]), STAGE3_POOLS["low"])
        assert r["pi2"] in pool_s2, f"pi2={r['pi2']} not in {pool_s2}"
        assert r["pi3"] in pool_s3, f"pi3={r['pi3']} not in {pool_s3}"
        assert r["pi1"] in {"A1a", "A1b", "A1c"}


def test_value_of_fixed_strategy_is_a_number(toy_df):
    v = q_learning.value_of_fixed_strategy(toy_df, "A1b", "A2a", "A3b")
    assert isinstance(v, float)
