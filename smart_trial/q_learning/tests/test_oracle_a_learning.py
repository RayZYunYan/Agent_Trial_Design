import numpy as np

from smart_trial.q_learning import a_learning, q_learning
from smart_trial.q_learning import consistency
from smart_trial.q_learning import oracle


def test_oracle_generate_schema():
    df = oracle.generate_synthetic_trajectories(30, seed=1)
    assert len(df) == 30
    assert "Y" in df.columns
    assert set(df["A1"].unique()).issubset({"A1a", "A1b", "A1c"})


def test_oracle_recovery_improves_with_n():
    spec = oracle.DEFAULT_DTR_SPEC
    result = oracle.recovery_rate([100, 400], n_replicates=5, seed=42, true_dtr_spec=spec)
    assert result["agreement_rate"].notna().all()
    assert (result["agreement_rate"] >= 0).all()
    assert (result["agreement_rate"] <= 1).all()


def test_a_learning_runs(toy_df):
    res = a_learning.fit(toy_df)
    assert len(res.rules) == len(toy_df)
    assert np.isfinite(res.value)


def test_q_a_agreement_runs(toy_df):
    q_res = q_learning.fit(toy_df)
    a_res = a_learning.fit(toy_df)
    summary = consistency.agreement_summary(q_res.rules, a_res.rules)
    assert summary["n"] == len(toy_df)
    assert 0.0 <= summary["agree_all"] <= 1.0
