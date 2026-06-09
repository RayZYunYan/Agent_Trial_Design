from smart_trial.q_learning.features import build_H1, build_H2, build_H3


def test_h1_shapes_and_columns(toy_df):
    H1 = build_H1(toy_df)
    assert len(H1) == len(toy_df)
    assert any(c.startswith("cat_") for c in H1.columns)
    assert "literacy_I" in H1.columns
    assert H1.isna().sum().sum() == 0


def test_h2_extends_h1(toy_df):
    H1 = build_H1(toy_df)
    H2 = build_H2(toy_df)
    assert set(H1.columns).issubset(set(H2.columns))
    for col in ["R1_total", "R1_responder", "A1_a", "A1_b", "A1_c"]:
        assert col in H2.columns


def test_h3_extends_h2(toy_df):
    H2 = build_H2(toy_df)
    H3 = build_H3(toy_df)
    assert set(H2.columns).issubset(set(H3.columns))
    for col in ["R2_final_conf", "R2_high", "A2_a", "A2_b", "A2_c"]:
        assert col in H3.columns
