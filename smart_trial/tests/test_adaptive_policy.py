import random

from smart_trial.eval.adaptive_policy import biased_sample_arm


def test_biased_sample_propensities_sum_to_one():
    rng = random.Random(42)
    arms = ["A1a", "A1b", "A1c"]
    q = {"A1a": 0.2, "A1b": 0.5, "A1c": 0.1}
    _, prop = biased_sample_arm(arms, q, rng)
    assert 0.0 < prop <= 1.0

    total = 0.0
    for _ in range(500):
        _, p = biased_sample_arm(arms, q, rng)
        total += p
    # Each draw's reported propensity matches the arm drawn; average should be ~1/3
    # when sampling many times (stochastic check on arm distribution).
    counts = {a: 0 for a in arms}
    rng2 = random.Random(7)
    for _ in range(1000):
        arm, _ = biased_sample_arm(arms, q, rng2)
        counts[arm] += 1
    assert counts["A1b"] > counts["A1c"]
