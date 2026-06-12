from smart_trial.core.randomizer import TrialRandomizer


def test_stage2_pool_responder_vs_non_responder():
    r = TrialRandomizer(seed=42)
    case = {"case_id": "test_001", "case_category": "Other"}
    r_non = {"responder": False, "total": 3}
    r_yes = {"responder": True, "total": 8}
    a2 = r.assign_stage2_arm(case, r_non)
    assert a2["arm"] in ("A2a", "A2b")
    assert "A2c" not in a2["pool"]

    a2b = r.assign_stage2_arm(case, r_yes)
    assert a2b["arm"] in ("A2a", "A2b", "A2c")


def test_deterministic_assignment():
    r1 = TrialRandomizer(seed=99)
    r2 = TrialRandomizer(seed=99)
    case = {"case_id": "stable", "case_category": "Cardiology"}
    assert r1.assign_stage1_arm(case) == r2.assign_stage1_arm(case)
