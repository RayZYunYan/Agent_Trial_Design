from smart_trial.core.judge import StageJudge
from smart_trial.models.model_client import ModelClient


def _sample_case():
    return {
        "case_id": "test_judge_001",
        "chief_complaint": "chest pain",
        "age": "55",
        "gender": "male",
        "ground_truth_label": "Acute coronary syndrome",
        "red_flags": ["radiation to jaw", "diaphoresis"],
    }


def test_compute_r1_mock_returns_responder():
    judge = StageJudge(ModelClient("mock", "mock"), r1_responder_threshold=6)
    history = [
        {"role": "assistant", "content": "When did the pain start?"},
        {"role": "user", "content": "Two hours ago."},
    ]
    r1 = judge.compute_R1(history, _sample_case())
    assert r1["total"] >= 6
    assert r1["responder"] is True


def test_compute_r2_low_confidence_pool():
    judge = StageJudge(ModelClient("mock", "mock"), r2_high_confidence_threshold=0.7)
    r2 = judge.compute_R2([], [], high_threshold=0.7, stage2_conversation_slice=None, case=_sample_case())
    assert r2["confidence_level"] in ("high", "low")
    assert "final_confidence" in r2


def test_evaluate_outcome_red_flag_miss():
    judge = StageJudge(ModelClient("mock", "mock"))
    case = _sample_case()
    history = [{"role": "assistant", "content": "Tell me about the pain."}]
    r2 = {"confidence_level": "low", "final_confidence": 0.4}
    outcome = judge.evaluate_outcome(
        final_diagnosis="Likely musculoskeletal chest wall pain.",
        case=case,
        conversation_history=history,
        R2=r2,
    )
    assert "diag_correct" in outcome
    assert "red_flag_miss" in outcome
    assert outcome["red_flag_miss"] is True


def test_sanitize_coverage_delta_rejects_bad_evidence():
    judge = StageJudge(ModelClient("mock", "mock"))
    atomic_facts = ["fever", "blood culture positive", "joint swelling"]
    conversation = [
        {"role": "assistant", "content": "Any fever?"},
        {"role": "user", "content": "Yes, I have had a fever for two days."},
    ]
    parsed = {
        "newly_covered": [
            {
                "index": 1,
                "fact": "fever",
                "patient_evidence": "Yes, I have had a fever for two days.",
            },
            {
                "index": 2,
                "fact": "blood culture positive",
                "patient_evidence": "my knee is swollen",
            },
            {
                "index": 3,
                "fact": "joint swelling",
                "patient_evidence": "not sure",
            },
            {
                "index": 1,
                "fact": "fever",
                "patient_evidence": "duplicate index",
            },
        ]
    }
    result = judge._sanitize_coverage_delta(
        parsed,
        atomic_facts,
        covered_set=set(),
        new_conversation=conversation,
    )
    assert len(result["newly_covered"]) == 1
    assert result["newly_covered"][0]["index"] == 1


def test_sanitize_coverage_delta_requires_patient_quote():
    judge = StageJudge(ModelClient("mock", "mock"))
    atomic_facts = ["elevated WBC"]
    conversation = [{"role": "user", "content": "My white count was high on labs."}]
    parsed = {
        "newly_covered": [
            {
                "index": 1,
                "fact": "elevated WBC",
                "patient_evidence": "culture grew staph",
            }
        ]
    }
    result = judge._sanitize_coverage_delta(
        parsed,
        atomic_facts,
        covered_set=set(),
        new_conversation=conversation,
    )
    assert result["newly_covered"] == []


def test_sanitize_coverage_delta_rejects_lab_mismatch():
    judge = StageJudge(ModelClient("mock", "mock"))
    atomic_facts = ["Culture of joint fluid shows a bacteria."]
    conversation = [
        {"role": "user", "content": "I haven't really noticed anything weird when I pee."},
    ]
    parsed = {
        "newly_covered": [
            {
                "index": 1,
                "fact": "Culture of joint fluid shows a bacteria.",
                "patient_evidence": "I haven't really noticed anything weird when I pee.",
            }
        ]
    }
    result = judge._sanitize_coverage_delta(
        parsed,
        atomic_facts,
        covered_set=set(),
        new_conversation=conversation,
    )
    assert result["newly_covered"] == []


def test_filter_scorable_atomic_facts_excludes_physician_orders():
    from smart_trial.core.judge import filter_scorable_atomic_facts

    facts = filter_scorable_atomic_facts(
        [
            "Patient has fever.",
            "Physician orders antibiotic therapy for the patient.",
            "Culture positive.",
        ]
    )
    assert len(facts) == 2
    assert all("Physician orders" not in f for f in facts)


def test_context_coverage_matches_chief_complaint_demographics():
    from smart_trial.core.coverage import CoverageSession
    from smart_trial.core.judge import StageJudge

    case = {
        "case_id": "ctx_demo",
        "chief_complaint": (
            "A 21-year-old sexually active male complains of fever, pain during urination, "
            "and inflammation and pain in the right knee."
        ),
        "atomic_facts": [
            "Patient is a 21-year-old male.",
            "Patient is sexually active.",
            "Patient complains of fever.",
            "Culture of joint fluid shows a bacteria.",
        ],
    }
    history = [
        {
            "role": "assistant",
            "content": (
                "Hello, I'm your doctor today. A 21-year-old sexually active male complains of "
                "fever, pain during urination, and inflammation and pain in the right knee."
            ),
        },
        {"role": "user", "content": "Okay."},
    ]
    session = CoverageSession(StageJudge(ModelClient("mock", "mock")), case, threshold=0.5, check_every=5)
    session.seed_context_coverage(history)
    covered = {item["index"] for item in session.covered_facts}
    assert 1 in covered
    assert 2 in covered
    assert 3 in covered
    assert 4 not in covered
    assert session.covered_facts[0].get("evidence_source") == "context"


def test_probe_pending_flushes_between_checks():
    from smart_trial.core.coverage import CoverageSession
    from smart_trial.core.judge import StageJudge

    case = {
        "case_id": "flush",
        "chief_complaint": "fever",
        "atomic_facts": ["f1", "f2", "f3", "f4", "f5"],
    }
    session = CoverageSession(StageJudge(ModelClient("mock", "mock")), case, threshold=0.5, check_every=5)
    hist = [{"role": "assistant", "content": "Hi?"}, {"role": "user", "content": "Day 1."}]
    session.probe_if_due(5, hist)
    after_first = session.coverage_count
    hist.append({"role": "assistant", "content": "More?"})
    hist.append({"role": "user", "content": "Day 2."})
    session.probe_pending(hist, turn=6)
    assert session.coverage_count >= after_first


def test_evasive_evidence_allows_substantive_lab_reply():
    judge = StageJudge(ModelClient("mock", "mock"))
    evidence = (
        "Why does that matter? From what I remember, I don't think the bacteria "
        "from my cultures ferments maltose."
    )
    assert judge._is_evasive_evidence(evidence) is False
