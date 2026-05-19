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
