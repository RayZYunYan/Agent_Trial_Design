"""Fact coverage should count doctor-known initial text, not only dialogue."""
from mediq_experiment.score_coverage import _case_from_mediq_row, score_row
from smart_trial.core.judge import StageJudge, match_atomic_fact_to_context
from smart_trial.models.model_client import ModelClient


def test_match_initial_dizziness_facts():
    initial = (
        "A 64-year-old woman comes to the physician because of several episodes "
        "of dizziness during the last month."
    )
    assert match_atomic_fact_to_context(
        "A 64-year-old woman comes to the physician.", initial
    )
    assert match_atomic_fact_to_context(
        "She has experienced several episodes of dizziness in the last month.",
        initial,
    )
    assert (
        match_atomic_fact_to_context(
            "Episodes usually occur immediately after lying down.",
            initial,
        )
        is None
    )


def test_score_row_counts_initial_without_dialogue():
    row = {
        "id": 100,
        "interactive_system": {
            "letter_choice": "A",
            "questions": [],
            "answers": [],
        },
        "info": {
            "initial_info": (
                "A 64-year-old woman comes to the physician because of several "
                "episodes of dizziness during the last month."
            ),
            "question": "Which of the following is the most likely diagnosis?",
            "options": {
                "A": "Benign paroxysmal positional vertigo",
                "B": "Persistent postural-perceptual dizziness",
                "C": "Meniere disease",
                "D": "Acoustic neuroma",
            },
            "correct_answer_idx": "A",
            "facts": [
                "1. A 64-year-old woman comes to the physician.",
                "2. She has experienced several episodes of dizziness in the last month.",
                "3. Episodes usually occur immediately after lying down.",
                "4. She has no nausea.",
            ],
        },
    }
    scored = score_row(StageJudge(ModelClient("mock", "mock")), row)
    cov = scored["eval"]["fact_coverage"]
    assert cov["total_facts"] == 4
    assert cov["coverage_count"] >= 2
    assert cov["coverage_rate"] >= 0.5
    assert "context_coverage_count" not in cov
    assert "dialogue_coverage_count" not in cov
    sources = {item.get("evidence_source") for item in cov["covered_facts"]}
    assert "context" in sources


def test_case_from_mediq_row_includes_doctor_known_text():
    row = {
        "id": 1,
        "info": {
            "initial_info": "A patient has fever.",
            "question": "What is the diagnosis?",
            "options": {"A": "X", "B": "Y", "C": "Z", "D": "W"},
            "facts": ["Patient has fever."],
        },
    }
    case = _case_from_mediq_row(row)
    assert "fever" in case["doctor_known_text"].lower()
    assert "diagnosis" in case["doctor_known_text"].lower()
    assert "A: X" in case["doctor_known_text"]
