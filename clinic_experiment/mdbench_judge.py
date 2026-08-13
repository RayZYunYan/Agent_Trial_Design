"""Binary diagnosis judge for 3MDBench.

3MDBench's diagnosis space is a closed list of 34 labels (unlike AI
Hospital's open-ended diagnosis), so judging just needs a semantic match
between the extracted/predicted diagnosis text and the ground-truth label.
"""
from __future__ import annotations

from typing import Any, Dict

from smart_trial.models.model_client import ModelClient

JUDGE_PROMPT = (
    "You are a medical expert. Compare a doctor's stated diagnosis to the correct "
    "reference diagnosis for the same patient case. They may differ in wording, "
    "capitalization, or phrasing as long as they refer to the same underlying "
    "condition (synonyms count as a match).\n\n"
    "Reference diagnosis: {reference}\n"
    "Doctor's diagnosis: {predicted}\n\n"
    "Reply with only one word: CORRECT or INCORRECT."
)


def _parse_verdict(response: str) -> bool:
    text = response.upper()
    if "INCORRECT" in text:
        return False
    return "CORRECT" in text


def judge_diagnosis(
    judge_client: ModelClient, *, ground_truth: str, predicted_diagnosis_text: str
) -> Dict[str, Any]:
    predicted = (predicted_diagnosis_text or "").strip()
    if not predicted or predicted.lower() == "none":
        return {"correct": False, "predicted": predicted, "raw_judge_response": None}

    prompt = JUDGE_PROMPT.format(reference=ground_truth, predicted=predicted)
    response = judge_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
    return {
        "correct": _parse_verdict(response),
        "predicted": predicted,
        "raw_judge_response": response,
    }
