"""Cross-model diagnosis for 3MDBench: frozen dialogue transcript from one
doctor's consultation -> the OTHER doctor gives a diagnosis-only response
(from the closed 34-label list), scored by mdbench_judge."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from smart_trial.models.model_client import ModelClient

from clinic_experiment.mdbench_dialogue import DIAGNOSIS_LIST
from clinic_experiment.mdbench_judge import judge_diagnosis

CROSS_DIAGNOSIS_PROMPT = (
    "You are a highly experienced doctor. Another doctor has already interviewed this "
    "patient online. Based ONLY on the dialogue below, state your single diagnosis, "
    f"chosen from this list: {DIAGNOSIS_LIST}\n\n"
    "Do not ask any questions. Reply with only the diagnosis name.\n\n"
    "Dialogue:\n{transcript}"
)


def format_transcript(dialog_history: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in dialog_history)


def _load_existing_by_case_id(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[str(row["case_id"])] = row
    return rows


def run_cross_finalize(
    records: List[Dict[str, Any]],
    *,
    responder_client: ModelClient,
    judge_client: ModelClient,
    path: Optional[Path] = None,
    fresh: bool = False,
) -> List[Dict[str, Any]]:
    """records: [{case_id, ground_truth_diagnosis, dialog_history}, ...]

    If `path` is given, results are appended there as they're produced and
    cases already present (by case_id) are skipped -- resume-friendly.
    """
    if path and fresh:
        path.write_text("", encoding="utf-8")
        existing: Dict[str, Dict[str, Any]] = {}
    else:
        existing = _load_existing_by_case_id(path)

    results: List[Dict[str, Any]] = []
    for rec in records:
        cid = str(rec["case_id"])
        if cid in existing:
            results.append(existing[cid])
            continue

        transcript = format_transcript(rec["dialog_history"])
        prompt = CROSS_DIAGNOSIS_PROMPT.format(transcript=transcript)
        responder_text = responder_client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        judged = judge_diagnosis(
            judge_client,
            ground_truth=rec["ground_truth_diagnosis"],
            predicted_diagnosis_text=responder_text,
        )
        row = {
            "case_id": rec["case_id"],
            "responder_diagnosis_text": responder_text,
            "correct": judged["correct"],
        }
        if path:
            from mediq_experiment.io_utils import append_jsonl

            append_jsonl(path, row)
        results.append(row)
    return results


def accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    return {"n": n, "correct": correct, "accuracy": (correct / n) if n else None}
