"""Cross-model diagnosis: frozen dialogue transcript from one doctor's
consultation -> the OTHER doctor gives a diagnosis-only response (no more
questions), scored by the same diagnosis_judge used for own-dialogue Final
Accuracy. Produces the off-diagonal cells of the 2x2 matrix (image2's
"Dialogue a-> Diagnosis b" style numbers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from smart_trial.models.model_client import ModelClient

from clinic_experiment.diagnosis_judge import judge_diagnosis

CROSS_DIAGNOSIS_PROMPT = (
    "你是一位医学专家。另一位医生已经完成了对这位病人的问诊，下面是完整的问诊记录"
    "（包括医生的提问、病人的回答、以及检查员反馈的检查结果）。\n\n"
    "请你仅根据这段问诊记录，按照下面的格式给出你的诊断结果、诊断依据和治疗方案，不要再向病人提问。\n\n"
    "#症状#\n(1)xx\n(2)xx\n\n"
    "#辅助检查#\n(1)xx\n(2)xx\n\n"
    "#诊断结果#\nxx\n\n"
    "#诊断依据#\n(1)xx\n(2)xx\n\n"
    "#治疗方案#\n(1)xx\n(2)xx\n\n"
    "问诊记录：\n{transcript}"
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
    """records: [{case_id, reference_context, dialog_history}, ...]

    reference_context must already include a "diagnosis" or
    "ground_truth_diagnosis" key alongside symptom/medical_test/basis/treatment.

    If `path` is given, results are appended there as they're produced and
    cases already present (by case_id) are skipped -- resume-friendly, same
    pattern as the main dialogue loop in run_pipeline.py.
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
        responder_text = responder_client.chat(
            [{"role": "user", "content": prompt}], temperature=0.0
        )
        judged = judge_diagnosis(
            judge_client,
            reference_context=rec["reference_context"],
            doctor_diagnosis_text=responder_text,
        )
        row = {
            "case_id": rec["case_id"],
            "responder_diagnosis_text": responder_text,
            "diagnosis_choice": judged.get("diagnosis_choice"),
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
