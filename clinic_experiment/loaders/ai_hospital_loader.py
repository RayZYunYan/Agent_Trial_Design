"""Load AI Hospital patient records into the generic clinic-experiment case schema."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _diagnosis(record: Dict[str, Any]) -> str:
    # Same fallback order as AI Hospital's own eval.py: 诊断结果 first, else 初步诊断.
    return str(record.get("诊断结果") or record.get("初步诊断") or "").strip()


def load_cases(source_path: Path, num_cases: int, seed: int) -> List[Dict[str, Any]]:
    """Mirrors AI Hospital's own patient_profile / medical_record split:
    Patient only ever sees 一般资料/现病史/既往史/个人史; 查体/辅助检查 are
    withheld from the patient and go to the Reporter agent instead."""
    with source_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    cases: List[Dict[str, Any]] = []
    for item in raw:
        record = item.get("raw_medical_record") or {}
        diagnosis = _diagnosis(record)
        if not diagnosis or not record.get("主诉"):
            continue
        cases.append(
            {
                "case_id": str(item.get("id")),
                "chief_complaint": str(record.get("主诉") or "").strip(),
                "patient_profile": str(record.get("一般资料") or "").strip(),
                "patient_history": {
                    k: record.get(k)
                    for k in ("主诉", "现病史", "既往史", "个人史")
                    if record.get(k)
                },
                "exam_data": {
                    "查体": record.get("查体") or "无异常",
                    "辅助检查": record.get("辅助检查") or "无异常",
                },
                "ground_truth_diagnosis": diagnosis,
                "reference_context": {
                    "symptom": record.get("现病史"),
                    "medical_test": record.get("辅助检查"),
                    "basis": record.get("诊断依据"),
                    "treatment": record.get("诊治经过"),
                },
            }
        )

    rng = random.Random(seed)
    rng.shuffle(cases)
    return cases[:num_cases]
