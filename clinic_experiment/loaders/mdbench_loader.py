"""Load 3MDBench cases (text-only -- image field is dropped per plan)."""
from __future__ import annotations

import random
from typing import Any, Dict, List


def load_cases(hf_dataset: str, hf_split: str, num_cases: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split=hf_split)
    if "image" in ds.column_names:
        ds = ds.remove_columns(["image"])

    cases: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        cases.append(
            {
                "case_id": str(i),
                "general_complaint": str(row.get("general_complaint") or "").strip(),
                "complaints": str(row.get("complaints") or "").strip(),
                "personality": str(row.get("personality") or "").strip(),
                "ground_truth_diagnosis": str(row.get("diagnosis") or "").strip(),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(cases)
    return cases[:num_cases]
