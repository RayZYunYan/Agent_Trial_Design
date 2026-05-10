import json
from typing import List, Dict, Optional


SPECIALTY_TO_CATEGORY = {
    "Cardiology": "Cardiology",
    "Internal Medicine": "Cardiology",
    "Neurology": "Neuro",
    "Gastroenterology": "GI",
    "Pulmonology": "Pulm",
    "Infectious Disease": "Infectious",
    "Pediatrics": "Pediatrics",
    "Psychiatry": "Psychiatry",
    "Urology": "Other",
    "Surgery": "Other",
    "Orthopedics": "Other",
    "Dermatology": "Other",
    "Endocrinology": "Other",
    "Hematology": "Other",
    "Rheumatology": "Other",
    "Nephrology": "Other",
    "Oncology": "Other",
}

# Fallback red flags by category, used until red_flag_cache.json is generated
CATEGORY_RED_FLAGS = {
    "Cardiology": [
        "radiation to left arm or jaw", "diaphoresis",
        "shortness of breath", "syncope", "family history of heart disease",
    ],
    "Neuro": [
        "sudden onset severe headache", "neck stiffness",
        "focal neurological deficit", "altered consciousness",
    ],
    "GI": [
        "blood in stool", "severe abdominal rigidity",
        "jaundice", "rapid weight loss",
    ],
    "Pulm": [
        "hemoptysis", "severe dyspnea at rest",
        "cyanosis", "oxygen saturation below 90 percent",
    ],
    "Infectious": [
        "high fever above 39C", "petechiae",
        "altered mental status", "rigors",
    ],
    "Pediatrics": [
        "altered consciousness", "high fever", "rash with fever", "severe dehydration",
    ],
    "Psychiatry": [
        "suicidal ideation", "homicidal ideation", "acute psychosis",
    ],
    "Other": ["severe pain", "rapid deterioration", "loss of consciousness"],
}


def load_local_data(filepath: str, max_cases: Optional[int] = None) -> List[Dict]:
    """Load cases from a local JSONL file (e.g. data/all_dev_good.jsonl)."""
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases.append(_parse_item(item))
            if max_cases and len(cases) >= max_cases:
                break
    return cases


def _parse_item(item: Dict) -> Dict:
    patient_meta = item.get("patient", {})

    gender = patient_meta.get("gender", "unknown").lower()
    if gender not in ("male", "female"):
        gender = "unknown"

    context = item.get("context", [])
    chief_complaint = context[0] if context else ""
    full_record = " ".join(context)

    gpt_specialty = patient_meta.get("gpt_specialty", "Other")
    category = SPECIALTY_TO_CATEGORY.get(gpt_specialty, "Other")

    return {
        "case_id": f"medqa_{item['id']:04d}",
        "chief_complaint": chief_complaint,
        "age": patient_meta.get("age", "unknown"),
        "gender": gender,
        "full_record": full_record,
        "atomic_facts": item.get("facts", []),
        "ground_truth_answer": item.get("answer", ""),
        "ground_truth_idx": item.get("answer_idx", ""),
        "question": item.get("question", ""),
        "options": item.get("options", {}),
        "case_category": category,
        "specialties": patient_meta.get("specialties", []),
        "red_flags": CATEGORY_RED_FLAGS.get(category, CATEGORY_RED_FLAGS["Other"]),
    }


def load_red_flag_cache(filepath: str) -> Dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_red_flag_cache(cases: List[Dict], cache: Dict) -> None:
    """Override category-level red flags with per-case LLM-generated ones."""
    for case in cases:
        if case["case_id"] in cache:
            case["red_flags"] = cache[case["case_id"]]
