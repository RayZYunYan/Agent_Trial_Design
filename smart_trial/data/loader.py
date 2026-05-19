"""
Case loading: local MediQ-style JSONL (repo data/) or HuggingFace `stellalisy/mediQ`.

Use `load_cases_from_config` with `trial_config.yaml` `data` section to pick the source.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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

CATEGORY_RED_FLAGS = {
    "Cardiology": [
        "radiation to left arm or jaw",
        "diaphoresis",
        "shortness of breath",
        "syncope",
        "family history of heart disease",
    ],
    "Neuro": [
        "sudden onset severe headache",
        "neck stiffness",
        "focal neurological deficit",
        "altered consciousness",
    ],
    "GI": [
        "blood in stool",
        "severe abdominal rigidity",
        "jaundice",
        "rapid weight loss",
    ],
    "Pulm": [
        "hemoptysis",
        "severe dyspnea at rest",
        "cyanosis",
        "oxygen saturation below 90 percent",
    ],
    "Infectious": [
        "high fever above 39C",
        "petechiae",
        "altered mental status",
        "rigors",
    ],
    "Pediatrics": [
        "altered consciousness",
        "high fever",
        "rash with fever",
        "severe dehydration",
    ],
    "Psychiatry": [
        "suicidal ideation",
        "homicidal ideation",
        "acute psychosis",
    ],
    "Other": ["severe pain", "rapid deterioration", "loss of consciousness"],
}

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _infer_category_from_text(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["chest pain", "heart", "cardiac", "mi ", "stemi"]):
        return "Cardiology"
    if any(w in text_lower for w in ["headache", "stroke", "seizure", "weakness", "numbness"]):
        return "Neuro"
    if any(w in text_lower for w in ["abdominal", "nausea", "vomit", "diarrhea", "gi "]):
        return "GI"
    if any(w in text_lower for w in ["cough", "shortness of breath", "sob", "lung"]):
        return "Pulm"
    if any(w in text_lower for w in ["fever", "infect", "sepsis"]):
        return "Infectious"
    return "Other"


def _extract_age_gender(text: str) -> tuple:
    age = "unknown"
    gender = "unknown"
    m_age = re.search(r"(\d{1,3})\s*[- ]?\s*year[s]?\s*[- ]?\s*old", text, re.I)
    if m_age:
        age = m_age.group(1)
    if re.search(r"\bmale\b|\bman\b|\bhe\b", text, re.I) and not re.search(r"\bfemale\b|\bwoman\b|\bshe\b", text, re.I):
        gender = "male"
    elif re.search(r"\bfemale\b|\bwoman\b|\bshe\b", text, re.I):
        gender = "female"
    return age, gender


def _chief_from_question(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    sentences = re.split(r"(?<=[.!?])\s+", first)
    return sentences[0][:400] if sentences else first[:400]


def ground_truth_option_text(case: Dict[str, Any]) -> str:
    """Map MCQ letter / idx to the option string for judges."""
    opts = case.get("options") or {}
    letter = case.get("ground_truth_idx") or case.get("ground_truth_answer") or ""
    letter = str(letter).strip().upper()[:1]
    if letter and isinstance(opts, dict) and letter in opts:
        return str(opts[letter])
    return str(case.get("ground_truth_answer", "") or "")


def load_local_data(filepath: str, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load cases from a local JSONL (e.g. data/all_dev_good.jsonl)."""
    path = Path(filepath)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cases.append(_parse_local_item(item))
            if max_cases and len(cases) >= max_cases:
                break
    return cases


def _parse_local_item(item: Dict[str, Any]) -> Dict[str, Any]:
    patient_meta = item.get("patient", {})

    gender = patient_meta.get("gender", "unknown").lower()
    if gender not in ("male", "female"):
        gender = "unknown"

    context = item.get("context", [])
    chief_complaint = context[0] if context else ""
    full_record = " ".join(context)

    gpt_specialty = patient_meta.get("gpt_specialty", "Other")
    category = SPECIALTY_TO_CATEGORY.get(gpt_specialty, "Other")

    case = {
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
    case["ground_truth_label"] = ground_truth_option_text(case)
    return case


def load_imedqa(split: str = "validation", max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load `stellalisy/mediQ` from HuggingFace (requires `datasets` and network).
    Normalized keys match `load_local_data` where possible.
    """
    from datasets import load_dataset

    ds = load_dataset("stellalisy/mediQ", split=split)
    n = len(ds) if max_cases is None else min(len(ds), max_cases)
    ds = ds.select(range(n))

    cases: List[Dict[str, Any]] = []
    for i, item in enumerate(ds):
        row = item if isinstance(item, dict) else dict(item)
        cases.append(_parse_hf_medqa_row(row, split=split, index=i))
    return cases


def _normalize_options(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {str(k).upper()[:1]: str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        letters = "ABCDEFGHIJ"
        return {letters[i]: str(opt) for i, opt in enumerate(raw) if i < len(letters)}
    return {}


def _parse_hf_medqa_row(item: Dict[str, Any], split: str, index: int) -> Dict[str, Any]:
    question = (item.get("question") or item.get("sentences") or "").strip()
    if isinstance(question, list):
        question = " ".join(str(s) for s in question)

    options = _normalize_options(item.get("options") or item.get("choices"))

    ans = item.get("answer")
    if ans is None:
        ans = item.get("answer_idx") or item.get("cop") or ""
    letter = str(ans).strip().upper()[:1]
    if letter and letter not in options and len(str(ans)) == 1:
        letter = str(ans).upper()

    chief = item.get("chief_complaint") or _chief_from_question(question)
    age, gender = _extract_age_gender(question[:1200])
    meta_age = item.get("age")
    meta_gender = item.get("gender")
    if meta_age:
        age = str(meta_age)
    if meta_gender:
        gender = str(meta_gender).lower()
        if gender not in ("male", "female"):
            gender = "unknown"

    meta = item.get("meta_info")
    specialty = item.get("specialty")
    if specialty is None and isinstance(meta, dict):
        specialty = meta.get("specialty")
    category = SPECIALTY_TO_CATEGORY.get(str(specialty or ""), _infer_category_from_text(question))

    case_id = f"imedqa_{split}_{index:04d}"
    case = {
        "case_id": case_id,
        "chief_complaint": chief or "unspecified complaint",
        "age": age,
        "gender": gender,
        "full_record": question,
        "atomic_facts": [],
        "ground_truth_answer": letter or str(ans),
        "ground_truth_idx": letter,
        "question": question,
        "options": options,
        "case_category": category,
        "specialties": [specialty] if specialty else [],
        "red_flags": CATEGORY_RED_FLAGS.get(category, CATEGORY_RED_FLAGS["Other"]),
    }
    case["ground_truth_label"] = ground_truth_option_text(case)
    return case


def load_red_flag_cache(filepath: str) -> Dict[str, Any]:
    path = Path(filepath)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_red_flag_cache(cases: List[Dict[str, Any]], cache: Dict[str, Any]) -> None:
    for case in cases:
        cid = case["case_id"]
        if cid in cache:
            case["red_flags"] = cache[cid]


def resolve_data_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def load_cases_from_config(
    config: Dict[str, Any],
    *,
    apply_red_flags_from_config: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load cases using `data` section from trial_config.yaml.

    source: `huggingface` | `local`
    """
    data_cfg = config.get("data") or {}
    source = (data_cfg.get("source") or "local").lower()
    if source == "huggingface":
        hf = data_cfg.get("huggingface") or {}
        cases = load_imedqa(
            split=str(hf.get("split", "validation")),
            max_cases=hf.get("max_cases"),
        )
    else:
        local = data_cfg.get("local") or {}
        path = local.get("path", "data/all_dev_good.jsonl")
        max_cases = local.get("max_cases")
        cases = load_local_data(path, max_cases=max_cases)

    if apply_red_flags_from_config:
        cache_path = data_cfg.get("red_flag_cache")
        if cache_path:
            path = resolve_data_path(str(cache_path))
            if path.is_file():
                cache = load_red_flag_cache(str(path))
                apply_red_flag_cache(cases, cache)
    return cases


def pick_diverse_cases(cases: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """Pick up to n cases with distinct case_category values (first match per category)."""
    picked: List[Dict[str, Any]] = []
    seen: set = set()
    for case in cases:
        cat = case.get("case_category", "Other")
        if cat in seen:
            continue
        seen.add(cat)
        picked.append(case)
        if len(picked) >= n:
            break
    return picked


def get_case_by_id(cases: List[Dict[str, Any]], case_id: str) -> Optional[Dict[str, Any]]:
    for c in cases:
        if c.get("case_id") == case_id:
            return c
    return None
