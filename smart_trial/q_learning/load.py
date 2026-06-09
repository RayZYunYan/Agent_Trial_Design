"""Load encounter JSONL files into a flat pandas DataFrame keyed by encounter_id.

Accepts either a directory of *.jsonl files or a single encounters.jsonl file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from .config import DEFAULT_ENCOUNTERS_DIR, DEFAULT_ENCOUNTERS_FILE, OUTCOME_KEY


REQUIRED_TOP_LEVEL = [
    "encounter_id",
    "case_id",
    "case_category",
    "stage1_arm",
    "R1",
    "stage2_arm",
    "R2",
    "stage3_arm",
    "outcome",
    "trajectory",
]

# Map persona fields → literacy_F / literacy_I / literacy_C for H1 features.
# New PatientPersona uses vocabulary_register (Nutbeam functional/interactive/critical).
# Legacy logs used language_level (basic/intermediate/advanced) from pre-unified personas.
_REGISTER_TO_LITERACY = {
    "functional": "literacy_F",
    "interactive": "literacy_I",
    "critical": "literacy_C",
    "f": "literacy_F",
    "i": "literacy_I",
    "c": "literacy_C",
}
_LANGUAGE_LEVEL_TO_LITERACY = {
    "basic": "literacy_F",
    "intermediate": "literacy_I",
    "advanced": "literacy_C",
}
_PERSONA_ID_TO_LITERACY = {
    "literacy_F": "literacy_F",
    "literacy_I": "literacy_I",
    "literacy_C": "literacy_C",
    "low_literacy": "literacy_F",
    "default": "literacy_I",
}


def _register_value_to_literacy(value: Any) -> Optional[str]:
    """Normalize a register / language_level string to literacy_F/I/C."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in _REGISTER_TO_LITERACY:
        return _REGISTER_TO_LITERACY[key]
    if key in _LANGUAGE_LEVEL_TO_LITERACY:
        return _LANGUAGE_LEVEL_TO_LITERACY[key]
    return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _literacy_from_persona_block(block: Dict[str, Any]) -> Optional[str]:
    """Derive literacy_F/I/C from a logged persona dict (main or legacy format)."""
    pid = block.get("persona_id")
    if pid is not None:
        pid_str = str(pid)
        if pid_str in _PERSONA_ID_TO_LITERACY:
            return _PERSONA_ID_TO_LITERACY[pid_str]
        if pid_str.startswith("literacy_"):
            return pid_str

    for field in ("vocabulary_register", "jargon_comprehension", "language_level"):
        lit = _register_value_to_literacy(block.get(field))
        if lit:
            return lit

    # Legacy HealthLiteracyPersona axes (F / I / C) nested under "axes".
    axes = block.get("axes")
    if isinstance(axes, dict):
        for field in ("vocabulary_register", "jargon_comprehension", "language_level"):
            lit = _register_value_to_literacy(axes.get(field))
            if lit:
                return lit

    return None


def _resolve_literacy_id(enc: Dict[str, Any]) -> Optional[str]:
    lp = enc.get("literacy_persona")
    if isinstance(lp, dict):
        lit = _literacy_from_persona_block(lp)
        if lit:
            return lit

    persona = enc.get("persona")
    if isinstance(persona, dict):
        lit = _literacy_from_persona_block(persona)
        if lit:
            return lit

    return None


def load_encounters(source: Optional[Union[Path, str]] = None) -> List[Dict[str, Any]]:
    """Read encounters from a directory, a single .jsonl file, or defaults."""
    if source is None:
        if DEFAULT_ENCOUNTERS_FILE.is_file():
            source = DEFAULT_ENCOUNTERS_FILE
        else:
            source = DEFAULT_ENCOUNTERS_DIR

    path = Path(source)
    out: List[Dict[str, Any]] = []

    if path.is_file() and path.suffix == ".jsonl":
        out.extend(_iter_jsonl(path))
        return out

    if path.is_dir():
        for jsonl_path in sorted(path.glob("*.jsonl")):
            out.extend(_iter_jsonl(jsonl_path))
        return out

    return out


def _validate(enc: Dict[str, Any]) -> bool:
    for key in REQUIRED_TOP_LEVEL:
        if key not in enc:
            return False
    if not isinstance(enc.get("R1"), dict) or not isinstance(enc.get("R2"), dict):
        return False
    if not isinstance(enc.get("outcome"), dict) or OUTCOME_KEY not in enc["outcome"]:
        return False
    return True


def to_dataframe(encounters: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten encounters into a tabular DataFrame; drops malformed rows."""
    rows: List[Dict[str, Any]] = []
    for enc in encounters:
        if not _validate(enc):
            continue
        r1 = enc["R1"]
        r2 = enc["R2"]
        traj = enc.get("trajectory", []) or []
        stage1_turns = [t for t in traj if t.get("stage") == 1]
        stage2_turns = [t for t in traj if t.get("stage") == 2]
        stage3_turns = [t for t in traj if t.get("stage") == 3]
        rows.append({
            "encounter_id": enc["encounter_id"],
            "case_id": enc["case_id"],
            "case_category": enc.get("case_category", "Other"),
            "seed": enc.get("seed"),
            "chief_complaint": enc.get("chief_complaint", ""),
            "A1": enc["stage1_arm"],
            "R1_total": r1.get("total"),
            "R1_responder": bool(r1.get("responder", False)),
            "R1_OPQRST": r1.get("OPQRST"),
            "R1_red_flags": r1.get("red_flags"),
            "R1_pmh": r1.get("past_medical_history"),
            "R1_meds": r1.get("medications_allergies"),
            "R1_social": r1.get("social_family_history"),
            "stage1_turns": len(stage1_turns),
            "A2": enc["stage2_arm"],
            "stage2_pool": enc.get("stage2_pool"),
            "R2_final_conf": r2.get("final_confidence"),
            "R2_avg_conf": r2.get("avg_confidence"),
            "R2_level": r2.get("confidence_level"),
            "stage2_turns": len(stage2_turns),
            "A3": enc["stage3_arm"],
            "stage3_pool": enc.get("stage3_pool"),
            "stage3_turns": len(stage3_turns),
            "literacy_id": _resolve_literacy_id(enc),
            "Y": int(bool(enc["outcome"].get(OUTCOME_KEY))),
            "outcome_red_flag_miss": bool(enc["outcome"].get("red_flag_miss", False)),
            "outcome_dangerous": bool(enc["outcome"].get("dangerous_advice", False)),
        })
    return pd.DataFrame(rows)


def load(source: Optional[Union[Path, str]] = None) -> pd.DataFrame:
    """Convenience: load encounters and return DataFrame."""
    return to_dataframe(load_encounters(source))
