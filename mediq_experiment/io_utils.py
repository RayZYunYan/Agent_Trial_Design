"""Shared helpers for the MediQ experiment pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_subset_jsonl(
    source: Path,
    dest: Path,
    max_cases: Optional[int] = None,
    *,
    id_min: Optional[int] = None,
    id_max: Optional[int] = None,
) -> Path:
    """Filter cases by inclusive id range and/or take at most max_cases matches."""
    ensure_dir(dest.parent)
    count = 0
    with source.open(encoding="utf-8") as src, dest.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("id")
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            if id_min is not None and cid_int < int(id_min):
                continue
            if id_max is not None and cid_int > int(id_max):
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if max_cases is not None and count >= int(max_cases):
                break
    return dest


def normalize_facts(raw: Any) -> List[str]:
    """Strip leading '1. ' indices from MediQ fact lists."""
    if not raw:
        return []
    out: List[str] = []
    for item in raw:
        text = re.sub(r"^\d+\.\s*", "", str(item).strip())
        if text:
            out.append(text)
    return out


def transcript_to_history(questions: List[str], answers: List[str]) -> List[Dict[str, str]]:
    """MediQ Q/A lists → SMART judge conversation (assistant=doctor, user=patient)."""
    history: List[Dict[str, str]] = []
    for q, a in zip(questions or [], answers or []):
        history.append({"role": "assistant", "content": str(q)})
        history.append({"role": "user", "content": str(a)})
    return history


def format_options(options: Dict[str, Any]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in (options or {}).items())


def format_transcript(questions: List[str], answers: List[str], initial_info: str = "") -> str:
    lines: List[str] = []
    if initial_info:
        lines.append(f"Initial patient info: {initial_info}")
    for i, (q, a) in enumerate(zip(questions or [], answers or []), start=1):
        lines.append(f"Turn {i}")
        lines.append(f"Doctor: {q}")
        lines.append(f"Patient: {a}")
    return "\n".join(lines)


def extract_letter(text: str, valid: Optional[Iterable[str]] = None) -> Optional[str]:
    """Pull a single MCQ letter from model output."""
    allowed = {str(x).upper() for x in (valid or list("ABCDEFG"))}
    if not text:
        return None
    cleaned = text.strip().upper()
    if cleaned in allowed:
        return cleaned
    # Prefer explicit patterns
    for pattern in (
        r"\b(?:ANSWER|CHOICE|LETTER)\s*[:\-]\s*([A-G])\b",
        r"\b([A-G])\s*[.):]\s*",
        r"\b([A-G])\b",
    ):
        m = re.search(pattern, cleaned)
        if m and m.group(1) in allowed:
            return m.group(1)
    return None


def accuracy_summary(rows: List[Dict[str, Any]], correct_key: str = "correct") -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "correct": 0, "accuracy": None}
    correct = sum(1 for r in rows if r.get(correct_key))
    return {"n": n, "correct": correct, "accuracy": correct / n}
