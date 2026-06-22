"""
Build merged MediQ + CRAFT-MD JSONL for SMART trial (2685 cases).

Sources:
  - stellalisy/mediQ validation (1272)
  - stellalisy/mediQ test (1273)
  - data/all_craft_md.jsonl (140)

Usage (from repo root):
  python -m smart_trial.scripts.build_merged_dataset
  python -m smart_trial.scripts.build_merged_dataset --out data/all_mediq_craft_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CRAFT_PATH = PROJECT_ROOT / "data" / "all_craft_md.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "all_mediq_craft_merged.jsonl"


def _facts_from_hf(row: Dict[str, Any]) -> List[str]:
    facts_old = row.get("facts_old") or []
    atomic = row.get("atomic_facts") or []
    if facts_old:
        return [str(f) for f in facts_old]
    return [str(f) for f in atomic]


def _normalize_hf_row(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    context = list(row.get("context") or [])
    patient = row.get("patient") or {}
    if hasattr(patient, "items"):
        patient = dict(patient)
    else:
        patient = dict(patient) if patient else {}

    return {
        "question": row.get("question", ""),
        "context": context,
        "context_len": row.get("context_len") or len(context),
        "options": dict(row.get("options") or {}),
        "answer": row.get("answer", ""),
        "answer_idx": row.get("answer_idx", ""),
        "explanation": row.get("explanation") or "",
        "facts": _facts_from_hf(row),
        "patient": patient,
        "_source": source,
        "_source_id": row.get("id"),
    }


def _normalize_craft_row(row: Dict[str, Any]) -> Dict[str, Any]:
    context = list(row.get("context") or [])
    patient = dict(row.get("patient") or {})
    return {
        "question": row.get("question", ""),
        "context": context,
        "context_len": len(context),
        "options": dict(row.get("options") or {}),
        "answer": row.get("answer", ""),
        "answer_idx": row.get("answer_idx", ""),
        "explanation": "",
        "facts": list(row.get("facts") or []),
        "patient": patient,
        "_source": "craft_md",
        "_source_id": row.get("id"),
    }


def load_hf_split(split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("stellalisy/mediQ", split=split)
    return [_normalize_hf_row(dict(row), source=f"mediq_{split}") for row in ds]


def load_craft(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_normalize_craft_row(json.loads(line)))
    return rows


def build_merged(craft_path: Path) -> List[Dict[str, Any]]:
    val = load_hf_split("validation")
    test = load_hf_split("test")
    craft = load_craft(craft_path)
    merged = val + test + craft
    for i, row in enumerate(merged):
        row["id"] = i
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge mediQ val+test and CRAFT-MD into one JSONL")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument(
        "--craft",
        default=str(CRAFT_PATH),
        help="Path to all_craft_md.jsonl",
    )
    args = parser.parse_args()

    craft_path = Path(args.craft)
    if not craft_path.is_absolute():
        craft_path = PROJECT_ROOT / craft_path

    merged = build_merged(craft_path)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in merged:
        src = row.get("_source", "?")
        counts[src] = counts.get(src, 0) + 1

    print(f"Wrote {len(merged)} cases -> {out_path}")
    for src in sorted(counts):
        print(f"  {src}: {counts[src]}")


if __name__ == "__main__":
    main()
