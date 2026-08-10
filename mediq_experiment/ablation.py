"""One-shot fact ablation: prepare MediQ cases with controlled doctor-visible facts."""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from mediq_experiment.io_utils import ROOT, ensure_dir, normalize_facts, write_jsonl

ABLATION_MODES = (
    "base",
    "one_random",
    "two_most_important_claude",
    "two_most_important_gpt",
    "all_facts",
)

MOST_IMPORTANT_LABEL = "most_important"


def strip_fact_index(text: str) -> str:
    return re.sub(r"^\d+\.\s*", "", str(text).strip())


def _case_id_int(row: Dict[str, Any]) -> Optional[int]:
    try:
        return int(row.get("id"))
    except (TypeError, ValueError):
        return None


def load_cases(
    source: Path,
    *,
    id_min: Optional[int] = None,
    id_max: Optional[int] = None,
    max_cases: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with source.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = _case_id_int(row)
            if cid is None:
                continue
            if id_min is not None and cid < id_min:
                continue
            if id_max is not None and cid > id_max:
                continue
            rows.append(row)
            if max_cases is not None and len(rows) >= int(max_cases):
                break
    return rows


def _stable_index(seed: int, case_id: int, n: int) -> int:
    """Deterministic index in [0, n) shared across all doctors for a case."""
    if n <= 0:
        raise ValueError("cannot sample from empty fact list")
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % n


def select_facts_for_mode(
    row: Dict[str, Any],
    mode: str,
    *,
    seed: int = 42,
) -> List[str]:
    """Return the fact strings that become doctor-visible initial_info (may be empty)."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode}. Expected one of {ABLATION_MODES}")

    facts = normalize_facts(row.get("facts"))
    if mode == "base":
        return []

    if mode == "all_facts":
        return facts

    if mode == "one_random":
        if not facts:
            warnings.warn(f"case {row.get('id')}: no facts for one_random; using empty initial_info")
            return []
        cid = _case_id_int(row)
        if cid is None:
            raise ValueError(f"case missing integer id: {row.get('id')}")
        idx = _stable_index(seed, cid, len(facts))
        return [facts[idx]]

    # two_most_important_*
    labels = row.get("fact_importance") or []
    if len(labels) != len(row.get("facts") or []):
        # Allow mismatch vs raw facts list length; align to normalized facts by raw index
        raw = list(row.get("facts") or [])
        if len(labels) != len(raw):
            warnings.warn(
                f"case {row.get('id')}: fact_importance length {len(labels)} "
                f"!= facts length {len(raw)}; falling back to empty"
            )
            return []
        selected = [
            strip_fact_index(f)
            for f, lab in zip(raw, labels)
            if str(lab).strip().lower() == MOST_IMPORTANT_LABEL
        ]
    else:
        selected = [
            fact
            for fact, lab in zip(facts, labels)
            if str(lab).strip().lower() == MOST_IMPORTANT_LABEL
        ]

    if len(selected) != 2:
        warnings.warn(
            f"case {row.get('id')}: expected 2 most_important facts, got {len(selected)}"
        )
    return selected


def facts_to_initial_info(facts: Sequence[str]) -> str:
    """Join facts for PATIENT INFORMATION. Empty string for base (must be explicit)."""
    cleaned = [strip_fact_index(f) for f in facts if str(f).strip()]
    return "\n".join(cleaned)


def prepare_case(
    row: Dict[str, Any],
    mode: str,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Copy case and set initial_info explicitly so Patient never falls back to context[0].
    Also records ablation metadata for debugging / resume audits.
    """
    out = dict(row)
    selected = select_facts_for_mode(row, mode, seed=seed)
    out["initial_info"] = facts_to_initial_info(selected)
    out["ablation"] = {
        "mode": mode,
        "seed": seed,
        "n_facts_shown": len(selected),
        "facts_shown": list(selected),
    }
    return out


def source_path_for_mode(mode: str, ablation_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> Path:
    ranked = ablation_cfg.get("ranked") or {}
    if mode == "two_most_important_claude":
        rel = ranked.get("claude") or "data/mediQ_cases100-199_claude_ranked.jsonl"
        return ROOT / str(rel)
    if mode == "two_most_important_gpt":
        rel = ranked.get("gpt") or "data/mediQ_cases100-199_gpt_ranked.jsonl"
        return ROOT / str(rel)
    # base / one_random / all_facts: prefer ranked claude (has facts+labels) else data.source_path
    default = ranked.get("claude") or data_cfg.get("source_path") or "data/all_dev_good.jsonl"
    return ROOT / str(default)


def prepare_ablation_jsonl(
    *,
    mode: str,
    dest: Path,
    ablation_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
) -> Path:
    """Write prepared cases for one mode (shared by all doctors)."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode}")

    source = source_path_for_mode(mode, ablation_cfg, data_cfg)
    if not source.exists():
        raise FileNotFoundError(f"Ablation source not found for mode={mode}: {source}")

    id_min = data_cfg.get("id_min")
    id_max = data_cfg.get("id_max")
    max_cases = data_cfg.get("max_cases")
    if id_min is not None:
        id_min = int(id_min)
    if id_max is not None:
        id_max = int(id_max)

    seed = int(ablation_cfg.get("seed", 42))
    rows = load_cases(source, id_min=id_min, id_max=id_max, max_cases=max_cases)
    prepared = [prepare_case(row, mode, seed=seed) for row in rows]
    write_jsonl(dest, prepared)
    return dest


def expected_case_ids(prepared_path: Path) -> List[Any]:
    ids: List[Any] = []
    with prepared_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids.append(json.loads(line).get("id"))
    return ids


def results_complete(results_path: Path, expected_ids: Iterable[Any]) -> bool:
    """True when results.jsonl already contains every expected case id (resume gate)."""
    if not results_path.exists() or results_path.stat().st_size == 0:
        return False
    done = set()
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            done.add(json.loads(line).get("id"))
    expected = set(expected_ids)
    return bool(expected) and expected.issubset(done)
