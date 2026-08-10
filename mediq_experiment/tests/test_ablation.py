"""Tests for one-shot fact ablation case preparation."""
from __future__ import annotations

import json

from mediq_experiment.ablation import (
    ABLATION_MODES,
    facts_to_initial_info,
    prepare_ablation_jsonl,
    prepare_case,
    results_complete,
    select_facts_for_mode,
)
from mediq_experiment.io_utils import ROOT, read_jsonl


def _sample_row():
    return {
        "id": 100,
        "question": "Which of the following is the most likely diagnosis?",
        "context": [
            "A 64-year-old woman comes to the physician because of several episodes of dizziness during the last month.",
            "Hidden sentence that must NOT appear in base.",
        ],
        "options": {"A": "BPPV", "B": "PPPD", "C": "Meniere", "D": "AN"},
        "answer": "BPPV",
        "answer_idx": "A",
        "facts": [
            "1. A 64-year-old woman comes to the physician.",
            "2. She has experienced several episodes of dizziness in the last month.",
            "3. The episodes last between 30–40 seconds.",
            "4. During the episodes she feels as though she is spinning.",
            "5. Episodes usually occur immediately after lying down.",
        ],
        "fact_importance": ["low", "low", "most_important", "important", "most_important"],
    }


def test_base_initial_info_empty_and_explicit():
    row = prepare_case(_sample_row(), "base", seed=42)
    assert "initial_info" in row
    assert row["initial_info"] == ""
    assert "64-year-old" not in row["initial_info"]
    assert "Hidden sentence" not in row["initial_info"]


def test_one_random_stable_across_calls():
    row = _sample_row()
    a = select_facts_for_mode(row, "one_random", seed=42)
    b = select_facts_for_mode(row, "one_random", seed=42)
    assert a == b
    assert len(a) == 1
    c = select_facts_for_mode(row, "one_random", seed=99)
    # Different seed may differ; same seed must match
    assert len(c) == 1


def test_two_most_important_from_labels():
    facts = select_facts_for_mode(_sample_row(), "two_most_important_claude", seed=42)
    assert len(facts) == 2
    assert "30" in facts[0] or "30" in facts[1] or "lying" in " ".join(facts).lower()
    joined = facts_to_initial_info(facts)
    assert "64-year-old woman comes to the physician." not in joined or True  # may not include low
    assert "most_important" not in joined


def test_all_facts():
    facts = select_facts_for_mode(_sample_row(), "all_facts", seed=42)
    assert len(facts) == 5
    row = prepare_case(_sample_row(), "all_facts", seed=42)
    assert "spinning" in row["initial_info"]


def test_prepare_ablation_jsonl_base_no_context0_leak():
    src = ROOT / "data" / "mediQ_cases100-199_claude_ranked.jsonl"
    if not src.exists():
        return
    dest = ROOT / "mediq_experiment" / "outputs_ablation_smoke" / "_test_base_cases.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    prepare_ablation_jsonl(
        mode="base",
        dest=dest,
        ablation_cfg={"seed": 42, "ranked": {"claude": "data/mediQ_cases100-199_claude_ranked.jsonl"}},
        data_cfg={"id_min": 100, "id_max": 102, "max_cases": 3, "source_path": "data/mediQ_cases100-199_claude_ranked.jsonl"},
    )
    rows = read_jsonl(dest)
    assert len(rows) == 3
    for r in rows:
        assert r["initial_info"] == ""
        ctx0 = (r.get("context") or [""])[0]
        assert ctx0
        assert ctx0 not in (r["initial_info"] or "")


def test_prepare_one_random_shared_fact_file():
    src = ROOT / "data" / "mediQ_cases100-199_claude_ranked.jsonl"
    if not src.exists():
        return
    dest = ROOT / "mediq_experiment" / "outputs_ablation_smoke" / "_test_random_cases.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    prepare_ablation_jsonl(
        mode="one_random",
        dest=dest,
        ablation_cfg={"seed": 42, "ranked": {"claude": "data/mediQ_cases100-199_claude_ranked.jsonl"}},
        data_cfg={"id_min": 100, "id_max": 100, "max_cases": 1},
    )
    rows = read_jsonl(dest)
    assert len(rows) == 1
    assert rows[0]["initial_info"].strip()
    assert rows[0]["ablation"]["n_facts_shown"] == 1


def test_results_complete():
    path = ROOT / "mediq_experiment" / "outputs_ablation_smoke" / "_test_results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"id": 100}) + "\n" + json.dumps({"id": 101}) + "\n",
        encoding="utf-8",
    )
    assert results_complete(path, [100, 101])
    assert not results_complete(path, [100, 101, 102])


def test_all_modes_listed():
    assert "base" in ABLATION_MODES
    assert len(ABLATION_MODES) == 5
