"""Assemble a unified summary.json across every agent in outputs/<dataset>/<run_id>/.

The pipeline itself writes a summary.json at the END of each invocation, but
runs split by agent (e.g. Qwen phase, Llama phase) only cover their own agents.
This helper reads the per-agent JSONLs directly and produces a single summary
matching the shape run_pipeline writes.

Usage:
  python -m clinic_experiment.merge_summary --dataset mdbench --run-id X0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from mediq_experiment.io_utils import ROOT, load_yaml


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _accuracy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "correct": 0, "accuracy": None}
    scored = [r for r in rows if not r.get("judge_parse_failed")]
    correct = sum(1 for r in scored if r.get("correct"))
    return {
        "n": len(rows),
        "scored": len(scored),
        "correct": correct,
        "accuracy": (correct / len(scored)) if scored else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ai_hospital", "mdbench"], default="mdbench")
    ap.add_argument("--run-id", default="X0")
    ap.add_argument("--config", type=Path, default=ROOT / "clinic_experiment" / "config.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_dir = ROOT / "clinic_experiment" / "outputs" / args.dataset / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    agent_ids = [a["id"] for a in (cfg.get("agents") or [])]
    per_agent_meta = {
        a["id"]: f"{a['provider']}/{a['name']}" for a in (cfg.get("agents") or [])
    }

    dialogues = {aid: _load_jsonl(run_dir / f"agent_{aid}" / "dialogues.jsonl") for aid in agent_ids}
    base = {aid: _load_jsonl(run_dir / "base_prompt_only" / f"agent_{aid}.jsonl") for aid in agent_ids}
    oracle = {aid: _load_jsonl(run_dir / "all_facts" / f"agent_{aid}.jsonl") for aid in agent_ids}

    matrix: Dict[str, Dict[str, Any]] = {
        "base_prompt_only": {aid: _accuracy(base[aid]) for aid in agent_ids},
        **{f"dialogue_{aid}": {aid: _accuracy(dialogues[aid])} for aid in agent_ids},
        "all_facts": {aid: _accuracy(oracle[aid]) for aid in agent_ids},
    }

    n_cases = max((len(dialogues[aid]) for aid in agent_ids), default=0)
    summary = {
        "dataset": args.dataset,
        "run_id": args.run_id,
        "n_cases": n_cases,
        "agents": per_agent_meta,
        "dialogue_stats": {
            aid: {
                "rows": len(dialogues[aid]),
                "mean_turns": (
                    sum(r.get("n_turns", 0) for r in dialogues[aid]) / len(dialogues[aid])
                    if dialogues[aid]
                    else None
                ),
                "turn_cap_hits": sum(1 for r in dialogues[aid] if not r.get("dialogue_ended", True)),
            }
            for aid in agent_ids
        },
        "matrix": matrix,
    }
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
