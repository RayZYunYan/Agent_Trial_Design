"""Assemble a unified summary.json + RESULTS.md across every agent in
outputs/<dataset>/<run_id>/.

The pipeline itself writes a summary.json at the END of each invocation, but
runs split by agent (e.g. Qwen phase, Llama phase) only cover their own
agents. This helper reads the per-agent JSONLs directly and produces:
  - summary.json: machine-readable full 5-agent x 5-agent matrix
  - RESULTS.md: human-readable matrix in AI-Hospital paper's layout

Cross-diagonal cells (dialogue_X_diagnosis_Y where X != Y) are populated
from cross/*.jsonl if cross_finalize was run; otherwise shown as "—".

Usage:
  python -m clinic_experiment.merge_summary --dataset mdbench --run-id X0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _fmt_cell(cell: Optional[Dict[str, Any]]) -> str:
    if cell is None:
        return "—"
    acc = cell.get("accuracy")
    n = cell.get("n", 0)
    if acc is None:
        return f"— (n={n})"
    return f"{acc*100:.0f}% (n={n})"


def _render_markdown(agent_ids: List[str], per_agent_meta: Dict[str, str],
                     rows: List[str], matrix: Dict[str, Dict[str, Any]],
                     n_cases: int, run_id: str) -> str:
    """AI Hospital paper's matrix format: rows = information source, cols =
    diagnosing model."""
    lines: List[str] = []
    lines.append(f"# mdbench {run_id} — accuracy matrix")
    lines.append("")
    lines.append(f"Cases per cell targeted: **{n_cases}** (actual n annotated per cell).")
    lines.append("")
    lines.append("Rows = information source given to the doctor.  ")
    lines.append("Columns = diagnosing model (which agent produced the final diagnosis).  ")
    lines.append("Cells marked **—** were not collected in this run "
                 "(cross-diagonal `dialogue_X × diagnose_Y` requires `pipeline.cross_finalize: true`).")
    lines.append("")
    header = "| | " + " | ".join(f"**{aid}**" for aid in agent_ids) + " |"
    sep = "|---|" + "|".join([":---:"] * len(agent_ids)) + "|"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        cells = [_fmt_cell(matrix.get(row, {}).get(aid)) for aid in agent_ids]
        pretty = row.replace("_", " ")
        lines.append(f"| **{pretty}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Model roster")
    lines.append("")
    for aid in agent_ids:
        lines.append(f"- **{aid}**: `{per_agent_meta.get(aid, '(unknown)')}`")
    lines.append("")
    lines.append("Patient / Reporter / Judge (shared across agents): `anthropic/claude-haiku-4-5` via aicode007 relay.")
    lines.append("")
    return "\n".join(lines)


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

    cross_dir = run_dir / "cross"
    cross: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if cross_dir.exists():
        for src in agent_ids:
            cross[src] = {}
            for dst in agent_ids:
                if dst == src:
                    continue
                p = cross_dir / f"dialogue_{src}_diagnosis_{dst}.jsonl"
                cross[src][dst] = _load_jsonl(p)

    # Full 5x5 matrix in the paper's layout.
    matrix: Dict[str, Dict[str, Any]] = {}
    matrix["base_prompt_only"] = {aid: _accuracy(base[aid]) for aid in agent_ids}
    for src in agent_ids:
        row: Dict[str, Any] = {}
        for dst in agent_ids:
            if dst == src:
                row[dst] = _accuracy(dialogues[src])
            else:
                cross_rows = (cross.get(src) or {}).get(dst, [])
                row[dst] = _accuracy(cross_rows) if cross_rows else None
        matrix[f"dialogue_{src}"] = row
    matrix["all_facts"] = {aid: _accuracy(oracle[aid]) for aid in agent_ids}

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
    out_json = run_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    row_order = ["base_prompt_only"] + [f"dialogue_{aid}" for aid in agent_ids] + ["all_facts"]
    md = _render_markdown(agent_ids, per_agent_meta, row_order, matrix, n_cases, args.run_id)
    out_md = run_dir / "RESULTS.md"
    out_md.write_text(md, encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print()
    print(md)


if __name__ == "__main__":
    main()
