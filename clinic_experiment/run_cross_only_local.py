"""Standalone script for a collaborator's machine: runs ONLY the
cross-diagnosis pairs where agent E (Llama) is the responder, reusing frozen
dialogue transcripts already generated (on the main machine) for all 5
agents. D (Qwen) stays on the main machine. This offloads half of the
slowest, non-parallelizable part of the AI Hospital / 3MDBench matrix to a
second machine.

What you need before running this:
  1. Same repo, same branch (new_dataset), same Python venv set up
     (pip install -r requirements.txt, plus mlx-lm).
  2. The handoff data dropped in place -- see --data-dir below. You should
     have received a zip with agent_A/dialogues.jsonl, agent_B/..., etc.
     Unzip it so the structure is:
       clinic_experiment/handoff_data/ai_hospital/agent_A/dialogues.jsonl
       clinic_experiment/handoff_data/ai_hospital/agent_B/dialogues.jsonl
       ... (one folder per agent, same filenames as the main pipeline)
  3. Your own ANTHROPIC_API_KEY in .env (used for judging only -- no OpenAI
     key needed, since A/B/C are never called here, only referenced as
     frozen text).
  4. Only the Llama (E) local MLX server -- no need to run Qwen (D) on your
     machine:
       python -m mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
         --port 8082 --max-tokens 4096 --prompt-cache-size 1

Usage:
  python -m clinic_experiment.run_cross_only_local --dataset ai_hospital
  python -m clinic_experiment.run_cross_only_local --dataset mdbench

Output goes to clinic_experiment/handoff_outputs/<dataset>/cross/*.jsonl --
send that folder back so it can be merged into the main run's cross/ folder
(same filenames, same schema, so it's a straight drop-in).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from mediq_experiment.io_utils import ROOT, ensure_dir, load_yaml
from smart_trial.models.model_client import ModelClient

from clinic_experiment.loaders.ai_hospital_loader import load_cases as load_ai_hospital_cases
from clinic_experiment.ai_hospital_cross_finalize import run_cross_finalize, accuracy
from clinic_experiment.loaders.mdbench_loader import load_cases as load_mdbench_cases
from clinic_experiment.mdbench_cross_finalize import (
    run_cross_finalize as run_mdbench_cross_finalize,
    accuracy as mdbench_accuracy,
)

# Only E (Llama) as responder -- D (Qwen) stays on the main machine. B->E may
# already be done there -- listed anyway so re-running this script against a
# fuller data dump later needs no code changes (existing output files are
# skipped case-by-case, same resume logic as the main pipeline).
PAIRS = [
    ("A", "E"), ("B", "E"), ("C", "E"), ("D", "E"),
]


def _load_dialogues(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[str(row["case_id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D/E-as-responder cross-diagnosis only.")
    parser.add_argument("--dataset", choices=["ai_hospital", "mdbench"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "clinic_experiment" / "config.yaml")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "clinic_experiment" / "handoff_data",
        help="Directory containing agent_A/dialogues.jsonl etc (received from the main machine)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "clinic_experiment" / "handoff_outputs",
        help="Where to write resulting cross-diagnosis jsonl files (send these back)",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_yaml(args.config)

    judge_cfg = (cfg.get("models") or {}).get("judge", {}) or {}
    judge = ModelClient(
        judge_cfg.get("provider", "anthropic"),
        judge_cfg.get("name", "claude-haiku-4-5"),
        temperature=0.0,
    )

    local_agents_cfg = {
        spec["id"]: spec for spec in (cfg.get("agents") or []) if spec.get("provider") == "mlx_local"
    }
    if not local_agents_cfg:
        raise RuntimeError("config.yaml has no mlx_local agents (D/E) -- check config")
    local_clients = {
        aid: ModelClient(spec["provider"], spec["name"], base_url=spec.get("base_url"), temperature=0.0)
        for aid, spec in local_agents_cfg.items()
    }

    data_dir = args.data_dir / args.dataset
    out_dir = ensure_dir(args.output_dir / args.dataset / "cross")

    if args.dataset == "ai_hospital":
        data_cfg = (cfg.get("data") or {}).get("ai_hospital") or {}
        cases = load_ai_hospital_cases(
            ROOT / str(data_cfg.get("source_path", "data/ai_hospital_patients.json")),
            int(data_cfg.get("num_cases", 100)),
            int(data_cfg.get("seed", 42)),
        )
        ref_by_id = {
            str(c["case_id"]): {
                **(c.get("reference_context") or {}),
                "ground_truth_diagnosis": c.get("ground_truth_diagnosis"),
            }
            for c in cases
        }
        for src, dst in PAIRS:
            if dst not in local_clients:
                continue
            src_path = data_dir / f"agent_{src}" / "dialogues.jsonl"
            if not src_path.exists():
                print(f"skip {src}->{dst}: no dialogue data for {src} at {src_path}")
                continue
            src_rows = _load_dialogues(src_path)
            records = [
                {"case_id": cid, "reference_context": ref_by_id[cid], "dialog_history": row["dialog_history"]}
                for cid, row in src_rows.items() if cid in ref_by_id
            ]
            print(f"=== {src}'s dialogue -> {dst} diagnoses ({len(records)} cases) ===")
            results = run_cross_finalize(
                records,
                responder_client=local_clients[dst],
                judge_client=judge,
                path=out_dir / f"dialogue_{src}_diagnosis_{dst}.jsonl",
                max_workers=1,
            )
            print(f"    done: {accuracy(results)}")
    else:
        data_cfg = (cfg.get("data") or {}).get("mdbench") or {}
        cases = load_mdbench_cases(
            str(data_cfg.get("hf_dataset", "univanxx/3mdbench")),
            str(data_cfg.get("hf_split", "test")),
            int(data_cfg.get("num_cases", 100)),
            int(data_cfg.get("seed", 42)),
        )
        gt_by_id = {str(c["case_id"]): c["ground_truth_diagnosis"] for c in cases}
        for src, dst in PAIRS:
            if dst not in local_clients:
                continue
            src_path = data_dir / f"agent_{src}" / "dialogues.jsonl"
            if not src_path.exists():
                print(f"skip {src}->{dst}: no dialogue data for {src} at {src_path}")
                continue
            src_rows = _load_dialogues(src_path)
            records = [
                {"case_id": cid, "ground_truth_diagnosis": gt_by_id[cid], "dialog_history": row["dialog_history"]}
                for cid, row in src_rows.items() if cid in gt_by_id
            ]
            print(f"=== {src}'s dialogue -> {dst} diagnoses ({len(records)} cases) ===")
            results = run_mdbench_cross_finalize(
                records,
                responder_client=local_clients[dst],
                judge_client=judge,
                path=out_dir / f"dialogue_{src}_diagnosis_{dst}.jsonl",
                max_workers=1,
            )
            print(f"    done: {mdbench_accuracy(results)}")


if __name__ == "__main__":
    main()
