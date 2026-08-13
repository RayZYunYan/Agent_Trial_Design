"""N-agent AI Hospital / 3MDBench experiment.

Each agent in config.yaml's `agents` list runs a full consultation against
the SAME shared Patient (and Reporter) model/config, so the only variable
between agents is the diagnosing/dialogue model itself -- patient behavior is
held constant. Every agent's frozen dialogue transcript is then cross-fed to
every OTHER agent for a diagnosis-only response, and each agent also gives a
"base prompt only" (zero case info) diagnosis. This produces a
(1 + n_agents) x n_agents accuracy matrix per dataset:

  rows:    Base prompt only, Agent_1 dialogue, Agent_2 dialogue, ...
  columns: Agent_1 diagnosis, Agent_2 diagnosis, ...

("All facts" row -- diagnosis from a direct fact list, no dialogue -- is not
yet implemented; it needs the atomic-fact-extraction step, still pending.)

Usage (from repo root):
  python -m clinic_experiment.run_pipeline --dataset ai_hospital --max-cases 5   # smoke test
  python -m clinic_experiment.run_pipeline --dataset ai_hospital                # full run (config.yaml num_cases)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mediq_experiment.io_utils import ROOT, ensure_dir, load_yaml, append_jsonl
from smart_trial.models.model_client import ModelClient

from clinic_experiment.loaders.ai_hospital_loader import load_cases as load_ai_hospital_cases
from clinic_experiment.ai_hospital_dialogue import run_consultation
from clinic_experiment.diagnosis_judge import judge_diagnosis
from clinic_experiment.ai_hospital_cross_finalize import run_cross_finalize, accuracy

from clinic_experiment.loaders.mdbench_loader import load_cases as load_mdbench_cases
from clinic_experiment.mdbench_dialogue import (
    run_consultation as run_mdbench_consultation,
    extract_diagnosis,
)
from clinic_experiment.mdbench_judge import judge_diagnosis as judge_mdbench_diagnosis
from clinic_experiment.mdbench_cross_finalize import (
    run_cross_finalize as run_mdbench_cross_finalize,
    accuracy as mdbench_accuracy,
)

from clinic_experiment.base_prompt_diagnosis import (
    ai_hospital_base_diagnosis,
    mdbench_base_diagnosis,
)


def _client_from_block(block: Dict[str, Any]) -> ModelClient:
    return ModelClient(
        block["provider"],
        block["name"],
        temperature=float(block.get("temperature", 0.6)),
        base_url=block.get("base_url"),
    )


def _client(cfg: Dict[str, Any], key: str) -> ModelClient:
    block = (cfg.get("models") or {}).get(key) or {}
    return _client_from_block(block)


def _agent_clients(cfg: Dict[str, Any]) -> "Dict[str, ModelClient]":
    specs = cfg.get("agents") or []
    return {spec["id"]: _client_from_block(spec) for spec in specs}


def _reference_context(case: Dict[str, Any]) -> Dict[str, Any]:
    ref = dict(case.get("reference_context") or {})
    ref["ground_truth_diagnosis"] = case.get("ground_truth_diagnosis")
    return ref


def _load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[str(row["case_id"])] = row
    return rows


def run_ai_hospital(
    cfg: Dict[str, Any], *, max_cases: Optional[int] = None, fresh: bool = False
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")

    data_cfg = (cfg.get("data") or {}).get("ai_hospital") or {}
    dlg_cfg = cfg.get("dialogue") or {}
    base_output_dir = ROOT / str((cfg.get("pipeline") or {}).get("output_dir", "clinic_experiment/outputs"))
    output_dir = ensure_dir(base_output_dir / "ai_hospital")

    source_path = ROOT / str(data_cfg.get("source_path", "data/ai_hospital_patients.json"))
    num_cases = int(max_cases if max_cases is not None else data_cfg.get("num_cases", 100))
    seed = int(data_cfg.get("seed", 42))
    cases = load_ai_hospital_cases(source_path, num_cases, seed)
    print(f"Loaded {len(cases)} AI Hospital cases (requested {num_cases}, seed {seed})")

    agents = _agent_clients(cfg)
    agent_ids = list(agents.keys())
    patient = _client(cfg, "patient")
    reporter = _client(cfg, "reporter")
    judge = _client(cfg, "judge")

    max_turn = int(dlg_cfg.get("max_conversation_turn", 10))
    temperature = float(dlg_cfg.get("temperature", 0.6))

    # 1. Each agent's own full dialogue + own diagnosis.
    dialogues: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = ensure_dir(output_dir / f"agent_{aid}") / "dialogues.jsonl"
        if fresh:
            path.write_text("", encoding="utf-8")
            existing: Dict[str, Dict[str, Any]] = {}
        else:
            existing = _load_existing(path)
            if existing:
                print(f"Resuming agent_{aid}: {len(existing)} cases already done")

        results: List[Dict[str, Any]] = []
        for i, case in enumerate(cases, start=1):
            cid = str(case["case_id"])
            if cid in existing:
                results.append(existing[cid])
                continue
            print(f"[agent_{aid} {i}/{len(cases)}] case_id={cid}")
            ref = _reference_context(case)
            outcome = run_consultation(
                case,
                doctor_client=agents[aid],
                patient_client=patient,
                reporter_client=reporter,
                max_conversation_turn=max_turn,
                temperature=temperature,
            )
            judged = judge_diagnosis(
                judge, reference_context=ref, doctor_diagnosis_text=outcome["final_diagnosis_text"]
            )
            row = {
                "case_id": case["case_id"],
                "dialog_history": outcome["dialog_history"],
                "final_diagnosis_text": outcome["final_diagnosis_text"],
                "diagnosis_choice": judged.get("diagnosis_choice"),
                "correct": judged["correct"],
            }
            append_jsonl(path, row)
            results.append(row)
        dialogues[aid] = results

    # 2. Cross-diagnosis: every agent's dialogue transcript -> every OTHER agent diagnoses it.
    cross_dir = ensure_dir(output_dir / "cross")
    cross_results: Dict[Any, List[Dict[str, Any]]] = {}
    for src in agent_ids:
        records = [
            {
                "case_id": r["case_id"],
                "reference_context": _reference_context(c),
                "dialog_history": r["dialog_history"],
            }
            for c, r in zip(cases, dialogues[src])
        ]
        for dst in agent_ids:
            if dst == src:
                continue
            print(f"=== Cross finalize: {src}'s dialogue -> {dst} diagnoses ===")
            cross_results[(src, dst)] = run_cross_finalize(
                records,
                responder_client=agents[dst],
                judge_client=judge,
                path=cross_dir / f"dialogue_{src}_diagnosis_{dst}.jsonl",
                fresh=fresh,
            )

    # 3. Base prompt only: zero case info, one diagnosis per agent, judged per case.
    base_dir = ensure_dir(output_dir / "base_prompt_only")
    base_results: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = base_dir / f"agent_{aid}.jsonl"
        if fresh:
            path.write_text("", encoding="utf-8")
            existing = {}
        else:
            existing = _load_existing(path)
        if existing:
            diag_text = next(iter(existing.values()))["diagnosis_text"]
        else:
            diag_text = ai_hospital_base_diagnosis(agents[aid])
        print(f"=== Base prompt only ({aid}): {diag_text[:80]!r}... ===")
        results = []
        for case in cases:
            cid = str(case["case_id"])
            if cid in existing:
                results.append(existing[cid])
                continue
            judged = judge_diagnosis(
                judge, reference_context=_reference_context(case), doctor_diagnosis_text=diag_text
            )
            row = {
                "case_id": case["case_id"],
                "diagnosis_text": diag_text,
                "diagnosis_choice": judged.get("diagnosis_choice"),
                "correct": judged["correct"],
            }
            append_jsonl(path, row)
            results.append(row)
        base_results[aid] = results

    # 4. Assemble the (1 + n_agents) x n_agents matrix.
    matrix: Dict[str, Dict[str, Any]] = {
        "base_prompt_only": {aid: accuracy(base_results[aid]) for aid in agent_ids}
    }
    for src in agent_ids:
        row: Dict[str, Any] = {}
        for dst in agent_ids:
            row[dst] = accuracy(dialogues[src]) if dst == src else accuracy(cross_results[(src, dst)])
        matrix[f"dialogue_{src}"] = row

    agents_cfg = cfg.get("agents") or []
    summary = {
        "dataset": "ai_hospital",
        "n_cases": len(cases),
        "agents": {spec["id"]: f"{spec['provider']}/{spec['name']}" for spec in agents_cfg},
        "matrix": matrix,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Done. Summary at {summary_path} ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_mdbench(
    cfg: Dict[str, Any], *, max_cases: Optional[int] = None, fresh: bool = False
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")

    data_cfg = (cfg.get("data") or {}).get("mdbench") or {}
    dlg_cfg = cfg.get("dialogue") or {}
    base_output_dir = ROOT / str((cfg.get("pipeline") or {}).get("output_dir", "clinic_experiment/outputs"))
    output_dir = ensure_dir(base_output_dir / "mdbench")

    hf_dataset = str(data_cfg.get("hf_dataset", "univanxx/3mdbench"))
    hf_split = str(data_cfg.get("hf_split", "test"))
    num_cases = int(max_cases if max_cases is not None else data_cfg.get("num_cases", 100))
    seed = int(data_cfg.get("seed", 42))
    cases = load_mdbench_cases(hf_dataset, hf_split, num_cases, seed)
    print(f"Loaded {len(cases)} 3MDBench cases (requested {num_cases}, seed {seed})")

    agents = _agent_clients(cfg)
    agent_ids = list(agents.keys())
    patient = _client(cfg, "patient")
    judge = _client(cfg, "judge")  # also used as the diagnosis-extraction model

    temperature = float(dlg_cfg.get("temperature", 0.6))

    # 1. Each agent's own full dialogue + own diagnosis.
    dialogues: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = ensure_dir(output_dir / f"agent_{aid}") / "dialogues.jsonl"
        if fresh:
            path.write_text("", encoding="utf-8")
            existing: Dict[str, Dict[str, Any]] = {}
        else:
            existing = _load_existing(path)
            if existing:
                print(f"Resuming agent_{aid}: {len(existing)} cases already done")

        results: List[Dict[str, Any]] = []
        for i, case in enumerate(cases, start=1):
            cid = str(case["case_id"])
            if cid in existing:
                results.append(existing[cid])
                continue
            print(f"[agent_{aid} {i}/{len(cases)}] case_id={cid}")
            gt = case["ground_truth_diagnosis"]
            outcome = run_mdbench_consultation(
                case, doctor_client=agents[aid], patient_client=patient, temperature=temperature
            )
            predicted = extract_diagnosis(judge, outcome["dialog_history"])
            judged = judge_mdbench_diagnosis(judge, ground_truth=gt, predicted_diagnosis_text=predicted)
            row = {
                "case_id": case["case_id"],
                "dialog_history": outcome["dialog_history"],
                "dialogue_ended": outcome["dialogue_ended"],
                "predicted_diagnosis": predicted,
                "correct": judged["correct"],
            }
            append_jsonl(path, row)
            results.append(row)
        dialogues[aid] = results

    # 2. Cross-diagnosis: every agent's dialogue transcript -> every OTHER agent diagnoses it.
    cross_dir = ensure_dir(output_dir / "cross")
    cross_results: Dict[Any, List[Dict[str, Any]]] = {}
    for src in agent_ids:
        records = [
            {
                "case_id": r["case_id"],
                "ground_truth_diagnosis": c["ground_truth_diagnosis"],
                "dialog_history": r["dialog_history"],
            }
            for c, r in zip(cases, dialogues[src])
        ]
        for dst in agent_ids:
            if dst == src:
                continue
            print(f"=== Cross finalize: {src}'s dialogue -> {dst} diagnoses ===")
            cross_results[(src, dst)] = run_mdbench_cross_finalize(
                records,
                responder_client=agents[dst],
                judge_client=judge,
                path=cross_dir / f"dialogue_{src}_diagnosis_{dst}.jsonl",
                fresh=fresh,
            )

    # 3. Base prompt only: zero case info, one diagnosis per agent, judged per case.
    base_dir = ensure_dir(output_dir / "base_prompt_only")
    base_results: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = base_dir / f"agent_{aid}.jsonl"
        if fresh:
            path.write_text("", encoding="utf-8")
            existing = {}
        else:
            existing = _load_existing(path)
        if existing:
            diag_text = next(iter(existing.values()))["diagnosis_text"]
        else:
            diag_text = mdbench_base_diagnosis(agents[aid])
        print(f"=== Base prompt only ({aid}): {diag_text[:80]!r}... ===")
        results = []
        for case in cases:
            cid = str(case["case_id"])
            if cid in existing:
                results.append(existing[cid])
                continue
            judged = judge_mdbench_diagnosis(
                judge, ground_truth=case["ground_truth_diagnosis"], predicted_diagnosis_text=diag_text
            )
            row = {
                "case_id": case["case_id"],
                "diagnosis_text": diag_text,
                "correct": judged["correct"],
            }
            append_jsonl(path, row)
            results.append(row)
        base_results[aid] = results

    # 4. Assemble the (1 + n_agents) x n_agents matrix.
    matrix: Dict[str, Dict[str, Any]] = {
        "base_prompt_only": {aid: mdbench_accuracy(base_results[aid]) for aid in agent_ids}
    }
    for src in agent_ids:
        row: Dict[str, Any] = {}
        for dst in agent_ids:
            row[dst] = (
                mdbench_accuracy(dialogues[src]) if dst == src else mdbench_accuracy(cross_results[(src, dst)])
            )
        matrix[f"dialogue_{src}"] = row

    agents_cfg = cfg.get("agents") or []
    summary = {
        "dataset": "mdbench",
        "n_cases": len(cases),
        "agents": {spec["id"]: f"{spec['provider']}/{spec['name']}" for spec in agents_cfg},
        "matrix": matrix,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Done. Summary at {summary_path} ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Clinic N-agent experiment runner.")
    parser.add_argument("--dataset", choices=["ai_hospital", "mdbench"], default="ai_hospital")
    parser.add_argument("--config", type=Path, default=ROOT / "clinic_experiment" / "config.yaml")
    parser.add_argument(
        "--max-cases", type=int, default=None, help="Override case count (e.g. 5 for a smoke test)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing dialogues.jsonl and start over (default: resume, skip cases already done)",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    cfg = load_yaml(args.config)

    if args.dataset == "ai_hospital":
        run_ai_hospital(cfg, max_cases=args.max_cases, fresh=args.fresh)
    else:
        run_mdbench(cfg, max_cases=args.max_cases, fresh=args.fresh)


if __name__ == "__main__":
    main()
