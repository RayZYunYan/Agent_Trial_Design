"""N-agent AI Hospital / 3MDBench experiment.

Each agent in config.yaml's `agents` list runs a full consultation against
the SAME shared Patient (and Reporter) model/config, so the only variable
between agents is the diagnosing/dialogue model itself -- patient behavior is
held constant within a run. Rows produced per dataset:

  base_prompt_only   -- per-case diagnosis from the opening vignette only
  dialogue_<agent>   -- each agent's own consultation (+ cross-finalize cells
                        when pipeline.cross_finalize is on: every transcript
                        re-diagnosed by every OTHER agent)
  all_facts          -- per-case diagnosis from the full structured record
                        (oracle upper anchor)

Simulator-condition experiments (D2): every run is namespaced by a run_id
(pipeline.run_id or --run-id) into outputs/<dataset>/<run_id>/, and the
resolved condition config is frozen to run_config.json in that directory.
Resuming into a run_id whose stored config differs from the current one
aborts, so conditions can never silently mix — vary patient backbone,
`models.patient.persona` (dataset-native / "forthcoming" / "guarded" / free
text), or `models.patient.leak_guard` under a NEW run_id instead.

Usage (from repo root):
  python -m clinic_experiment.run_pipeline --dataset ai_hospital --max-cases 5   # smoke test
  python -m clinic_experiment.run_pipeline --dataset ai_hospital --run-id C0     # full run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mediq_experiment.io_utils import ROOT, ensure_dir, load_yaml
from smart_trial.models.model_client import ModelClient

from clinic_experiment.concurrency import load_existing_rows, run_cases
from clinic_experiment.loaders.ai_hospital_loader import load_cases as load_ai_hospital_cases
from clinic_experiment.ai_hospital_dialogue import run_consultation
from clinic_experiment.diagnosis_judge import judge_diagnosis
from clinic_experiment.ai_hospital_cross_finalize import (
    run_cross_finalize,
    accuracy,
    find_first_diagnosis_turn as find_first_diagnosis_turn_zh,
)

from clinic_experiment.loaders.mdbench_loader import load_cases as load_mdbench_cases
from clinic_experiment.mdbench_dialogue import (
    run_consultation as run_mdbench_consultation,
    extract_diagnosis,
)
from clinic_experiment.mdbench_judge import judge_diagnosis as judge_mdbench_diagnosis
from clinic_experiment.mdbench_cross_finalize import (
    run_cross_finalize as run_mdbench_cross_finalize,
    accuracy as mdbench_accuracy,
    find_first_diagnosis_turn,
)

from clinic_experiment.base_prompt_diagnosis import (
    ai_hospital_base_diagnosis,
    mdbench_base_diagnosis,
)
from clinic_experiment.oracle_diagnosis import (
    ai_hospital_oracle_diagnosis,
    mdbench_oracle_diagnosis,
)


def _client_from_block(block: Dict[str, Any]) -> ModelClient:
    return ModelClient(
        block["provider"],
        block["name"],
        temperature=float(block.get("temperature", 0.6)),
        base_url=block.get("base_url"),
        api_key_env=block.get("api_key_env"),
    )


def _client(cfg: Dict[str, Any], key: str) -> ModelClient:
    block = (cfg.get("models") or {}).get(key) or {}
    return _client_from_block(block)


def _agent_clients(cfg: Dict[str, Any]) -> "Dict[str, ModelClient]":
    specs = cfg.get("agents") or []
    return {spec["id"]: _client_from_block(spec) for spec in specs}


def _pipeline_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("pipeline") or {}


def _patient_condition(cfg: Dict[str, Any]) -> Dict[str, Any]:
    patient_cfg = (cfg.get("models") or {}).get("patient") or {}
    return {
        "persona": patient_cfg.get("persona") or None,
        "leak_guard": bool(patient_cfg.get("leak_guard", False)),
    }


def _condition_snapshot(cfg: Dict[str, Any], dataset: str, seed: int) -> Dict[str, Any]:
    """Everything that defines a run condition. Changing any of it under the
    same run_id would silently mix conditions in the append-only JSONLs."""
    data_cfg = (cfg.get("data") or {}).get(dataset) or {}
    return json.loads(
        json.dumps(
            {
                "dataset": dataset,
                "seed": seed,
                "agents": cfg.get("agents") or [],
                "models": cfg.get("models") or {},
                "dialogue": cfg.get("dialogue") or {},
                "data": data_cfg,
                "cross_finalize": bool(_pipeline_cfg(cfg).get("cross_finalize", True)),
            },
            sort_keys=True,
        )
    )


def _init_run_dir(
    base_output_dir: Path,
    dataset: str,
    run_id: str,
    snapshot: Dict[str, Any],
    allow_mismatch: bool,
) -> Path:
    run_dir = ensure_dir(base_output_dir / dataset / run_id)
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        stored_snapshot = {k: stored.get(k) for k in snapshot}
        if stored_snapshot != snapshot:
            diff_keys = [k for k in snapshot if stored_snapshot.get(k) != snapshot[k]]
            message = (
                f"run_config mismatch for run_id={run_id!r} (differing: {diff_keys}).\n"
                f"Existing: {config_path}\n"
                "Resuming would mix conditions in the same output files. Use a new "
                "--run-id for a changed condition, or --allow-config-mismatch to "
                "override (only if you are certain the difference is harmless)."
            )
            if not allow_mismatch:
                raise SystemExit(message)
            print(f"WARNING: {message}\nProceeding due to --allow-config-mismatch.")
    else:
        config_path.write_text(
            json.dumps(
                {**snapshot, "created_at": datetime.now(timezone.utc).isoformat()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return run_dir


def _reference_context(case: Dict[str, Any]) -> Dict[str, Any]:
    ref = dict(case.get("reference_context") or {})
    ref["ground_truth_diagnosis"] = case.get("ground_truth_diagnosis")
    return ref


_load_existing = load_existing_rows


def _backfill_first_diagnosis(
    rows: List[Dict[str, Any]], agent_dir: Path, finder
) -> None:
    """Fill first_diagnosis_index on dialogue rows that predate cross being
    enabled. A sidecar JSON caches computed indices so the finder (an LLM
    call) runs at most once per case across ALL process runs, and the
    truncation point can never drift between resumes."""
    missing = [r for r in rows if "first_diagnosis_index" not in r]
    if not missing:
        return
    sidecar = agent_dir / "first_diagnosis_index.json"
    cache: Dict[str, Any] = (
        json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    )
    dirty = False
    for r in missing:
        cid = str(r["case_id"])
        if cid in cache:
            r["first_diagnosis_index"] = cache[cid]
        else:
            r["first_diagnosis_index"] = finder(r)
            cache[cid] = r["first_diagnosis_index"]
            dirty = True
    if dirty:
        sidecar.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _usage_report(clients: Dict[str, ModelClient]) -> Dict[str, Any]:
    return {
        name: {"model": f"{c.provider}/{c.model_name}", **c.usage_totals}
        for name, c in clients.items()
    }


def run_ai_hospital(
    cfg: Dict[str, Any],
    *,
    max_cases: Optional[int] = None,
    fresh: bool = False,
    run_id: Optional[str] = None,
    allow_config_mismatch: bool = False,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env", override=True)
    # A stale ANTHROPIC_BASE_URL exported in the parent shell would silently
    # redirect direct-Anthropic calls (patient/reporter/judge/agent B) to a
    # relay. Agents that want a relay pass base_url explicitly via config.
    import os as _os
    _os.environ.pop("ANTHROPIC_BASE_URL", None)

    data_cfg = (cfg.get("data") or {}).get("ai_hospital") or {}
    dlg_cfg = cfg.get("dialogue") or {}
    pipe_cfg = _pipeline_cfg(cfg)
    base_output_dir = ROOT / str(pipe_cfg.get("output_dir", "clinic_experiment/outputs"))
    run_id = run_id or str(pipe_cfg.get("run_id", "C0"))
    concurrency = int(pipe_cfg.get("concurrency", 1))
    do_cross = bool(pipe_cfg.get("cross_finalize", True))

    seed = int(data_cfg.get("seed", 42))
    snapshot = _condition_snapshot(cfg, "ai_hospital", seed)
    output_dir = _init_run_dir(base_output_dir, "ai_hospital", run_id, snapshot, allow_config_mismatch)

    source_path = ROOT / str(data_cfg.get("source_path", "data/ai_hospital_patients.json"))
    num_cases = int(max_cases if max_cases is not None else data_cfg.get("num_cases", 100))
    cases = load_ai_hospital_cases(source_path, num_cases, seed)
    print(f"Loaded {len(cases)} AI Hospital cases (requested {num_cases}, seed {seed}) -> run_id={run_id}")

    agents = _agent_clients(cfg)
    agent_ids = list(agents.keys())
    patient = _client(cfg, "patient")
    reporter = _client(cfg, "reporter")
    judge = _client(cfg, "judge")
    condition = _patient_condition(cfg)

    max_turn = int(dlg_cfg.get("max_conversation_turn", 10))
    temperature = float(dlg_cfg.get("temperature", 0.6))

    # 1. Each agent's own full dialogue + own diagnosis.
    dialogues: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = ensure_dir(output_dir / f"agent_{aid}") / "dialogues.jsonl"
        existing = _load_existing(path, fresh)
        if existing:
            print(f"Resuming agent_{aid}: {len(existing)} cases already done")

        def worker(case: Dict[str, Any], aid: str = aid) -> Dict[str, Any]:
            outcome = run_consultation(
                case,
                doctor_client=agents[aid],
                patient_client=patient,
                reporter_client=reporter,
                max_conversation_turn=max_turn,
                temperature=temperature,
                persona_override=condition["persona"],
                leak_guard=condition["leak_guard"],
            )
            judged = judge_diagnosis(
                judge,
                reference_context=_reference_context(case),
                doctor_diagnosis_text=outcome["final_diagnosis_text"],
            )
            row = {
                "case_id": case["case_id"],
                "dialog_history": outcome["dialog_history"],
                "final_diagnosis_text": outcome["final_diagnosis_text"],
                "n_turns": outcome["n_turns"],
                "ended_by_patient": outcome["ended_by_patient"],
                "usage": outcome["usage"],
                "diagnosis_choice": judged.get("diagnosis_choice"),
                "correct": judged["correct"],
                "judge_parse_failed": judged.get("judge_parse_failed", False),
            }
            if do_cross:
                # Found once here, reused by every destination's cross cell.
                row["first_diagnosis_index"] = find_first_diagnosis_turn_zh(
                    judge, outcome["dialog_history"]
                )
            return row

        dialogues[aid] = run_cases(
            cases, existing, worker, path=path, max_workers=concurrency, desc=f"agent_{aid}"
        )

    # 2. Cross-diagnosis: every agent's dialogue transcript -> every OTHER agent.
    cross_results: Dict[Any, List[Dict[str, Any]]] = {}
    if do_cross:
        cross_dir = ensure_dir(output_dir / "cross")
        case_by_id = {str(c["case_id"]): c for c in cases}
        for src in agent_ids:
            _backfill_first_diagnosis(
                dialogues[src],
                output_dir / f"agent_{src}",
                lambda r: find_first_diagnosis_turn_zh(judge, r["dialog_history"]),
            )
            records = [
                {
                    "case_id": r["case_id"],
                    "reference_context": _reference_context(case_by_id[str(r["case_id"])]),
                    "dialog_history": r["dialog_history"],
                    "first_diagnosis_index": r["first_diagnosis_index"],
                    "final_diagnosis_text": r.get("final_diagnosis_text"),
                }
                for r in dialogues[src]
                if str(r["case_id"]) in case_by_id
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
                    max_workers=concurrency,
                )

    # 3. Base-prompt anchor: per-case diagnosis from the opening vignette only.
    base_dir = ensure_dir(output_dir / "base_prompt_only")
    base_results: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = base_dir / f"agent_{aid}.jsonl"
        existing = _load_existing(path, fresh)

        def base_worker(case: Dict[str, Any], aid: str = aid) -> Dict[str, Any]:
            diag_text = ai_hospital_base_diagnosis(agents[aid], case)
            judged = judge_diagnosis(
                judge, reference_context=_reference_context(case), doctor_diagnosis_text=diag_text
            )
            return {
                "case_id": case["case_id"],
                "diagnosis_text": diag_text,
                "diagnosis_choice": judged.get("diagnosis_choice"),
                "correct": judged["correct"],
                "judge_parse_failed": judged.get("judge_parse_failed", False),
            }

        base_results[aid] = run_cases(
            cases, existing, base_worker, path=path, max_workers=concurrency, desc=f"base_{aid}"
        )

    # 4. All-facts oracle anchor: per-case diagnosis from the full record.
    oracle_dir = ensure_dir(output_dir / "all_facts")
    oracle_results: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = oracle_dir / f"agent_{aid}.jsonl"
        existing = _load_existing(path, fresh)

        def oracle_worker(case: Dict[str, Any], aid: str = aid) -> Dict[str, Any]:
            diag_text = ai_hospital_oracle_diagnosis(agents[aid], case)
            judged = judge_diagnosis(
                judge, reference_context=_reference_context(case), doctor_diagnosis_text=diag_text
            )
            return {
                "case_id": case["case_id"],
                "diagnosis_text": diag_text,
                "diagnosis_choice": judged.get("diagnosis_choice"),
                "correct": judged["correct"],
                "judge_parse_failed": judged.get("judge_parse_failed", False),
            }

        oracle_results[aid] = run_cases(
            cases, existing, oracle_worker, path=path, max_workers=concurrency, desc=f"oracle_{aid}"
        )

    # 5. Assemble the accuracy matrix.
    matrix: Dict[str, Dict[str, Any]] = {
        "base_prompt_only": {aid: accuracy(base_results[aid]) for aid in agent_ids}
    }
    for src in agent_ids:
        row: Dict[str, Any] = {}
        for dst in agent_ids:
            if dst == src:
                row[dst] = accuracy(dialogues[src])
            elif do_cross:
                row[dst] = accuracy(cross_results[(src, dst)])
        matrix[f"dialogue_{src}"] = row
    matrix["all_facts"] = {aid: accuracy(oracle_results[aid]) for aid in agent_ids}

    agents_cfg = cfg.get("agents") or []
    summary = {
        "dataset": "ai_hospital",
        "run_id": run_id,
        "n_cases": len(cases),
        "condition": condition,
        "agents": {spec["id"]: f"{spec['provider']}/{spec['name']}" for spec in agents_cfg},
        "dialogue_stats": {
            aid: {
                "mean_turns": (
                    sum(r.get("n_turns", 0) for r in dialogues[aid]) / len(dialogues[aid])
                    if dialogues[aid]
                    else None
                ),
                "turn_cap_hits": sum(1 for r in dialogues[aid] if not r.get("ended_by_patient")),
            }
            for aid in agent_ids
        },
        "matrix": matrix,
        "usage": _usage_report(
            {"patient": patient, "reporter": reporter, "judge": judge, **{f"agent_{a}": c for a, c in agents.items()}}
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Done. Summary at {summary_path} ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_mdbench(
    cfg: Dict[str, Any],
    *,
    max_cases: Optional[int] = None,
    fresh: bool = False,
    run_id: Optional[str] = None,
    allow_config_mismatch: bool = False,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env", override=True)
    # A stale ANTHROPIC_BASE_URL exported in the parent shell would silently
    # redirect direct-Anthropic calls (patient/reporter/judge/agent B) to a
    # relay. Agents that want a relay pass base_url explicitly via config.
    import os as _os
    _os.environ.pop("ANTHROPIC_BASE_URL", None)

    data_cfg = (cfg.get("data") or {}).get("mdbench") or {}
    dlg_cfg = cfg.get("dialogue") or {}
    pipe_cfg = _pipeline_cfg(cfg)
    base_output_dir = ROOT / str(pipe_cfg.get("output_dir", "clinic_experiment/outputs"))
    run_id = run_id or str(pipe_cfg.get("run_id", "C0"))
    concurrency = int(pipe_cfg.get("concurrency", 1))
    do_cross = bool(pipe_cfg.get("cross_finalize", True))

    seed = int(data_cfg.get("seed", 42))
    snapshot = _condition_snapshot(cfg, "mdbench", seed)
    output_dir = _init_run_dir(base_output_dir, "mdbench", run_id, snapshot, allow_config_mismatch)

    hf_dataset = str(data_cfg.get("hf_dataset", "univanxx/3mdbench"))
    hf_split = str(data_cfg.get("hf_split", "test"))
    num_cases = int(max_cases if max_cases is not None else data_cfg.get("num_cases", 100))
    cases = load_mdbench_cases(hf_dataset, hf_split, num_cases, seed)
    print(f"Loaded {len(cases)} 3MDBench cases (requested {num_cases}, seed {seed}) -> run_id={run_id}")

    agents = _agent_clients(cfg)
    agent_ids = list(agents.keys())
    patient = _client(cfg, "patient")
    judge = _client(cfg, "judge")  # also used as the diagnosis-extraction model
    condition = _patient_condition(cfg)

    temperature = float(dlg_cfg.get("temperature", 0.6))

    # 1. Each agent's own full dialogue + own diagnosis.
    dialogues: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = ensure_dir(output_dir / f"agent_{aid}") / "dialogues.jsonl"
        existing = _load_existing(path, fresh)
        if existing:
            print(f"Resuming agent_{aid}: {len(existing)} cases already done")

        def worker(case: Dict[str, Any], aid: str = aid) -> Dict[str, Any]:
            gt = case["ground_truth_diagnosis"]
            outcome = run_mdbench_consultation(
                case,
                doctor_client=agents[aid],
                patient_client=patient,
                temperature=temperature,
                persona_override=condition["persona"],
                leak_guard=condition["leak_guard"],
            )
            predicted = extract_diagnosis(judge, outcome["dialog_history"])
            judged = judge_mdbench_diagnosis(judge, ground_truth=gt, predicted_diagnosis_text=predicted)
            row = {
                "case_id": case["case_id"],
                "dialog_history": outcome["dialog_history"],
                "dialogue_ended": outcome["dialogue_ended"],
                "n_turns": outcome["n_turns"],
                "usage": outcome["usage"],
                "predicted_diagnosis": predicted,
                "correct": judged["correct"],
                "judge_parse_failed": judged.get("judge_parse_failed", False),
            }
            if do_cross:
                # Found once here, reused by every destination model's cross cell.
                idx = find_first_diagnosis_turn(judge, outcome["dialog_history"])
                row["first_diagnosis_index"] = idx
                # Did the source doctor revise after first stating a diagnosis?
                # Cross cells see pre-first-diagnosis info while the diagonal
                # scores the FINAL answer — this flag marks where they diverge.
                if idx is not None and predicted and predicted.lower() != "none":
                    first_label = extract_diagnosis(
                        judge, outcome["dialog_history"][: idx + 1]
                    )
                    row["diagnosis_revised"] = bool(
                        first_label
                        and first_label.lower() != "none"
                        and first_label.strip().lower() != predicted.strip().lower()
                    )
            return row

        dialogues[aid] = run_cases(
            cases, existing, worker, path=path, max_workers=concurrency, desc=f"agent_{aid}"
        )

    # 2. Cross-diagnosis: every agent's dialogue transcript -> every OTHER agent.
    cross_results: Dict[Any, List[Dict[str, Any]]] = {}
    if do_cross:
        cross_dir = ensure_dir(output_dir / "cross")
        case_by_id = {str(c["case_id"]): c for c in cases}
        for src in agent_ids:
            _backfill_first_diagnosis(
                dialogues[src],
                output_dir / f"agent_{src}",
                lambda r: find_first_diagnosis_turn(judge, r["dialog_history"]),
            )
            records = [
                {
                    "case_id": r["case_id"],
                    "ground_truth_diagnosis": case_by_id[str(r["case_id"])]["ground_truth_diagnosis"],
                    "dialog_history": r["dialog_history"],
                    "first_diagnosis_index": r["first_diagnosis_index"],
                }
                for r in dialogues[src]
                if str(r["case_id"]) in case_by_id
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
                    max_workers=concurrency,
                )

    # 3. Base-prompt anchor: per-case diagnosis from the opening complaint only.
    base_dir = ensure_dir(output_dir / "base_prompt_only")
    base_results: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = base_dir / f"agent_{aid}.jsonl"
        existing = _load_existing(path, fresh)

        def base_worker(case: Dict[str, Any], aid: str = aid) -> Dict[str, Any]:
            diag_text = mdbench_base_diagnosis(agents[aid], case)
            predicted = extract_diagnosis(judge, [{"role": "Doctor", "content": diag_text}])
            judged = judge_mdbench_diagnosis(
                judge, ground_truth=case["ground_truth_diagnosis"], predicted_diagnosis_text=predicted
            )
            return {
                "case_id": case["case_id"],
                "diagnosis_text": diag_text,
                "predicted_diagnosis": predicted,
                "correct": judged["correct"],
                "judge_parse_failed": judged.get("judge_parse_failed", False),
            }

        base_results[aid] = run_cases(
            cases, existing, base_worker, path=path, max_workers=concurrency, desc=f"base_{aid}"
        )

    # 4. All-facts oracle anchor: per-case diagnosis from the full complaint set.
    oracle_dir = ensure_dir(output_dir / "all_facts")
    oracle_results: Dict[str, List[Dict[str, Any]]] = {}
    for aid in agent_ids:
        path = oracle_dir / f"agent_{aid}.jsonl"
        existing = _load_existing(path, fresh)

        def oracle_worker(case: Dict[str, Any], aid: str = aid) -> Dict[str, Any]:
            diag_text = mdbench_oracle_diagnosis(agents[aid], case)
            predicted = extract_diagnosis(judge, [{"role": "Doctor", "content": diag_text}])
            judged = judge_mdbench_diagnosis(
                judge, ground_truth=case["ground_truth_diagnosis"], predicted_diagnosis_text=predicted
            )
            return {
                "case_id": case["case_id"],
                "diagnosis_text": diag_text,
                "predicted_diagnosis": predicted,
                "correct": judged["correct"],
                "judge_parse_failed": judged.get("judge_parse_failed", False),
            }

        oracle_results[aid] = run_cases(
            cases, existing, oracle_worker, path=path, max_workers=concurrency, desc=f"oracle_{aid}"
        )

    # 5. Assemble the accuracy matrix.
    matrix: Dict[str, Dict[str, Any]] = {
        "base_prompt_only": {aid: mdbench_accuracy(base_results[aid]) for aid in agent_ids}
    }
    for src in agent_ids:
        row: Dict[str, Any] = {}
        for dst in agent_ids:
            if dst == src:
                row[dst] = mdbench_accuracy(dialogues[src])
            elif do_cross:
                row[dst] = mdbench_accuracy(cross_results[(src, dst)])
        matrix[f"dialogue_{src}"] = row
    matrix["all_facts"] = {aid: mdbench_accuracy(oracle_results[aid]) for aid in agent_ids}

    agents_cfg = cfg.get("agents") or []
    summary = {
        "dataset": "mdbench",
        "run_id": run_id,
        "n_cases": len(cases),
        "condition": condition,
        "agents": {spec["id"]: f"{spec['provider']}/{spec['name']}" for spec in agents_cfg},
        "dialogue_stats": {
            aid: {
                "mean_turns": (
                    sum(r.get("n_turns", 0) for r in dialogues[aid]) / len(dialogues[aid])
                    if dialogues[aid]
                    else None
                ),
                "turn_cap_hits": sum(1 for r in dialogues[aid] if not r.get("dialogue_ended")),
            }
            for aid in agent_ids
        },
        "matrix": matrix,
        "usage": _usage_report(
            {"patient": patient, "judge": judge, **{f"agent_{a}": c for a, c in agents.items()}}
        ),
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
        "--run-id",
        default=None,
        help="Condition/run namespace (default: pipeline.run_id from config). "
        "Outputs go to outputs/<dataset>/<run_id>/",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing rows in this run_id and start over (default: resume)",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Resume into a run_id even though the stored run_config.json differs (dangerous)",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma-separated agent-id filter (e.g. 'A,B,C,D'). Skips every agent "
        "not listed. Lets you sequentialize memory-heavy local models on the "
        "same run_id: run one MLX-hosted agent at a time so both weight sets "
        "never coexist. Implies --allow-config-mismatch since the frozen "
        "run_config.json will differ across phases.",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    cfg = load_yaml(args.config)

    if args.agents:
        keep = {a.strip() for a in args.agents.split(",") if a.strip()}
        cfg["agents"] = [a for a in (cfg.get("agents") or []) if a.get("id") in keep]
        missing = keep - {a.get("id") for a in cfg["agents"]}
        if missing:
            print(f"WARN: --agents referenced unknown ids {sorted(missing)}", file=sys.stderr)
        args.allow_config_mismatch = True

    kwargs = dict(
        max_cases=args.max_cases,
        fresh=args.fresh,
        run_id=args.run_id,
        allow_config_mismatch=args.allow_config_mismatch,
    )
    if args.dataset == "ai_hospital":
        run_ai_hospital(cfg, **kwargs)
    else:
        run_mdbench(cfg, **kwargs)


if __name__ == "__main__":
    main()
