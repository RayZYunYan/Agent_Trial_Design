"""
One-shot fact ablation pipeline (no interactive dialogue).

Modes: base | one_random | two_most_important_claude | two_most_important_gpt | all_facts
Each mode × doctor_a…e. Interrupt-safe: re-run the same command to resume
(MediQ skips case ids already in results.jsonl; incomplete doctor runs are re-entered).

Usage (repo root):
  python -m mediq_experiment.run_ablation --mode base
  python -m mediq_experiment.run_ablation --all-modes
  python -m mediq_experiment.run_ablation --mode one_random --dry-run
  python -m mediq_experiment.run_ablation --mode base --config mediq_experiment/config_ablation_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from mediq_experiment.ablation import (
    ABLATION_MODES,
    expected_case_ids,
    prepare_ablation_jsonl,
    results_complete,
)
from mediq_experiment.io_utils import ROOT, ensure_dir, load_yaml, read_jsonl
from mediq_experiment.model_chat import is_local_provider, resolve_provider
from mediq_experiment.run_mediq import run_mediq_role
from mediq_experiment.run_pipeline import _doctor_letter, _list_doctors, _model_name


def _accuracy_for_results(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    n = len(rows)
    correct = 0
    for row in rows:
        interactive = row.get("interactive_system") or {}
        if interactive.get("correct"):
            correct += 1
    return {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
    }


def run_ablation_mode(
    cfg: Dict[str, Any],
    mode: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown mode {mode}. Choose from {ABLATION_MODES}")

    data_cfg = dict(cfg.get("data") or {})
    mediq_cfg = dict(cfg.get("mediq") or {})
    pipe_cfg = cfg.get("pipeline") or {}
    ablation_cfg = dict(cfg.get("ablation") or {})
    models = cfg.get("models") or {}

    # Force one-shot (no dialogue), regardless of interactive defaults.
    mediq_cfg["max_questions"] = 0

    output_root = ensure_dir(
        ROOT / str(pipe_cfg.get("output_dir", "mediq_experiment/outputs_ablation"))
    )
    mode_dir = ensure_dir(output_root / mode)
    prepared_path = mode_dir / "cases_ablation.jsonl"

    prepare_ablation_jsonl(
        mode=mode,
        dest=prepared_path,
        ablation_cfg=ablation_cfg,
        data_cfg=data_cfg,
    )
    expected_ids = expected_case_ids(prepared_path)
    print(
        f"\n=== Ablation mode={mode} | prepared {len(expected_ids)} cases -> {prepared_path} ==="
    )

    use_vllm = bool(mediq_cfg.get("use_vllm", False))
    use_mlx = bool(mediq_cfg.get("use_mlx", False))
    mediq_use_api = mediq_cfg.get("use_api")
    doctors = _list_doctors(cfg)

    doctor_meta: Dict[str, Dict[str, Any]] = {}
    for key, block in doctors:
        provider = resolve_provider(block, mediq_use_api=mediq_use_api)
        local = is_local_provider(provider)
        doctor_meta[key] = {
            "key": key,
            "letter": _doctor_letter(key),
            "name": _model_name(block, key, use_mlx=use_mlx and local),
            "provider": provider,
            "local": local,
        }

    patient_block = models.get("patient") or {}
    patient = _model_name(patient_block, "patient")

    report: Dict[str, Any] = {
        "mode": mode,
        "prepared_cases": str(prepared_path),
        "n_cases": len(expected_ids),
        "max_questions": 0,
        "dry_run": dry_run,
        "doctors": {},
    }

    if dry_run:
        # Sanity: base must have empty initial_info; others non-empty when facts exist
        sample = read_jsonl(prepared_path)[:3]
        report["dry_run_samples"] = [
            {
                "id": r.get("id"),
                "initial_info": r.get("initial_info"),
                "ablation": r.get("ablation"),
                "question": r.get("question"),
            }
            for r in sample
        ]
        print(json.dumps(report["dry_run_samples"], indent=2, ensure_ascii=False))
        for key, meta in doctor_meta.items():
            out = mode_dir / key / "results.jsonl"
            report["doctors"][key] = {
                "model": meta["name"],
                "provider": meta["provider"],
                "results": str(out),
                "complete": results_complete(out, expected_ids),
                "skipped": "dry_run",
            }
        return report

    for key, meta in doctor_meta.items():
        out = mode_dir / key / "results.jsonl"
        if force and out.exists():
            print(f"=== --force: removing {out} ===")
            out.unlink()

        if not force and results_complete(out, expected_ids):
            print(f"=== Skip (complete): {mode}/{key} -> {out} ===")
            acc = _accuracy_for_results(out)
            report["doctors"][key] = {
                "model": meta["name"],
                "provider": meta["provider"],
                "results": str(out),
                "complete": True,
                "skipped": True,
                **acc,
            }
            continue

        # Incomplete or missing: re-enter MediQ (skips finished case ids inside benchmark).
        print(f"=== Run / resume: {mode}/{key} ===")
        results_path = run_mediq_role(
            role_name=key,
            expert_model=meta["name"],
            patient_model=patient,
            source_data=prepared_path,
            max_cases=None,
            output_dir=mode_dir,
            mediq_cfg=mediq_cfg,
            expert_provider=meta["provider"],
            prepared_cases=prepared_path,
        )
        acc = _accuracy_for_results(results_path)
        complete = results_complete(results_path, expected_ids)
        report["doctors"][key] = {
            "model": meta["name"],
            "provider": meta["provider"],
            "results": str(results_path),
            "complete": complete,
            "skipped": False,
            **acc,
        }
        if not complete:
            print(
                f"WARNING: {mode}/{key} incomplete "
                f"({acc['n']}/{len(expected_ids)}). Re-run the same command to resume."
            )

    summary_path = mode_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== Mode {mode} summary -> {summary_path} ===")
    return report


def run_ablation(
    config_path: Path,
    *,
    modes: Sequence[str],
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg = load_yaml(config_path)
    if not modes:
        modes = list((cfg.get("ablation") or {}).get("modes") or ABLATION_MODES)

    overall = {
        "config": str(config_path),
        "modes": list(modes),
        "results": {},
    }
    for mode in modes:
        overall["results"][mode] = run_ablation_mode(
            cfg, mode, dry_run=dry_run, force=force
        )

    output_root = ROOT / str(
        (cfg.get("pipeline") or {}).get("output_dir", "mediq_experiment/outputs_ablation")
    )
    ensure_dir(output_root)
    summary_path = output_root / "summary_all.json"
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Ablation done. Overall summary -> {summary_path} ===")
    return overall


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="One-shot fact ablation (base / random / most-important / all_facts)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "mediq_experiment" / "config_ablation.yaml",
        help="Ablation config path.",
    )
    parser.add_argument(
        "--mode",
        choices=list(ABLATION_MODES),
        action="append",
        dest="modes",
        help="Run one mode (repeatable). Default: all modes from config.",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Run every mode in config ablation.modes (or all built-in modes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare cases and print samples only; do not call models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results.jsonl already has all case ids "
        "(does not delete partial files; MediQ still skips existing ids unless you delete results).",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.all_modes:
        cfg = load_yaml(args.config)
        modes = list((cfg.get("ablation") or {}).get("modes") or ABLATION_MODES)
    elif args.modes:
        modes = list(args.modes)
    else:
        # Default: all modes (same as --all-modes) for one-command launch
        cfg = load_yaml(args.config)
        modes = list((cfg.get("ablation") or {}).get("modes") or ABLATION_MODES)

    run_ablation(args.config, modes=modes, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
