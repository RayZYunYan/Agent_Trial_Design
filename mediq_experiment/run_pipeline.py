"""
One-click MediQ multi-doctor experiment (A–E).

Steps (configurable in mediq_experiment/config.yaml):
  1. Run upstream MediQ for each doctor_* (skip non-empty results.jsonl)
  2. End-of-case fact coverage (skip non-empty scored.jsonl)
  3. Full cross finalize: every transcript → every other doctor answers once
     (skip non-empty cross/*_transcript_*_answers.jsonl)

Usage (from repo root):
  python -m mediq_experiment.run_pipeline
  python -m mediq_experiment.run_pipeline --config mediq_experiment/config.yaml
  python -m mediq_experiment.run_pipeline --skip-mediq   # only score/cross existing outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from mediq_experiment.cross_finalize import run_cross
from mediq_experiment.io_utils import ROOT, accuracy_summary, ensure_dir, load_yaml, read_jsonl
from mediq_experiment.model_chat import is_local_provider, resolve_provider
from mediq_experiment.run_mediq import run_mediq_role
from mediq_experiment.score_coverage import score_file


def _artifact_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _doctor_letter(key: str) -> str:
    # doctor_a → a
    if key.startswith("doctor_") and len(key) > len("doctor_"):
        return key[len("doctor_") :]
    return key


def _list_doctors(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    models = cfg.get("models") or {}
    doctors = []
    for key, block in models.items():
        if not str(key).startswith("doctor_"):
            continue
        if not isinstance(block, dict) or not block.get("name"):
            raise ValueError(f"config.models.{key}.name is required")
        doctors.append((str(key), block))
    doctors.sort(key=lambda x: _doctor_letter(x[0]))
    if not doctors:
        raise ValueError("config.models must define at least one doctor_* entry")
    return doctors


def _model_name(block: Dict[str, Any], key: str, *, use_mlx: bool = False) -> str:
    """Prefer mlx_name for Apple Silicon when configured."""
    if use_mlx and block.get("mlx_name"):
        return str(block["mlx_name"])
    name = block.get("name")
    if not name:
        raise ValueError(f"config.models.{key}.name is required")
    return str(name)


def run_pipeline(config_path: Path, *, skip_mediq: bool = False) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg = load_yaml(config_path)

    data_cfg = cfg.get("data") or {}
    mediq_cfg = cfg.get("mediq") or {}
    pipe_cfg = cfg.get("pipeline") or {}
    models = cfg.get("models") or {}
    skip_existing = bool(pipe_cfg.get("skip_existing", True))

    source = ROOT / str(data_cfg.get("source_path", "data/all_dev_good.jsonl"))
    if not source.exists():
        raise FileNotFoundError(f"Data not found: {source}")

    max_cases = data_cfg.get("max_cases")
    id_min = data_cfg.get("id_min")
    id_max = data_cfg.get("id_max")
    if id_min is not None:
        id_min = int(id_min)
    if id_max is not None:
        id_max = int(id_max)
    output_dir = ensure_dir(ROOT / str(pipe_cfg.get("output_dir", "mediq_experiment/outputs")))

    use_vllm = bool(mediq_cfg.get("use_vllm", False))
    use_mlx = bool(mediq_cfg.get("use_mlx", False))
    load_in_4bit = bool(mediq_cfg.get("load_in_4bit", False))

    doctors = _list_doctors(cfg)
    mediq_use_api = mediq_cfg.get("use_api")
    doctor_meta: Dict[str, Dict[str, Any]] = {}
    for key, block in doctors:
        provider = resolve_provider(block, mediq_use_api=mediq_use_api)
        local = is_local_provider(provider)
        doctor_meta[key] = {
            "key": key,
            "letter": _doctor_letter(key),
            "name": _model_name(block, key, use_mlx=use_mlx and local),
            "hf_name": _model_name(block, key, use_mlx=False),
            "provider": provider,
            "local": local,
        }

    patient_block = models.get("patient") or {}
    patient = _model_name(patient_block, "patient")
    patient_provider = resolve_provider(patient_block, mediq_use_api=mediq_use_api)

    judge = models.get("judge") or {}
    judge_provider = resolve_provider(judge, mediq_use_api=mediq_use_api)
    judge_model = str(judge.get("name", "claude-haiku-4-5"))
    judge_temp = float(judge.get("temperature", 0.1))

    report: Dict[str, Any] = {
        "config": str(config_path),
        "data": str(source),
        "max_cases": max_cases,
        "id_min": id_min,
        "id_max": id_max,
        "accelerator": ("mlx" if use_mlx else ("vllm" if use_vllm else "transformers")),
        "models": {
            **{k: f"{m['provider']}/{m['name']}" for k, m in doctor_meta.items()},
            "patient": f"{patient_provider}/{patient}",
            "judge": f"{judge_provider}/{judge_model}",
        },
        "skip_existing": skip_existing,
    }

    results_paths: Dict[str, Path] = {}
    for key, meta in doctor_meta.items():
        path = output_dir / key / "results.jsonl"
        results_paths[key] = path

    if pipe_cfg.get("run_mediq", True) and not skip_mediq:
        for key, meta in doctor_meta.items():
            path = results_paths[key]
            if skip_existing and _artifact_ready(path):
                print(f"\n=== Skip MediQ (exists): {key} → {path} ===")
                continue
            results_paths[key] = run_mediq_role(
                role_name=key,
                expert_model=meta["name"],
                patient_model=patient,
                source_data=source,
                max_cases=max_cases,
                output_dir=output_dir,
                mediq_cfg=mediq_cfg,
                expert_provider=meta["provider"],
                id_min=id_min,
                id_max=id_max,
            )
    else:
        missing = [str(p) for p in results_paths.values() if not _artifact_ready(p)]
        if missing:
            raise FileNotFoundError(
                "Missing MediQ outputs:\n  "
                + "\n  ".join(missing)
                + "\nRun without --skip-mediq first (or disable skip for incomplete files)."
            )

    report["mediq_outputs"] = {k: str(p) for k, p in results_paths.items()}
    report["mediq_self"] = {}
    for key, path in results_paths.items():
        rows = read_jsonl(path)
        n = len(rows)
        correct = 0
        for row in rows:
            interactive = row.get("interactive_system") or {}
            if interactive.get("correct"):
                correct += 1
        report["mediq_self"][key] = {
            "n": n,
            "correct": correct,
            "accuracy": (correct / n) if n else None,
            "model": doctor_meta[key]["name"],
        }

    if pipe_cfg.get("score_coverage", True):
        print("\n=== End-of-case fact coverage ===")
        report["coverage"] = {}
        for key, path in results_paths.items():
            scored = output_dir / key / "scored.jsonl"
            report["coverage"][key] = score_file(
                path,
                scored,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_temperature=judge_temp,
                skip_if_exists=skip_existing,
            )

    if pipe_cfg.get("cross_finalize", True):
        print("\n=== Cross-model finalize (full) ===")
        cross_dir = ensure_dir(output_dir / "cross")
        report["cross"] = {}
        keys = list(doctor_meta.keys())
        for src_key in keys:
            for dst_key in keys:
                if src_key == dst_key:
                    continue
                src_letter = doctor_meta[src_key]["letter"]
                dst_letter = doctor_meta[dst_key]["letter"]
                out_name = f"{src_letter}_transcript_{dst_letter}_answers.jsonl"
                out_path = cross_dir / out_name
                pair_id = f"{src_letter}_to_{dst_letter}"
                if skip_existing and _artifact_ready(out_path):
                    rows = read_jsonl(out_path)
                    summary = accuracy_summary(rows)
                    summary.update(
                        {
                            "source": str(results_paths[src_key]),
                            "output": str(out_path),
                            "responder_model": doctor_meta[dst_key]["name"],
                            "skipped": True,
                        }
                    )
                    print(f"  skip cross (exists): {out_name}")
                    report["cross"][pair_id] = summary
                    continue
                dst = doctor_meta[dst_key]
                report["cross"][pair_id] = run_cross(
                    results_paths[src_key],
                    out_path,
                    responder_provider=dst["provider"],
                    responder_model=dst["name"],
                    use_vllm=use_vllm and dst["local"],
                    use_mlx=use_mlx and dst["local"],
                    load_in_4bit=load_in_4bit and dst["local"],
                    hf_fallback_name=dst["hf_name"] if dst["local"] else None,
                )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Done. Summary written to {summary_path} ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="One-click MediQ multi-doctor experiment (A–E).")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "mediq_experiment" / "config.yaml",
        help="Path to experiment config (models are hot-swappable here).",
    )
    parser.add_argument(
        "--skip-mediq",
        action="store_true",
        help="Skip dialogue runs; only score coverage / cross on existing results.jsonl.",
    )
    args = parser.parse_args(argv)
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    run_pipeline(args.config, skip_mediq=args.skip_mediq)


if __name__ == "__main__":
    main()
