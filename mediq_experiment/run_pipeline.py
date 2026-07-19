"""
One-click MediQ two-model experiment.

Steps (configurable in mediq_experiment/config.yaml):
  1. Run upstream MediQ for doctor A and doctor B
  2. End-of-case fact coverage + final accuracy
  3. Cross finalize: A transcript → B answer, B transcript → A answer

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
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from mediq_experiment.cross_finalize import run_cross
from mediq_experiment.io_utils import ROOT, ensure_dir, load_yaml
from mediq_experiment.run_mediq import run_mediq_role
from mediq_experiment.score_coverage import score_file


def _model_name(cfg: Dict[str, Any], key: str) -> str:
    block = (cfg.get("models") or {}).get(key) or {}
    name = block.get("name")
    if not name:
        raise ValueError(f"config.models.{key}.name is required")
    return str(name)


def _model_provider(cfg: Dict[str, Any], key: str) -> str:
    """Resolve provider from config, else infer from model name."""
    block = (cfg.get("models") or {}).get(key) or {}
    provider = block.get("provider")
    if provider:
        return str(provider)
    name = str(block.get("name") or "").lower()
    if "claude" in name:
        return "anthropic"
    if "gpt" in name or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    mediq_api = (cfg.get("mediq") or {}).get("use_api")
    if mediq_api:
        return str(mediq_api)
    return "openai"


def run_pipeline(config_path: Path, *, skip_mediq: bool = False) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg = load_yaml(config_path)

    data_cfg = cfg.get("data") or {}
    mediq_cfg = cfg.get("mediq") or {}
    pipe_cfg = cfg.get("pipeline") or {}
    models = cfg.get("models") or {}

    source = ROOT / str(data_cfg.get("source_path", "data/all_dev_good.jsonl"))
    if not source.exists():
        raise FileNotFoundError(f"Data not found: {source}")

    max_cases = data_cfg.get("max_cases")
    output_dir = ensure_dir(ROOT / str(pipe_cfg.get("output_dir", "mediq_experiment/outputs")))

    doctor_a = _model_name(cfg, "doctor_a")
    doctor_b = _model_name(cfg, "doctor_b")
    patient = _model_name(cfg, "patient")
    provider_a = _model_provider(cfg, "doctor_a")
    provider_b = _model_provider(cfg, "doctor_b")
    judge = models.get("judge") or {}
    judge_provider = str(judge.get("provider") or _model_provider(cfg, "judge"))
    judge_model = str(judge.get("name", "claude-haiku-4-5"))
    judge_temp = float(judge.get("temperature", 0.1))

    report: Dict[str, Any] = {
        "config": str(config_path),
        "data": str(source),
        "max_cases": max_cases,
        "models": {
            "doctor_a": f"{provider_a}/{doctor_a}",
            "doctor_b": f"{provider_b}/{doctor_b}",
            "patient": f"{_model_provider(cfg, 'patient')}/{patient}",
            "judge": f"{judge_provider}/{judge_model}",
        },
    }

    path_a = output_dir / "doctor_a" / "results.jsonl"
    path_b = output_dir / "doctor_b" / "results.jsonl"

    if pipe_cfg.get("run_mediq", True) and not skip_mediq:
        path_a = run_mediq_role(
            role_name="doctor_a",
            expert_model=doctor_a,
            patient_model=patient,
            source_data=source,
            max_cases=max_cases,
            output_dir=output_dir,
            mediq_cfg=mediq_cfg,
            expert_provider=provider_a,
        )
        path_b = run_mediq_role(
            role_name="doctor_b",
            expert_model=doctor_b,
            patient_model=patient,
            source_data=source,
            max_cases=max_cases,
            output_dir=output_dir,
            mediq_cfg=mediq_cfg,
            expert_provider=provider_b,
        )
    else:
        if not path_a.exists() or not path_b.exists():
            raise FileNotFoundError(
                f"Missing MediQ outputs. Expected:\n  {path_a}\n  {path_b}\n"
                "Run without --skip-mediq first."
            )

    report["mediq_outputs"] = {"doctor_a": str(path_a), "doctor_b": str(path_b)}

    if pipe_cfg.get("score_coverage", True):
        print("\n=== End-of-case fact coverage ===")
        scored_a = output_dir / "doctor_a" / "scored.jsonl"
        scored_b = output_dir / "doctor_b" / "scored.jsonl"
        report["coverage"] = {
            "doctor_a": score_file(
                path_a,
                scored_a,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_temperature=judge_temp,
            ),
            "doctor_b": score_file(
                path_b,
                scored_b,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_temperature=judge_temp,
            ),
        }

    if pipe_cfg.get("cross_finalize", True):
        print("\n=== Cross-model finalize ===")
        cross_dir = ensure_dir(output_dir / "cross")
        a_to_b = cross_dir / "a_transcript_b_answers.jsonl"
        b_to_a = cross_dir / "b_transcript_a_answers.jsonl"
        report["cross"] = {
            "a_to_b": run_cross(
                path_a,
                a_to_b,
                responder_provider=provider_b,
                responder_model=doctor_b,
            ),
            "b_to_a": run_cross(
                path_b,
                b_to_a,
                responder_provider=provider_a,
                responder_model=doctor_a,
            ),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Done. Summary written to {summary_path} ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="One-click MediQ two-model experiment.")
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
