"""Run upstream MediQ benchmark (src/mediQ_benchmark.py) with hot-swappable models."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mediq_experiment.io_utils import ROOT, ensure_dir, write_subset_jsonl


def build_mediq_cmd(
    *,
    expert_model: str,
    patient_model: str,
    data_dir: Path,
    dev_filename: str,
    output_filename: Path,
    log_dir: Path,
    mediq_cfg: Dict[str, Any],
    expert_provider: Optional[str] = None,
) -> List[str]:
    ensure_dir(log_dir)
    ensure_dir(output_filename.parent)
    cmd = [
        sys.executable,
        "mediQ_benchmark.py",
        "--expert_module",
        "expert",
        "--expert_class",
        str(mediq_cfg.get("expert_class", "BasicExpert")),
        "--patient_module",
        "patient",
        "--patient_class",
        str(mediq_cfg.get("patient_class", "FactSelectPatient")),
        "--expert_model",
        expert_model,
        "--expert_model_question_generator",
        expert_model,
        "--patient_model",
        patient_model,
        "--data_dir",
        str(data_dir.resolve()),
        "--dev_filename",
        dev_filename,
        "--output_filename",
        str(output_filename.resolve()),
        "--max_questions",
        str(int(mediq_cfg.get("max_questions", 10))),
        "--log_filename",
        str((log_dir / "results.log").resolve()),
        "--history_log_filename",
        str((log_dir / "history.log").resolve()),
        "--detail_log_filename",
        str((log_dir / "detail.log").resolve()),
        "--message_log_filename",
        str((log_dir / "messages.log").resolve()),
        "--temperature",
        str(mediq_cfg.get("temperature", 0.6)),
        "--max_tokens",
        str(int(mediq_cfg.get("max_tokens", 256))),
        "--self_consistency",
        str(int(mediq_cfg.get("self_consistency", 1))),
        "--abstain_threshold",
        str(float(mediq_cfg.get("abstain_threshold", 4.0))),
        "--api_account",
        "mediQ",
    ]
    # Prefer explicit doctor provider; helper also infers from model id for patient.
    use_api = expert_provider or mediq_cfg.get("use_api")
    if use_api:
        cmd.extend(["--use_api", str(use_api)])
    if mediq_cfg.get("rationale_generation", True):
        cmd.append("--rationale_generation")
    return cmd


def run_mediq_role(
    *,
    role_name: str,
    expert_model: str,
    patient_model: str,
    source_data: Path,
    max_cases: Optional[int],
    output_dir: Path,
    mediq_cfg: Dict[str, Any],
    expert_provider: Optional[str] = None,
) -> Path:
    """Prepare subset, run MediQ from ``src/``, return results JSONL path."""
    run_dir = ensure_dir(output_dir / role_name)
    subset_path = run_dir / "cases_subset.jsonl"
    write_subset_jsonl(source_data, subset_path, max_cases)
    out_jsonl = run_dir / "results.jsonl"

    # Resume-friendly: MediQ skips ids already in output. Delete to force re-run.
    cmd = build_mediq_cmd(
        expert_model=expert_model,
        patient_model=patient_model,
        data_dir=subset_path.parent,
        dev_filename=subset_path.name,
        output_filename=out_jsonl,
        log_dir=run_dir / "logs",
        mediq_cfg=mediq_cfg,
        expert_provider=expert_provider,
    )
    print(f"\n=== MediQ run: {role_name} (doctor={expert_model}, provider={expert_provider}) ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT / "src"), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"MediQ run failed for {role_name} (exit {proc.returncode})")
    if not out_jsonl.exists():
        raise RuntimeError(f"MediQ produced no output: {out_jsonl}")
    return out_jsonl
