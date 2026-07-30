"""
Preflight check before smoke / full MediQ A–E runs.

Checks:
  - config YAML loads and lists A–E
  - HF_HOME exists (and optionally contains cached local models)
  - .env / API keys for patient+judge (and doctor A if present)

Usage (repo root):
  python -m mediq_experiment.check_setup
  python -m mediq_experiment.check_setup --require-models
  python -m mediq_experiment.check_setup --config mediq_experiment/config_smoke.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from mediq_experiment.io_utils import ROOT, load_yaml
from mediq_experiment.model_chat import is_local_provider, resolve_provider
from mediq_experiment.run_pipeline import _list_doctors

DEFAULT_HF_HOME = (
    "/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/hf_cache"
)


def _hub_cache_hint(hf_home: Path, repo_id: str) -> Path:
    # HF hub layout: $HF_HOME/hub/models--org--name
    safe = "models--" + repo_id.replace("/", "--")
    return hf_home / "hub" / safe


def _model_present(hf_home: Path, repo_id: str) -> bool:
    p = _hub_cache_hint(hf_home, repo_id)
    if p.exists() and any(p.rglob("*.safetensors")):
        return True
    if p.exists() and any(p.rglob("*.bin")):
        return True
    # also allow snapshot downloaded via --local-dir into HF_HOME/repo
    alt = hf_home / repo_id.replace("/", "__")
    if alt.exists():
        return True
    # flat folder named like org--name
    alt2 = hf_home / repo_id.replace("/", "--")
    if alt2.exists():
        return True
    return False


def check_setup(
    config_path: Path,
    *,
    require_models: bool = False,
    hf_home: Optional[Path] = None,
) -> int:
    load_dotenv(ROOT / ".env")
    errors: List[str] = []
    warnings: List[str] = []

    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        return 1

    cfg = load_yaml(config_path)
    doctors = _list_doctors(cfg)
    print(f"Config OK: {config_path}")
    print(f"  doctors: {[k for k, _ in doctors]}")
    for key, block in doctors:
        prov = resolve_provider(block)
        print(f"  {key}: {prov} / {block.get('name')} local={is_local_provider(prov)}")

    data = ROOT / str((cfg.get("data") or {}).get("source_path", "data/all_dev_good.jsonl"))
    if not data.exists():
        errors.append(f"data missing: {data}")
    else:
        print(f"Data OK: {data}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        errors.append("ANTHROPIC_API_KEY missing (patient/judge Haiku)")
    else:
        print("ANTHROPIC_API_KEY: set")

    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY missing (needed for doctor_a GPT)")
    else:
        print("OPENAI_API_KEY: set")

    home = Path(
        hf_home
        or os.environ.get("HF_HOME")
        or DEFAULT_HF_HOME
    )
    print(f"HF_HOME: {home} exists={home.exists()}")
    if not home.exists():
        warnings.append(f"HF_HOME does not exist yet: {home}")

    local_ids = []
    for key, block in doctors:
        if is_local_provider(resolve_provider(block)):
            local_ids.append(str(block["name"]))

    if home.exists():
        for repo_id in local_ids:
            ok = _model_present(home, repo_id)
            status = "FOUND" if ok else "NOT FOUND under HF hub cache layout"
            print(f"  model {repo_id}: {status}")
            if require_models and not ok:
                errors.append(
                    f"model not found in HF_HOME for {repo_id} "
                    f"(expected under { _hub_cache_hint(home, repo_id) })"
                )

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors:
        print("check_setup: FAIL")
        return 1
    print("check_setup: OK")
    return 0


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Preflight check for MediQ A–E experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "mediq_experiment" / "config.yaml",
    )
    parser.add_argument(
        "--require-models",
        action="store_true",
        help="Fail if local HF models are not visible under HF_HOME.",
    )
    parser.add_argument("--hf-home", type=Path, default=None)
    args = parser.parse_args(argv)
    sys.exit(
        check_setup(args.config, require_models=args.require_models, hf_home=args.hf_home)
    )


if __name__ == "__main__":
    main()
