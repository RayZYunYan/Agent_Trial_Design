"""Provider smoke check: for every role and agent in a config, make one tiny
real chat call and report status — the week-1 gate before any grid run.

  python -m clinic_experiment.check_providers                      # config.yaml
  python -m clinic_experiment.check_providers --config path.yaml

Reports per block: OK (with latency + token counts), MISSING KEY (env var
empty), or the API error line. Also flags ANTHROPIC_BASE_URL when set, since
a restricted relay there blocks every Anthropic role at once.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from mediq_experiment.io_utils import ROOT, load_yaml
from smart_trial.models.model_client import ModelClient

PING = [{"role": "user", "content": "Reply with exactly: ok"}]


def _blocks(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, block in (cfg.get("models") or {}).items():
        out.append((f"models.{name}", block))
    for spec in cfg.get("agents") or []:
        out.append((f"agent {spec.get('id')}", spec))
    return out


def check_block(label: str, block: Dict[str, Any]) -> bool:
    provider = block.get("provider")
    name = block.get("name")
    if provider not in ("mock", "mlx_local", "cursor_sdk"):
        env_var = block.get("api_key_env") or f"{str(provider).upper()}_API_KEY"
        if not os.environ.get(env_var):
            print(f"  {label:<18} {provider}/{name}: MISSING KEY ({env_var} empty)")
            return False
    try:
        client = ModelClient(
            provider,
            name,
            temperature=0.0,
            base_url=block.get("base_url"),
            api_key_env=block.get("api_key_env"),
        )
        text, meta = client.chat_ex(PING, max_retries=1)
        latency = meta.get("latency_s")
        latency_str = f"{latency:.2f}s" if latency is not None else "?"
        print(
            f"  {label:<18} {provider}/{name}: OK ({latency_str}, "
            f"in={meta.get('input_tokens')} out={meta.get('output_tokens')}) -> {text[:40]!r}"
        )
        return True
    except Exception as e:
        print(f"  {label:<18} {provider}/{name}: FAILED — {str(e)[:160]}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="One tiny live call per configured model block.")
    parser.add_argument("--config", type=Path, default=ROOT / "clinic_experiment" / "config.yaml")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    # A stale ANTHROPIC_BASE_URL exported in the parent shell would silently
    # redirect direct-Anthropic calls (patient/reporter/judge/agent B) to a
    # relay. Agents that want a relay pass base_url explicitly via config.
    os.environ.pop("ANTHROPIC_BASE_URL", None)
    if os.environ.get("ANTHROPIC_BASE_URL"):
        print(
            "NOTE: ANTHROPIC_BASE_URL is set (relay). If Anthropic calls fail with "
            "403/503 group errors, the token is restricted — use a real key and "
            "delete ANTHROPIC_BASE_URL from .env."
        )

    cfg = load_yaml(args.config)
    results = [check_block(label, block) for label, block in _blocks(cfg)]
    ok = sum(results)
    print(f"\n{ok}/{len(results)} blocks working")
    raise SystemExit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
