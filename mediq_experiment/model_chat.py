"""Chat helper for MediQ experiment: API (ModelClient) or local HF/vLLM (src.helper)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mediq_experiment.io_utils import ROOT

_API_PROVIDERS = frozenset({"openai", "anthropic", "groq", "gemini"})
_LOCAL_PROVIDERS = frozenset({"huggingface", "hf", "local", "vllm", "transformers", ""})


def is_local_provider(provider: Optional[str]) -> bool:
    p = (provider or "").strip().lower()
    if p in _API_PROVIDERS:
        return False
    if p in _LOCAL_PROVIDERS or p.startswith("hugging"):
        return True
    return p not in _API_PROVIDERS


def resolve_provider(block: Dict[str, Any], *, mediq_use_api: Optional[str] = None) -> str:
    """Resolve provider string from a models.* config block."""
    provider = block.get("provider")
    if provider:
        return str(provider).strip().lower()
    name = str(block.get("name") or "").lower()
    if "claude" in name:
        return "anthropic"
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    if "/" in name:  # Hugging Face repo id
        return "huggingface"
    if mediq_use_api:
        return str(mediq_use_api).strip().lower()
    return "huggingface"


def api_use_flag(provider: Optional[str]) -> Optional[str]:
    """Value for MediQ --use_api, or None for local HF/vLLM."""
    p = (provider or "").strip().lower()
    if p in _API_PROVIDERS:
        return p
    return None


def chat(
    *,
    provider: str,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    use_vllm: bool = False,
    use_mlx: bool = False,
    load_in_4bit: bool = False,
    hf_fallback_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Single-turn chat; local path uses MediQ helper.get_response."""
    if not is_local_provider(provider):
        from smart_trial.models.model_client import ModelClient

        client = ModelClient(provider, model_name, temperature=temperature)
        return client.chat(messages, system_prompt=system_prompt, temperature=temperature)

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from helper import get_response

    full = []
    if system_prompt:
        full.append({"role": "system", "content": system_prompt})
    full.extend(messages)
    text, _, _ = get_response(
        full,
        model_name=model_name,
        use_vllm=use_vllm,
        use_mlx=use_mlx,
        use_api=None,
        temperature=temperature,
        max_tokens=max_tokens,
        load_in_4bit=load_in_4bit,
        hf_fallback_name=hf_fallback_name,
    )
    return text or ""
