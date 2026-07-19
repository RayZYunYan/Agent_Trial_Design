"""API keys for MediQ ``helper.ModelCache``.

Loads from repo-root ``.env`` (``OPENAI_API_KEY``, ``GROQ_API_KEY``, ``ANTHROPIC_API_KEY``).
``args.api_account`` selects the dict key for OpenAI; Groq/Anthropic use their own env keys.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

_openai = (os.environ.get("OPENAI_API_KEY") or "").strip()
_groq = (os.environ.get("GROQ_API_KEY") or "").strip()
_anthropic = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()

# MediQ CLI default ``--api_account mediQ``.
mykey = {
    "mediQ": _openai,
    "openai": _openai,
    "groq": _groq,
    "anthropic": _anthropic,
}
