"""Build BM25 RAG index cache for grid eval (downloads MedRAG/textbooks once)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_trial.rag.retriever import BM25Retriever

CACHE = PROJECT_ROOT / "smart_trial" / "data" / "bm25_index.pkl"


def main() -> int:
    print(f"Building BM25 index -> {CACHE}")
    BM25Retriever.load(cache_path=str(CACHE), force_rebuild=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
