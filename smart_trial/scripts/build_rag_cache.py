#!/usr/bin/env python3
"""
Build RAG cache files required for BM25 and/or Contriever retrieval.

Run from repo root:
    python smart_trial/scripts/build_rag_cache.py            # build both
    python smart_trial/scripts/build_rag_cache.py --mode bm25
    python smart_trial/scripts/build_rag_cache.py --mode contriever
    python smart_trial/scripts/build_rag_cache.py --force    # rebuild even if cache exists

Output files (gitignored, must be built locally):
    smart_trial/data/bm25_index.pkl          (~172 MB, ~10-20 min)
    smart_trial/data/contriever_embeddings.npy (~289 MB, ~30 min on MPS/GPU)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SMART_TRIAL_ROOT = PROJECT_ROOT / "smart_trial"
BM25_CACHE = SMART_TRIAL_ROOT / "data" / "bm25_index.pkl"
CONTRIEVER_CACHE = SMART_TRIAL_ROOT / "data" / "contriever_embeddings.npy"
CONTRIEVER_MODEL = "facebook/contriever"


def build_bm25(force: bool = False) -> None:
    from smart_trial.rag.retriever import BM25Retriever

    if BM25_CACHE.exists() and not force:
        print(f"[BM25] Cache already exists at {BM25_CACHE} — skipping. Use --force to rebuild.")
        return

    print("[BM25] Building index from MedRAG/textbooks (downloads ~1 GB from HuggingFace) ...")
    BM25Retriever.load(cache_path=str(BM25_CACHE), force_rebuild=force)
    print(f"[BM25] Done. Saved to {BM25_CACHE}")


def build_contriever(force: bool = False) -> None:
    from smart_trial.rag.retriever import BM25Retriever, ContrieverRetriever

    if CONTRIEVER_CACHE.exists() and not force:
        print(f"[Contriever] Cache already exists at {CONTRIEVER_CACHE} — skipping. Use --force to rebuild.")
        return

    print("[Contriever] Loading passages from BM25 cache ...")
    bm25 = BM25Retriever.load(cache_path=str(BM25_CACHE))

    print(f"[Contriever] Encoding {len(bm25._passages):,} passages with {CONTRIEVER_MODEL} ...")
    print("[Contriever] This takes ~30 min on Apple MPS / ~3-5 min on NVIDIA GPU / ~18 hr on CPU.")
    ContrieverRetriever.load(
        passages=bm25._passages,
        cache_path=str(CONTRIEVER_CACHE),
        model_name=CONTRIEVER_MODEL,
        force_rebuild=force,
    )
    print(f"[Contriever] Done. Saved to {CONTRIEVER_CACHE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG cache files for SMART Trial.")
    parser.add_argument(
        "--mode",
        choices=["bm25", "contriever", "all"],
        default="all",
        help="Which cache to build (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if it already exists",
    )
    args = parser.parse_args()

    if args.mode in ("bm25", "all"):
        build_bm25(force=args.force)

    if args.mode in ("contriever", "all"):
        if not BM25_CACHE.exists():
            print("[Contriever] BM25 cache not found — building it first ...")
            build_bm25(force=False)
        build_contriever(force=args.force)

    print("\nAll requested caches are ready.")
    print("Set 'rag.retriever_mode: hybrid' in trial_config.yaml to use BM25 + Contriever RRF.")


if __name__ == "__main__":
    main()
