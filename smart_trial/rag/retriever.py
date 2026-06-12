"""
Medical knowledge retriever for SMART Trial RAG.

Backends:
  BM25Retriever  — keyword-based (rank_bm25), no GPU needed
  (FAISSRetriever — to be added later for semantic search)

Corpus: MedRAG/textbooks — 9 clinically relevant textbooks selected for
diagnostic reasoning (Harrison's, Nelson, Adams, DSM-5, Katzung, Robbins,
Schwartz, Williams OB/GYN, Novak Gynecology, First Aid Step 2).

Usage:
  retriever = BM25Retriever.load(cache_path="smart_trial/data/bm25_index.pkl")
  passages = retriever.retrieve("first-line antibiotics for gonorrhea", k=3)
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from tqdm import tqdm

# 9 textbooks selected for clinical diagnosis relevance
_CLINICAL_TEXTBOOKS = [
    "chunk/First_Aid_Step2.jsonl",
    "chunk/InternalMed_Harrison.jsonl",
    "chunk/Pediatrics_Nelson.jsonl",
    "chunk/Neurology_Adams.jsonl",
    "chunk/Psichiatry_DSM-5.jsonl",
    "chunk/Pharmacology_Katzung.jsonl",
    "chunk/Pathology_Robbins.jsonl",
    "chunk/Surgery_Schwartz.jsonl",
    "chunk/Gynecology_Novak.jsonl",
    "chunk/Obstentrics_Williams.jsonl",
]


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Return top-k relevant passages for the query."""


class BM25Retriever(BaseRetriever):
    def __init__(self, passages: List[str], bm25_index) -> None:
        self._passages = passages
        self._index = bm25_index

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        cache_path: str = "smart_trial/data/bm25_index.pkl",
        force_rebuild: bool = False,
    ) -> "BM25Retriever":
        path = Path(cache_path)
        if path.exists() and not force_rebuild:
            print(f"[RAG] Loading BM25 index from {path} ...")
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return cls(passages=obj["passages"], bm25_index=obj["index"])

        print("[RAG] Building BM25 index from MedRAG/textbooks ...")
        passages = cls._download_textbooks()
        index = cls._build_index(passages)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"passages": passages, "index": index}, f)
        print(f"[RAG] Index saved to {path}")
        return cls(passages=passages, bm25_index=index)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._passages[i] for i in top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _download_textbooks() -> List[str]:
        from datasets import load_dataset
        passages = []
        for book in tqdm(_CLINICAL_TEXTBOOKS, desc="Loading textbooks"):
            ds = load_dataset("MedRAG/textbooks", data_files=book, split="train")
            for row in ds:
                content = (row.get("content") or "").strip()
                if content:
                    passages.append(content)
        print(f"[RAG] {len(passages):,} passages loaded from {len(_CLINICAL_TEXTBOOKS)} textbooks")
        return passages

    @staticmethod
    def _build_index(passages: List[str]):
        from rank_bm25 import BM25Okapi
        tokenised = [p.lower().split() for p in tqdm(passages, desc="Tokenising")]
        return BM25Okapi(tokenised)
