"""
Medical knowledge retriever for SMART Trial RAG.

Backends:
  BM25Retriever       — keyword-based (rank_bm25), no GPU needed
  ContrieverRetriever — dense semantic retrieval (facebook/contriever)

Corpus: MedRAG/textbooks — 9 clinically relevant textbooks selected for
diagnostic reasoning (Harrison's, Nelson, Adams, DSM-5, Katzung, Robbins,
Schwartz, Williams OB/GYN, Novak Gynecology, First Aid Step 2).

Usage:
  # BM25
  retriever = BM25Retriever.load(cache_path="smart_trial/data/bm25_index.pkl")

  # Contriever (needs BM25 passages)
  bm25 = BM25Retriever.load(...)
  retriever = ContrieverRetriever.load(passages=bm25._passages, cache_path="...")

  passages = retriever.retrieve("first-line antibiotics for gonorrhea", k=3)
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

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

    def retrieve_topn(self, query: str, n: int) -> List[Tuple[int, str]]:
        """Return (corpus_index, passage) pairs for top-n results. Used by HybridRetriever."""
        raise NotImplementedError


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

    def retrieve_topn(self, query: str, n: int) -> List[Tuple[int, str]]:
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(i, self._passages[i]) for i in top_n]

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


class ContrieverRetriever(BaseRetriever):
    """Dense retrieval using facebook/contriever (mean-pooled BERT encoder)."""

    def __init__(self, passages: List[str], embeddings, tokenizer, model) -> None:
        self._passages = passages
        self._embeddings = embeddings  # np.ndarray, shape (N, 768)
        self._tokenizer = tokenizer
        self._model = model

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        passages: List[str],
        cache_path: str = "smart_trial/data/contriever_embeddings.npy",
        model_name: str = "facebook/contriever",
        force_rebuild: bool = False,
        batch_size: int = 64,
    ) -> "ContrieverRetriever":
        import numpy as np
        from transformers import AutoModel, AutoTokenizer

        import torch
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"[RAG] Loading Contriever model: {model_name} (device: {device})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        path = Path(cache_path)
        if path.exists() and not force_rebuild:
            print(f"[RAG] Loading Contriever embeddings from {path} ...")
            embeddings = np.load(str(path))
            return cls(passages=passages, embeddings=embeddings, tokenizer=tokenizer, model=model)

        print(f"[RAG] Building Contriever embeddings for {len(passages):,} passages ...")
        embeddings = cls._encode_passages(passages, tokenizer, model, batch_size, device)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), embeddings)
        print(f"[RAG] Embeddings saved to {path}")
        return cls(passages=passages, embeddings=embeddings, tokenizer=tokenizer, model=model)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        import numpy as np
        query_emb = self._encode_query(query)           # (1, 768)
        scores = (self._embeddings @ query_emb.T).squeeze()  # (N,)
        top_k = np.argsort(scores)[::-1][:k]
        return [self._passages[int(i)] for i in top_k]

    def retrieve_topn(self, query: str, n: int) -> List[Tuple[int, str]]:
        import numpy as np
        query_emb = self._encode_query(query)
        scores = (self._embeddings @ query_emb.T).squeeze()
        top_n = np.argsort(scores)[::-1][:n]
        return [(int(i), self._passages[int(i)]) for i in top_n]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_pool(model_output, attention_mask):
        import torch
        token_emb = model_output.last_hidden_state          # (B, T, 768)
        mask = attention_mask.unsqueeze(-1).float()         # (B, T, 1)
        return torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    @classmethod
    def _encode_passages(cls, passages: List[str], tokenizer, model, batch_size: int, device: str = "cpu"):
        import numpy as np
        import torch

        all_embs = []
        for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
            batch = passages[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            embs = cls._mean_pool(out, inputs["attention_mask"])
            all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs)

    def _encode_query(self, query: str):
        import torch
        device = next(self._model.parameters()).device
        inputs = self._tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        return self._mean_pool(out, inputs["attention_mask"]).cpu().numpy()  # (1, 768)


class HybridRetriever(BaseRetriever):
    """BM25 + Contriever combined with Reciprocal Rank Fusion (RRF).

    RRF score(doc) = 1/(rrf_k + rank_bm25) + 1/(rrf_k + rank_contriever)
    Both retrievers must be built from the same passages corpus so indices align.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        contriever: ContrieverRetriever,
        rrf_k: int = 60,
    ) -> None:
        self._bm25 = bm25
        self._contriever = contriever
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        n = max(k * 10, 30)  # broader candidate set before fusion
        bm25_ranked = self._bm25.retrieve_topn(query, n)
        contriever_ranked = self._contriever.retrieve_topn(query, n)

        print(f"[RAG:Hybrid] BM25 top-1:       {bm25_ranked[0][1][:80]!r}")
        print(f"[RAG:Hybrid] Contriever top-1: {contriever_ranked[0][1][:80]!r}")

        scores: dict = {}
        passages_map: dict = {}
        for rank, (idx, passage) in enumerate(bm25_ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            passages_map[idx] = passage
        for rank, (idx, passage) in enumerate(contriever_ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            passages_map[idx] = passage

        top_k = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
        print(f"[RAG:Hybrid] RRF top-1:        {passages_map[top_k[0]][:80]!r}")
        return [passages_map[i] for i in top_k]
