import pytest
from pathlib import Path
from unittest.mock import MagicMock

from smart_trial.core.orchestrator import TrialOrchestrator
from smart_trial.rag.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Test 1 — BM25Retriever loads and returns relevant passages
# ---------------------------------------------------------------------------

PKL_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_index.pkl"
NPY_PATH = Path(__file__).resolve().parent.parent / "data" / "contriever_embeddings.npy"


@pytest.mark.skipif(not PKL_PATH.exists(), reason="BM25 index not built yet — run BM25Retriever.load() first")
def test_bm25_retriever_returns_passages():
    from smart_trial.rag.retriever import BM25Retriever
    retriever = BM25Retriever.load(cache_path=str(PKL_PATH))
    results = retriever.retrieve("first-line antibiotics for gonorrhea", k=3)
    assert len(results) == 3
    assert all(isinstance(p, str) and len(p) > 0 for p in results)
    combined = " ".join(results).lower()
    assert any(kw in combined for kw in ["gonorrhea", "antibiotic", "treatment", "infection"])


# ---------------------------------------------------------------------------
# Test 2 — _parse_retrieval_query extracts text correctly
# ---------------------------------------------------------------------------

def test_parse_retrieval_query_extracts_text():
    query = TrialOrchestrator._parse_retrieval_query(
        "[CONFIDENCE: 0.55]\n[RETRIEVAL QUERY: first-line treatment for gonorrhea]\nDo you have discharge?"
    )
    assert query == "first-line treatment for gonorrhea"


def test_parse_retrieval_query_returns_none_when_absent():
    assert TrialOrchestrator._parse_retrieval_query("Do you have any pain?") is None


# ---------------------------------------------------------------------------
# Test 3 — _format_retrieval_result produces correct structure
# ---------------------------------------------------------------------------

def test_format_retrieval_result_structure():
    passages = ["Passage one about gonorrhea.", "Passage two about treatment."]
    result = TrialOrchestrator._format_retrieval_result(passages)
    assert result.startswith("[RETRIEVAL RESULT]")
    assert result.endswith("[/RETRIEVAL RESULT]")
    assert "Passage one about gonorrhea." in result
    assert "Passage two about treatment." in result


# ---------------------------------------------------------------------------
# Test 4 — HybridRetriever RRF fusion logic
# ---------------------------------------------------------------------------

def _make_mock_retriever(ranked_pairs):
    """Build a mock retriever whose retrieve_topn returns the given (idx, passage) list."""
    mock = MagicMock()
    mock.retrieve_topn.return_value = ranked_pairs
    return mock


def test_hybrid_rrf_prefers_doc_ranked_high_by_both():
    """A doc ranked #1 by BM25 and #1 by Contriever should beat a doc ranked #1 by only one."""
    # doc A (idx=0): rank 0 in BM25, rank 0 in Contriever  → RRF ≈ 1/61 + 1/61
    # doc B (idx=1): rank 0 in BM25 only                   → RRF ≈ 1/61
    # doc C (idx=2): rank 0 in Contriever only              → RRF ≈ 1/61
    bm25_mock = _make_mock_retriever([(0, "passage A"), (1, "passage B")])
    contriever_mock = _make_mock_retriever([(0, "passage A"), (2, "passage C")])

    hybrid = HybridRetriever(bm25=bm25_mock, contriever=contriever_mock, rrf_k=60)
    results = hybrid.retrieve("antibiotics for gonorrhea", k=1)

    assert results == ["passage A"], "doc ranked high by both retrievers should be top-1"


def test_hybrid_rrf_returns_k_passages():
    bm25_mock = _make_mock_retriever([(0, "p0"), (1, "p1"), (2, "p2")])
    contriever_mock = _make_mock_retriever([(2, "p2"), (0, "p0"), (1, "p1")])

    hybrid = HybridRetriever(bm25=bm25_mock, contriever=contriever_mock, rrf_k=60)
    results = hybrid.retrieve("query", k=2)

    assert len(results) == 2
    assert all(isinstance(p, str) for p in results)


def test_hybrid_rrf_merges_non_overlapping_candidates():
    """Passages only in one retriever should still appear in final results."""
    bm25_mock = _make_mock_retriever([(0, "only in bm25")])
    contriever_mock = _make_mock_retriever([(1, "only in contriever")])

    hybrid = HybridRetriever(bm25=bm25_mock, contriever=contriever_mock, rrf_k=60)
    results = hybrid.retrieve("query", k=2)

    assert len(results) == 2
    assert "only in bm25" in results
    assert "only in contriever" in results


# ---------------------------------------------------------------------------
# Test 5 — ContrieverRetriever skipped without cache
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPY_PATH.exists(), reason="Contriever embeddings not built yet — run ContrieverRetriever.load() first")
def test_contriever_retriever_returns_passages():
    from smart_trial.rag.retriever import BM25Retriever, ContrieverRetriever
    bm25 = BM25Retriever.load(cache_path=str(PKL_PATH))
    retriever = ContrieverRetriever.load(passages=bm25._passages, cache_path=str(NPY_PATH))
    results = retriever.retrieve("first-line antibiotics for gonorrhea", k=3)
    assert len(results) == 3
    assert all(isinstance(p, str) and len(p) > 0 for p in results)
