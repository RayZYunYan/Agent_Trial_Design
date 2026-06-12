import pytest
from pathlib import Path

from smart_trial.core.orchestrator import TrialOrchestrator


# ---------------------------------------------------------------------------
# Test 1 — BM25Retriever loads and returns relevant passages
# ---------------------------------------------------------------------------

PKL_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_index.pkl"


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
