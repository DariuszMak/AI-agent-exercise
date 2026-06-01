from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.agentic_retriever import AgenticRetriever
from src.rag.index import IndexStore


def _fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_docs(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "id": f"doc_{i}.txt",
            "chunk_id": str(i),
            "text": f"Document number {i} with enough text to be meaningful.",
            "token_count": 10,
            "char_start": 0,
            "char_end": 50,
        }
        for i in range(n)
    ]


@pytest.fixture()
def ready_store() -> IndexStore:
    docs = _make_docs()
    with patch("src.rag.index.embed", side_effect=_fake_embed):
        store = IndexStore()
        store.build(docs)
    return store


class TestAgenticRetrieverFromIndexStore:
    def test_from_ready_store(self, ready_store: IndexStore) -> None:
        retriever = AgenticRetriever.from_index_store(ready_store)
        assert isinstance(retriever, AgenticRetriever)

    def test_from_empty_store_raises(self) -> None:
        empty_store = IndexStore()
        with pytest.raises(ValueError, match="IndexStore nie jest gotowy"):
            AgenticRetriever.from_index_store(empty_store)

    def test_from_store_with_no_documents_raises(self) -> None:
        store = MagicMock()
        store.ready = False

        with pytest.raises(ValueError, match="IndexStore nie jest gotowy"):
            AgenticRetriever.from_index_store(store)


class TestAgenticRetrieverSearch:
    def test_search_returns_list(self, ready_store: IndexStore) -> None:
        retriever = AgenticRetriever.from_index_store(ready_store)
        with patch("src.rag.retriever.embed", side_effect=_fake_embed):
            results = retriever.search("test query", k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_search_results_have_required_keys(self, ready_store: IndexStore) -> None:
        retriever = AgenticRetriever.from_index_store(ready_store)
        with patch("src.rag.retriever.embed", side_effect=_fake_embed):
            results = retriever.search("document query")
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "id" in r

    def test_search_respects_k_param(self, ready_store: IndexStore) -> None:
        retriever = AgenticRetriever.from_index_store(ready_store)
        with patch("src.rag.retriever.embed", side_effect=_fake_embed):
            results_k1 = retriever.search("query", k=1)
            results_k3 = retriever.search("query", k=3)
        assert len(results_k1) == 1
        assert len(results_k3) == 3

    def test_search_uses_faiss_search(self, ready_store: IndexStore) -> None:
        retriever = AgenticRetriever.from_index_store(ready_store)
        with patch("src.rag.retriever.search") as mock_search:
            mock_search.return_value = [{"id": "x", "score": 0.9, "text": "t", "chunk_id": "0"}]
            results = retriever.search("q", k=1)
        mock_search.assert_called_once()
        assert results[0]["id"] == "x"
