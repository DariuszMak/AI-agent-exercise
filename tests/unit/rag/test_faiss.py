from unittest.mock import patch

import numpy as np
import pytest

from src.rag.index import IndexStore
from src.rag.retriever import search


def _fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def populated_store() -> IndexStore:
    documents = [
        {
            "id": "1",
            "chunk_id": "0",
            "text": "Empire State Building is a skyscraper in Manhattan.",
            "token_count": 4,
            "char_start": 0,
            "char_end": 21,
        },
        {"id": "2", "chunk_id": "0", "text": "Kot siedzi na dachu", "token_count": 4, "char_start": 0, "char_end": 19},
    ]
    with patch("src.rag.index.embed", side_effect=_fake_embed):
        store = IndexStore()
        store.build(documents)
    return store


def test_faiss_retrieval_returns_results(populated_store: IndexStore) -> None:
    assert populated_store.index is not None
    with patch("src.rag.retriever.embed", side_effect=_fake_embed):
        results = search(populated_store.index, populated_store.documents, "What is Empire State Building?", k=1)
    assert len(results) == 1


def test_retrieval_result_has_metadata(populated_store: IndexStore) -> None:
    assert populated_store.index is not None
    with patch("src.rag.retriever.embed", side_effect=_fake_embed):
        results = search(populated_store.index, populated_store.documents, "faktura", k=2)
    for r in results:
        assert "score" in r
        assert "token_count" in r
        assert "char_start" in r
        assert "char_end" in r


def test_retrieval_score_between_minus_one_and_one(populated_store: IndexStore) -> None:
    assert populated_store.index is not None
    with patch("src.rag.retriever.embed", side_effect=_fake_embed):
        results = search(populated_store.index, populated_store.documents, "test", k=2)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0
