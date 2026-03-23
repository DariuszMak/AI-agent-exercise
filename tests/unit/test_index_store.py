from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.rag.index import IndexStore


def _fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def sample_docs() -> list[dict[str, Any]]:
    return [
        {
            "id": "a.txt",
            "chunk_id": "0",
            "text": "KSeF is an invoicing system.",
            "token_count": 6,
            "char_start": 0,
            "char_end": 28,
        },
        {
            "id": "a.txt",
            "chunk_id": "1",
            "text": "It is used in Poland.",
            "token_count": 5,
            "char_start": 29,
            "char_end": 50,
        },
    ]


def test_store_not_ready_when_empty() -> None:
    store = IndexStore()
    assert not store.ready


def test_store_ready_after_build(sample_docs: list[dict[str, Any]]) -> None:
    with patch("src.rag.index.embed", side_effect=_fake_embed):
        store = IndexStore()
        store.build(sample_docs)
    assert store.ready


def test_build_populates_documents(sample_docs: list[dict[str, Any]]) -> None:
    with patch("src.rag.index.embed", side_effect=_fake_embed):
        store = IndexStore()
        store.build(sample_docs)
    assert len(store.documents) == len(sample_docs)


def test_save_and_load_roundtrip(tmp_path: Path, sample_docs: list[dict[str, Any]]) -> None:
    index_path = tmp_path / "index.faiss"
    docstore_path = tmp_path / "docs.json"

    with patch("src.rag.index.embed", side_effect=_fake_embed):
        store = IndexStore()
        store.build(sample_docs)
        store.save(index_path, docstore_path)

    assert index_path.exists()
    assert docstore_path.exists()

    loaded = IndexStore.load(index_path, docstore_path)
    assert loaded.ready
    assert len(loaded.documents) == len(sample_docs)
    assert loaded.documents[0]["id"] == "a.txt"


def test_save_raises_when_empty(tmp_path: Path) -> None:
    store = IndexStore()
    with pytest.raises(RuntimeError, match="Cannot save an empty index"):
        store.save(tmp_path / "i.faiss", tmp_path / "d.json")


def test_multiple_workers_load_same_files(tmp_path: Path, sample_docs: list[dict[str, Any]]) -> None:
    """Simulate two workers independently loading the same persisted index."""
    index_path = tmp_path / "index.faiss"
    docstore_path = tmp_path / "docs.json"

    with patch("src.rag.index.embed", side_effect=_fake_embed):
        builder = IndexStore()
        builder.build(sample_docs)
        builder.save(index_path, docstore_path)

    worker1 = IndexStore.load(index_path, docstore_path)
    worker2 = IndexStore.load(index_path, docstore_path)

    assert worker1.ready
    assert worker2.ready
    assert len(worker1.documents) == len(worker2.documents)