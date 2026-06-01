from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import create_app

if TYPE_CHECKING:
    from flask.testing import FlaskClient


def _fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def patched_client(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance
        import src.rag.embeddings as emb
        emb._model = None
        app = create_app(docs)
    app.config["TESTING"] = True
    return app.test_client()


def test_build_index_with_no_documents_returns_warning(tmp_path: Path) -> None:
    empty_docs = tmp_path / "documents"
    empty_docs.mkdir()

    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance
        import src.rag.embeddings as emb
        emb._model = None
        app = create_app(empty_docs)

    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post("/index")
    assert response.status_code == 200
    data = response.get_json()
    assert data["indexed"] == 0
    assert "warning" in data


def test_build_index_with_documents(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "test.txt").write_text("KSeF is an invoicing system used in Poland.", encoding="utf-8")

    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance
        import src.rag.embeddings as emb
        emb._model = None
        app = create_app(docs)

    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post("/index")
    assert response.status_code == 200
    assert response.get_json()["indexed"] > 0


def test_query_without_body_returns_400(patched_client: FlaskClient) -> None:
    patched_client.post("/index")
    response = patched_client.post("/query", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_ask_without_body_returns_400(patched_client: FlaskClient) -> None:
    patched_client.post("/index")
    response = patched_client.post("/ask", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_ask_missing_query_key_returns_400(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.txt").write_text("Test content for index.", encoding="utf-8")

    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance
        import src.rag.embeddings as emb
        emb._model = None
        app = create_app(docs)

    app.config["TESTING"] = True
    client = app.test_client()
    client.post("/index")

    response = client.post("/ask", json={"wrong_key": "value"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid question"


def test_autoload_true_with_existing_files(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.txt").write_text("Test document for autoload.", encoding="utf-8")
    index_path = tmp_path / "index.faiss"
    docstore_path = tmp_path / "docs.json"

    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance
        import src.rag.embeddings as emb
        emb._model = None
        app = create_app(docs, index_path=index_path, docstore_path=docstore_path, autoload=False)
        app.config["TESTING"] = True
        client = app.test_client()
        client.post("/index")

    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance
        emb._model = None
        app2 = create_app(docs, index_path=index_path, docstore_path=docstore_path, autoload=True)
        app2.config["TESTING"] = True
        client2 = app2.test_client()
        response = client2.post("/query", json={"query": "test"})

    assert response.status_code == 200


def test_autoload_true_without_existing_files(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    missing_index = tmp_path / "no_index.faiss"
    missing_docstore = tmp_path / "no_docs.json"

    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance
        import src.rag.embeddings as emb
        emb._model = None
        app = create_app(docs, index_path=missing_index, docstore_path=missing_docstore, autoload=True)

    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 400