from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.rag.api.app import create_app

if TYPE_CHECKING:
    from pathlib import Path


def _fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


def _patch_embed() -> Any:
    mock_st = MagicMock()
    instance = MagicMock()
    instance.encode.side_effect = _fake_embed
    mock_st.return_value = instance
    return patch("src.rag.embeddings.SentenceTransformer", mock_st)


def test_index_persistence(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.txt").write_text("KSeF to system do faktur.", encoding="utf-8")

    index_path = tmp_path / "index.faiss"
    docstore_path = tmp_path / "documents.json"

    import src.rag.embeddings as emb

    with _patch_embed():
        emb._model = None
        app1 = create_app(
            documents_path=docs,
            index_path=index_path,
            docstore_path=docstore_path,
            autoload=False,
        )
        app1.config["TESTING"] = True
        client1 = app1.test_client()

        response = client1.post("/index")
        assert response.status_code == 200
        assert response.get_json()["indexed"] > 0

    assert index_path.exists()
    assert docstore_path.exists()

    with _patch_embed():
        emb._model = None
        app2 = create_app(
            documents_path=docs,
            index_path=index_path,
            docstore_path=docstore_path,
            autoload=True,
        )
        app2.config["TESTING"] = True
        client2 = app2.test_client()

        response = client2.post("/query", json={"query": "Czym jest KSeF?"})
        assert response.status_code == 200

    results = response.get_json()
    assert results is not None
    assert len(results) > 0
    assert "KSeF" in results[0]["text"]
