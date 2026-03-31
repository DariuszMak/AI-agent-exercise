from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import create_app

if TYPE_CHECKING:
    from pathlib import Path

    from flask.testing import FlaskClient


def _fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def client(tmp_path: Path) -> FlaskClient:
    docs = tmp_path / "documents"
    docs.mkdir()
    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = lambda t: _fake_embed(t)
        mock_st.return_value = instance
        import src.rag.embeddings as emb

        emb._model = None
        app = create_app(docs)
    app.config["TESTING"] = True
    return app.test_client()
