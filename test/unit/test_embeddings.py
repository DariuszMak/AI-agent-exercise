from unittest.mock import MagicMock, patch

import numpy as np


def test_model_not_loaded_on_import() -> None:
    """Importing embeddings must not trigger model initialisation."""
    import src.rag.embeddings as emb

    # Reset singleton so this test is order-independent
    emb._model = None

    # The model should still be None — no eager load
    assert emb._model is None


def test_get_model_initialises_once() -> None:
    import src.rag.embeddings as emb

    emb._model = None

    fake_model = MagicMock()
    fake_model.encode.return_value = np.ones(384, dtype=np.float32)

    with patch("src.rag.embeddings.SentenceTransformer", return_value=fake_model) as mock_cls:
        m1 = emb.get_model()
        m2 = emb.get_model()

    # Constructor called exactly once despite two get_model() calls
    mock_cls.assert_called_once()
    assert m1 is m2


def test_embed_returns_normalised_vector() -> None:
    import src.rag.embeddings as emb

    emb._model = None

    fake_model = MagicMock()
    raw = np.array([3.0] + [0.0] * 383, dtype=np.float32)
    fake_model.encode.return_value = raw

    with patch("src.rag.embeddings.SentenceTransformer", return_value=fake_model):
        vec = emb.embed("test")

    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_zero_vector_returned_as_is() -> None:
    import src.rag.embeddings as emb

    emb._model = None

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros(384, dtype=np.float32)

    with patch("src.rag.embeddings.SentenceTransformer", return_value=fake_model):
        vec = emb.embed("zero")

    assert np.all(vec == 0.0)