from __future__ import annotations

import logging
import os

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Return the embedding model, initialising it on first call.

    Using a lazy singleton means:
    - Importing this module never loads 90 MB of weights.
    - Test suites that mock ``get_model`` pay zero model-load cost.
    - Gunicorn workers each call this once after forking, which is safe
      because ``SentenceTransformer`` is not fork-unsafe.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    vec = np.asarray(get_model().encode(text), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm