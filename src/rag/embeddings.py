from __future__ import annotations

import os

import numpy as np
import pytest
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

_MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:

    global _model
    if _model is None:
        logger.info("Loading embedding model %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    vec = np.asarray(get_model().encode(text), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == pytest.approx(0.0):
        return vec
    return vec / norm
