import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> np.ndarray:
    vec = np.asarray(_model.encode(text), dtype=np.float32)
    norm = np.linalg.norm(vec)

    if norm == 0.0:
        return vec

    return vec / norm
