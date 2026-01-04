import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> np.ndarray:
    vec = _model.encode(text)
    return vec / np.linalg.norm(vec)
