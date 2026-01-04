import numpy as np

from src.rag.embeddings import embed


def test_embedding_normalized() -> None:
    vec = embed("KSeF to system fakturowania")
    norm = np.linalg.norm(vec)

    assert vec.shape == (384,)
    assert abs(norm - 1.0) < 1e-5
