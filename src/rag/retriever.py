import numpy as np

from .embeddings import embed


def search(index, documents, query: str, k: int = 3):
    query_vec = np.array([embed(query)], dtype="float32")
    _, ids = index.search(query_vec, k)
    return [documents[i] for i in ids[0]]
