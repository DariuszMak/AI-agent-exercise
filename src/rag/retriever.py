from collections.abc import Sequence

import faiss  # type: ignore[import-untyped]
import numpy as np

from .embeddings import embed


def search(
    index: faiss.Index,
    documents: Sequence[dict[str, str]],
    query: str,
    k: int = 3,
) -> list[dict[str, str]]:
    query_vec = np.array([embed(query)], dtype="float32")
    _, ids = index.search(query_vec, k)
    return [documents[i] for i in ids[0]]
