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
    scores, ids = index.search(query_vec, k)

    results = []
    for rank, i in enumerate(ids[0]):
        results.append(
            {
                "id": documents[i]["id"],
                "chunk_id": documents[i]["chunk_id"],
                "score": float(scores[0][rank]),
                "text": documents[i]["text"],
            }
        )

    return results
