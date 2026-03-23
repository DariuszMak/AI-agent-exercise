from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np

from .embeddings import embed


def search(
    index: faiss.Index,
    documents: Sequence[dict[str, Any]],
    query: str,
    k: int = 3,
) -> list[dict[str, Any]]:
    query_vec = np.array([embed(query)], dtype="float32")
    scores, ids = index.search(query_vec, k)

    results = []
    for rank, i in enumerate(ids[0]):
        doc = documents[i]
        results.append(
            {
                "id": doc["id"],
                "chunk_id": doc["chunk_id"],
                "score": float(scores[0][rank]),
                "text": doc["text"],
                # surface the extra metadata so callers can debug retrieval
                "token_count": doc.get("token_count"),
                "char_start": doc.get("char_start"),
                "char_end": doc.get("char_end"),
            }
        )

    return results