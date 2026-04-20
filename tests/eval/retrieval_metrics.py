"""
Deterministic retrieval metrics — no LLM required.

Metrics
-------
hit_rate  : fraction of questions where the expected doc appears in top-k results
mrr       : Mean Reciprocal Rank — rewards finding the right doc earlier in the list
ndcg      : Normalised Discounted Cumulative Gain — position-aware quality score

These are fast, stable, and safe to run in CI on every commit.
They catch regressions in chunking, embedding, or retrieval logic
without any external service dependency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask.testing import FlaskClient


def _get_results(client: FlaskClient, question: str, k: int) -> list[dict[str, Any]]:
    resp = client.post("/query", json={"query": question})
    assert resp.status_code == 200, f"Query failed: {resp.get_json()}"
    results: list[dict[str, Any]] = resp.get_json()
    return results[:k]


def hit_rate(
    client: FlaskClient,
    qa_pairs: list[tuple[str, str]],
    k: int = 3,
) -> float:
    """
    Fraction of questions where *expected_doc_id* appears in the top-k results.

    Parameters
    ----------
    client:
        Flask test client with a built index.
    qa_pairs:
        List of (question, expected_doc_id) tuples.
    k:
        How many retrieved chunks to consider.

    Returns
    -------
    float in [0, 1]
    """
    if not qa_pairs:
        return 0.0

    hits = sum(any(r["id"] == expected for r in _get_results(client, q, k)) for q, expected in qa_pairs)
    return hits / len(qa_pairs)


def mrr(
    client: FlaskClient,
    qa_pairs: list[tuple[str, str]],
    k: int = 5,
) -> float:
    """
    Mean Reciprocal Rank.

    For each question, finds the rank of the first result whose id matches
    *expected_doc_id* and accumulates 1/rank.  Returns the mean across all
    questions.  A result not found in top-k contributes 0.

    Parameters
    ----------
    client:
        Flask test client with a built index.
    qa_pairs:
        List of (question, expected_doc_id) tuples.
    k:
        Maximum rank to consider.

    Returns
    -------
    float in [0, 1]
    """
    if not qa_pairs:
        return 0.0

    total = 0.0
    for question, expected in qa_pairs:
        results = _get_results(client, question, k)
        for rank, result in enumerate(results, start=1):
            if result["id"] == expected:
                total += 1.0 / rank
                break
    return total / len(qa_pairs)


def ndcg(
    client: FlaskClient,
    qa_pairs: list[tuple[str, str]],
    k: int = 5,
) -> float:
    """
    Normalised Discounted Cumulative Gain @ k.

    Treats finding the expected document at rank *r* as a gain of
    ``1 / log2(r + 1)``.  Normalises by the ideal DCG (expected doc at rank 1).

    Parameters
    ----------
    client:
        Flask test client with a built index.
    qa_pairs:
        List of (question, expected_doc_id) tuples.
    k:
        Maximum rank to consider.

    Returns
    -------
    float in [0, 1]
    """
    if not qa_pairs:
        return 0.0

    ideal_dcg = 1.0 / math.log2(2)  # rank-1 hit

    total = 0.0
    for question, expected in qa_pairs:
        results = _get_results(client, question, k)
        for rank, result in enumerate(results, start=1):
            if result["id"] == expected:
                total += (1.0 / math.log2(rank + 1)) / ideal_dcg
                break
    return total / len(qa_pairs)
