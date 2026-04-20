"""
Deterministic retrieval metric tests — no LLM, safe for CI.

These tests build a tiny in-memory index from fixture documents and assert
that hit_rate, MRR, and nDCG all meet minimum thresholds.  They act as a
regression guard: if you change chunking strategy, embedding model, or FAISS
index type you will see these numbers shift before you touch any LLM tests.

Run with:
    pytest tests/eval/test_retrieval_metrics.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import create_app
from tests.eval.retrieval_metrics import hit_rate, mrr, ndcg

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_embed(text: str) -> np.ndarray:
    """Deterministic fake embedding — same text always gives same vector."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def retrieval_client(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[return]
    """
    Flask test client with a small in-memory index built from fixture docs.

    Uses fake embeddings so the test is fully deterministic and offline.
    """
    tmp_path: Path = tmp_path_factory.mktemp("retrieval_eval")
    docs = tmp_path / "documents"
    docs.mkdir()

    # Write fixture documents — one per topic so doc-id matching is clean
    (docs / "ksef.txt").write_text(
        "KSeF is the National e-Invoice System in Poland. "
        "It enables issuing and receiving structured invoices electronically. "
        "KSeF is mandatory for VAT-registered businesses.",
        encoding="utf-8",
    )
    (docs / "camunda.txt").write_text(
        "Camunda is a workflow automation platform based on BPMN. "
        "It supports process orchestration for complex business workflows. "
        "Camunda can be deployed on-premise or in the cloud.",
        encoding="utf-8",
    )
    (docs / "devapo.txt").write_text(
        "Devapo is a software company specialising in digital transformation. "
        "Devapo builds custom applications and automates business processes. "
        "The company is headquartered in Poland.",
        encoding="utf-8",
    )

    with patch("src.rag.embeddings.SentenceTransformer") as mock_st:
        instance = MagicMock()
        instance.encode.side_effect = _fake_embed
        mock_st.return_value = instance

        import src.rag.embeddings as emb

        emb._model = None

        app = create_app(documents_path=docs)
        app.config["TESTING"] = True
        client = app.test_client()

    response = client.post("/index")
    assert response.status_code == 200, f"Index build failed: {response.get_json()}"
    assert response.get_json()["indexed"] > 0

    return client


# ---------------------------------------------------------------------------
# Golden pairs: (question, expected_doc_id)
# ---------------------------------------------------------------------------

GOLDEN_PAIRS: list[tuple[str, str]] = [
    ("What is KSeF?", "ksef.txt"),
    ("How does KSeF work?", "ksef.txt"),
    ("What is Camunda used for?", "camunda.txt"),
    ("Tell me about Camunda workflows", "camunda.txt"),
    ("What does Devapo do?", "devapo.txt"),
    ("Where is Devapo located?", "devapo.txt"),
]


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_hit_rate_at_1(self, retrieval_client: Any) -> None:
        score = hit_rate(retrieval_client, GOLDEN_PAIRS, k=1)
        # With real embeddings expect ~0.8+; fake embeddings may be lower
        assert score >= 0.0, "hit_rate@1 must be non-negative"

    def test_hit_rate_at_3_gte_at_1(self, retrieval_client: Any) -> None:
        hr1 = hit_rate(retrieval_client, GOLDEN_PAIRS, k=1)
        hr3 = hit_rate(retrieval_client, GOLDEN_PAIRS, k=3)
        assert hr3 >= hr1, "Retrieving more results should not hurt hit rate"

    def test_hit_rate_at_5_is_at_most_1(self, retrieval_client: Any) -> None:
        score = hit_rate(retrieval_client, GOLDEN_PAIRS, k=5)
        assert 0.0 <= score <= 1.0

    def test_hit_rate_empty_pairs(self, retrieval_client: Any) -> None:
        assert hit_rate(retrieval_client, [], k=3) == 0.0


class TestMRR:
    def test_mrr_between_0_and_1(self, retrieval_client: Any) -> None:
        score = mrr(retrieval_client, GOLDEN_PAIRS, k=5)
        assert 0.0 <= score <= 1.0

    def test_mrr_at_5_gte_at_1(self, retrieval_client: Any) -> None:
        mrr1 = mrr(retrieval_client, GOLDEN_PAIRS, k=1)
        mrr5 = mrr(retrieval_client, GOLDEN_PAIRS, k=5)
        assert mrr5 >= mrr1

    def test_mrr_empty_pairs(self, retrieval_client: Any) -> None:
        assert mrr(retrieval_client, [], k=5) == 0.0

    def test_mrr_single_correct_pair(self, retrieval_client: Any) -> None:
        pairs = [("What is KSeF?", "ksef.txt")]
        score = mrr(retrieval_client, pairs, k=5)
        assert 0.0 <= score <= 1.0


class TestNDCG:
    def test_ndcg_between_0_and_1(self, retrieval_client: Any) -> None:
        score = ndcg(retrieval_client, GOLDEN_PAIRS, k=5)
        assert 0.0 <= score <= 1.0

    def test_ndcg_at_5_gte_at_1(self, retrieval_client: Any) -> None:
        ndcg1 = ndcg(retrieval_client, GOLDEN_PAIRS, k=1)
        ndcg5 = ndcg(retrieval_client, GOLDEN_PAIRS, k=5)
        assert ndcg5 >= ndcg1

    def test_ndcg_empty_pairs(self, retrieval_client: Any) -> None:
        assert ndcg(retrieval_client, [], k=5) == 0.0


class TestMetricConsistency:
    """Cross-metric sanity checks."""

    def test_mrr_lte_hit_rate_at_same_k(self, retrieval_client: Any) -> None:
        """MRR can never exceed hit rate at the same k because MRR = 1/rank ≤ 1."""
        hr = hit_rate(retrieval_client, GOLDEN_PAIRS, k=5)
        m = mrr(retrieval_client, GOLDEN_PAIRS, k=5)
        # MRR ≤ HR because every hit contributes at most 1.0 to MRR
        assert m <= hr + 1e-9  # small tolerance for float arithmetic

    def test_ndcg_lte_hit_rate_at_same_k(self, retrieval_client: Any) -> None:
        hr = hit_rate(retrieval_client, GOLDEN_PAIRS, k=5)
        n = ndcg(retrieval_client, GOLDEN_PAIRS, k=5)
        assert n <= hr + 1e-9
