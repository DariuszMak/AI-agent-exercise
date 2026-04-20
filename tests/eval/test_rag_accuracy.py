"""
RAG accuracy tests — LLM-dependent, marked @pytest.mark.slow.

Two evaluation paths are available:
1. ollama_judge  — uses your local Ollama model, binary/float scores
2. ragas         — uses RAGAS library for continuous 0-1 metrics (optional)

Run only the fast tests:
    pytest tests/eval/test_rag_accuracy.py -v -m "not slow"

Run all including LLM-dependent tests:
    pytest tests/eval/test_rag_accuracy.py -v -m slow

Run only RAGAS tests:
    pytest tests/eval/test_rag_accuracy.py -v -m ragas
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.app import create_app
from tests.eval.golden_dataset import GOLDEN_DATASET
from tests.eval.ollama_judge import (
    score_answer_relevancy,
    score_completeness,
    score_context_relevancy,
    score_faithfulness,
)

if TYPE_CHECKING:
    from flask.testing import FlaskClient

DOCUMENTS_PATH = Path("storage/documents/EN")

# ---------------------------------------------------------------------------
# Thresholds — tune these as your system improves
# ---------------------------------------------------------------------------
FAITHFULNESS_MIN = 0.7
ANSWER_RELEVANCY_MIN = 0.7
CONTEXT_RELEVANCY_MIN = 0.6
COMPLETENESS_MIN = 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rag_client() -> FlaskClient:
    app = create_app(documents_path=DOCUMENTS_PATH, autoload=True)
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.post("/index")
    assert r.status_code == 200, f"Index build failed: {r.get_json()}"
    return client


def ask(client: FlaskClient, question: str) -> tuple[str, list[str]]:
    resp = client.post("/ask", json={"query": question})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    return data["answer"], [s["text"] for s in data["sources"]]


# ---------------------------------------------------------------------------
# Parametrised test cases derived from the golden dataset
# ---------------------------------------------------------------------------

QUESTIONS = [entry["question"] for entry in GOLDEN_DATASET]
GROUND_TRUTHS = {entry["question"]: entry["ground_truth"] for entry in GOLDEN_DATASET}


# ---------------------------------------------------------------------------
# Faithfulness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.slow
def test_faithfulness(rag_client: FlaskClient, question: str) -> None:
    """Every claim in the answer must be grounded in the retrieved context."""
    answer, contexts = ask(rag_client, question)
    result = score_faithfulness(answer, contexts)
    assert result["score"] >= FAITHFULNESS_MIN, f"Faithfulness too low ({result['score']:.2f}) — {result['reason']}"


# ---------------------------------------------------------------------------
# Answer relevancy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.slow
def test_answer_relevancy(rag_client: FlaskClient, question: str) -> None:
    """The answer must directly address the question."""
    answer, _contexts = ask(rag_client, question)
    result = score_answer_relevancy(question, answer)
    assert result["score"] >= ANSWER_RELEVANCY_MIN, (
        f"Answer relevancy too low ({result['score']:.2f}) — {result['reason']}"
    )


# ---------------------------------------------------------------------------
# Context relevancy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.slow
def test_context_relevancy(rag_client: FlaskClient, question: str) -> None:
    """The retrieved context must be useful for answering the question."""
    _answer, contexts = ask(rag_client, question)
    result = score_context_relevancy(question, contexts)
    assert result["score"] >= CONTEXT_RELEVANCY_MIN, (
        f"Context relevancy too low ({result['score']:.2f}) — {result['reason']}"
    )


# ---------------------------------------------------------------------------
# Completeness tests (requires ground truth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.slow
def test_completeness(rag_client: FlaskClient, question: str) -> None:
    """The answer should cover the key facts from the ground-truth reference."""
    answer, _contexts = ask(rag_client, question)
    ground_truth = GROUND_TRUTHS[question]
    result = score_completeness(question, answer, ground_truth)
    assert result["score"] >= COMPLETENESS_MIN, f"Completeness too low ({result['score']:.2f}) — {result['reason']}"


# ---------------------------------------------------------------------------
# RAGAS batch evaluation (optional — requires ragas + langchain-openai)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.ragas
def test_ragas_aggregate_scores(rag_client: FlaskClient) -> None:
    """
    Run RAGAS over the full golden dataset and assert aggregate thresholds.

    Skip this test if ragas is not installed:
        pytest -m "not ragas"
    """
    try:
        from tests.eval.ragas_eval import run_ragas_eval
    except ImportError:
        pytest.skip("ragas not installed")

    questions = QUESTIONS
    ground_truths = [GROUND_TRUTHS[q] for q in questions]

    results = run_ragas_eval(
        rag_client,
        questions,
        ground_truths=ground_truths,
        include_recall=True,
    )

    df = results.to_pandas()

    assert df["faithfulness"].mean() >= FAITHFULNESS_MIN, (
        f"Mean RAGAS faithfulness {df['faithfulness'].mean():.3f} below threshold"
    )
    assert df["answer_relevancy"].mean() >= ANSWER_RELEVANCY_MIN, (
        f"Mean RAGAS answer_relevancy {df['answer_relevancy'].mean():.3f} below threshold"
    )
