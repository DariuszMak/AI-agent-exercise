from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.rag.api.app import create_app
from tests.eval.ollama_judge import score_answer_relevancy, score_context_relevancy, score_faithfulness

if TYPE_CHECKING:
    from flask.testing import FlaskClient

DOCUMENTS_PATH = Path("storage/documents/EN")


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


TEST_CASES = [
    "What is KSeF?",
    "What is Camunda?",
    "What is Devapo?",
]


@pytest.mark.parametrize("question", TEST_CASES)
@pytest.mark.slow
def test_faithfulness(rag_client: FlaskClient, question: str) -> None:
    answer, contexts = ask(rag_client, question)
    result = score_faithfulness(answer, contexts)
    assert result["score"] == 1, f"Faithfulness FAILED — {result['reason']}"


@pytest.mark.parametrize("question", TEST_CASES)
@pytest.mark.slow
def test_answer_relevancy(rag_client: FlaskClient, question: str) -> None:
    answer, _contexts = ask(rag_client, question)
    result = score_answer_relevancy(question, answer)
    assert result["score"] == 1, f"Answer relevancy FAILED — {result['reason']}"


@pytest.mark.parametrize("question", TEST_CASES)
@pytest.mark.slow
def test_context_relevancy(rag_client: FlaskClient, question: str) -> None:
    _answer, contexts = ask(rag_client, question)
    result = score_context_relevancy(question, contexts)
    assert result["score"] == 1, f"Context relevancy FAILED — {result['reason']}"
