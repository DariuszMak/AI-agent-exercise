from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from src.app import create_app
from tests.eval.ollama_judge import OllamaJudge

DOCUMENTS_PATH_STR = "storage/documents/EN"
THRESHOLD = 0.5

judge = OllamaJudge(model="llama3")  # swap model name if you use llama3.2 etc.


@pytest.fixture(scope="module")
def rag_client():
    from pathlib import Path

    app = create_app(documents_path=Path(DOCUMENTS_PATH_STR), autoload=True)
    app.config["TESTING"] = True
    client = app.test_client()
    r = client.post("/index")
    assert r.status_code == 200, f"Index build failed: {r.get_json()}"
    return client


def ask(client, question: str) -> tuple[str, list[str]]:
    resp = client.post("/ask", json={"query": question})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    return data["answer"], [s["text"] for s in data["sources"]]


TEST_CASES = [
    {
        "input": "What is KSeF?",
        "expected_output": "KSeF is the National e-Invoice System in Poland.",
    },
    {
        "input": "What is Camunda?",
        "expected_output": "Camunda is a process automation platform.",
    },
    {
        "input": "What is Devapo?",
        "expected_output": "Devapo is a software company.",
    },
]


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["input"] for c in TEST_CASES])
@pytest.mark.slow
def test_rag_accuracy(rag_client, case: dict) -> None:
    answer, contexts = ask(rag_client, case["input"])

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=answer,
        expected_output=case["expected_output"],
        retrieval_context=contexts,
    )

    assert_test(
        test_case,
        metrics=[
            AnswerRelevancyMetric(threshold=THRESHOLD, model=judge),
            FaithfulnessMetric(threshold=THRESHOLD, model=judge),
            # ContextualPrecisionMetric(threshold=THRESHOLD, model=judge),
            # ContextualRecallMetric(threshold=THRESHOLD, model=judge),
        ],
    )