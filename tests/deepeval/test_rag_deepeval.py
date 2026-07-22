from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deepeval import assert_evaluation

if TYPE_CHECKING:
    from flask.testing import FlaskClient

DOCUMENTS_PATH = Path("storage/documents/EN")

TEST_CASES = [
    ("What is Empire State Building?", ["Empire State Building", "skyscraper", "Manhattan"]),
    ("What is Jeddah Tower?", ["Jeddah Tower", "supertall", "Saudi Arabia"]),
    ("Compare Empire State Building and Jeddah Tower heights.", ["height", "comparison", "building"]),
]


@pytest.mark.deepeval
@pytest.mark.slow
@pytest.mark.parametrize(("question", "expected_keywords"), TEST_CASES)
def test_rag_accuracy_deepeval(rag_client: FlaskClient, question: str, expected_keywords: list[str]) -> None:
    resp = rag_client.post("/ask", json={"query": question})
    assert resp.status_code == 200, f"Query failed: {resp.get_json()}"
    data = resp.get_json()

    answer = data["answer"]
    contexts = [s["text"] for s in data["sources"]]

    assert_evaluation(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=f"Informacje dotyczące {' '.join(expected_keywords)}",
        model="gemma:2b",
    )
