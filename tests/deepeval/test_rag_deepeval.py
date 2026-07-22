from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

if TYPE_CHECKING:
    from flask.testing import FlaskClient

DOCUMENTS_PATH = Path("storage/documents/EN")


class GemmaOllamaJudge(DeepEvalBaseLLM):
    """Lekki wrapper Ollama bez potrzeby instalowania langchain-ollama."""

    def __init__(self, model_name: str = "gemma2:2b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Ollama ({self.model_name})"


eval_model = GemmaOllamaJudge(model_name="gemma2:2b")

TEST_CASES = [
    (
        "What is Empire State Building?",
        "The Empire State Building is a famous skyscraper located in Manhattan.",
    ),
    (
        "What is Jeddah Tower?",
        "Jeddah Tower is a planned supertall skyscraper in Saudi Arabia.",
    ),
]


@pytest.mark.deepeval
@pytest.mark.slow
@pytest.mark.parametrize(("question", "ground_truth"), TEST_CASES)
def test_rag_accuracy_deepeval(rag_client: FlaskClient, question: str, ground_truth: str) -> None:
    resp = rag_client.post("/ask", json={"query": question})
    assert resp.status_code == 200, f"Query failed: {resp.get_json()}"

    data = resp.get_json()
    answer = data["answer"]
    contexts = [s["text"] for s in data["sources"]]

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=ground_truth,
    )

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output covers key factual details present in the expected output.",
        evaluation_steps=[
            "Check if key facts from expected_output are present in actual_output.",
            "Penalize missing key facts or factual contradictions.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=eval_model,
    )

    assert_test(test_case, [correctness_metric])
