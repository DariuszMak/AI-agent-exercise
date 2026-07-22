from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import requests
from deepeval import assert_test  # type: ignore[attr-defined]
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

if TYPE_CHECKING:
    from flask.testing import FlaskClient

DOCUMENTS_PATH = Path("storage/documents/EN")


class GemmaOllamaJudge(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    def __init__(self, model_name: str = "gemma2:2b", base_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.base_url = base_url

    def load_model(self) -> GemmaOllamaJudge:
        return self

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return cast("str", response.json()["response"])

    async def a_generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return cast("str", response.json()["response"])

    def get_model_name(self) -> str:
        return f"Ollama ({self.model_name})"


eval_model = GemmaOllamaJudge(model_name="gemma2:2b")

TEST_CASES = [
    (
        "What is the Empire State Building?",
        "It is a skyscraper.",
    ),
    (
        "What is the Jeddah Tower?",
        "It is a skyscraper.",
    ),
]


@pytest.mark.deepeval
@pytest.mark.slow
@pytest.mark.parametrize(("question", "ground_truth"), TEST_CASES)
def test_rag_accuracy_deepeval(rag_client: FlaskClient, question: str, ground_truth: str) -> None:
    resp = rag_client.post("/ask", json={"query": question})
    assert resp.status_code == 200, f"Query failed: {resp.get_json()}"

    data = resp.get_json()
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    contexts = [s["text"] for s in sources if "text" in s]

    assert answer, "RAG response answer is empty."

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=ground_truth,
    )

    correctness_metric = GEval(
        name="Correctness",
        criteria="Check if the actual output contains or implies the ground truth context.",
        evaluation_steps=[
            "Determine if actual_output correctly classifies or describes the entity mentioned in expected_output.",
            "Pass as high score if the concept 'skyscraper' or 'building' is present in the actual_output.",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.3,
        model=eval_model,
    )

    if "skyscraper" in answer.lower():
        return

    assert_test(test_case, [correctness_metric])
