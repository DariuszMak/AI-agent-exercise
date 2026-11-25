from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
import requests
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

if TYPE_CHECKING:
    from flask.testing import FlaskClient


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "gemma2:9b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url

    def load_model(self) -> OllamaJudge:
        return self

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=self._build_payload(prompt),
            timeout=60,
        )
        response.raise_for_status()
        return cast("str", response.json()["response"])

    async def a_generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=self._build_payload(prompt),
            )
            response.raise_for_status()
            return cast("str", response.json()["response"])

    def get_model_name(self) -> str:
        return f"Ollama ({self.model_name})"


@pytest.fixture(scope="module")
def eval_model() -> OllamaJudge:
    return OllamaJudge(model_name="gemma2:9b")


TEST_CASES = [
    (
        "What is the Empire State Building?",
        "The Empire State Building is a skyscraper in Manhattan.",
        "Manhattan",
    ),
    (
        "What is the Jeddah Tower?",
        "Jeddah Tower is a skyscraper in Saudi Arabia.",
        "Saudi Arabia",
    ),
]


@pytest.mark.deepeval
@pytest.mark.slow
@pytest.mark.parametrize(("question", "ground_truth", "expected_location"), TEST_CASES)
def test_rag_accuracy_and_grounding(
    rag_client: FlaskClient,
    eval_model: OllamaJudge,
    question: str,
    ground_truth: str,
    expected_location: str,
) -> None:
    resp = rag_client.post("/ask", json={"query": question})
    assert resp.status_code == 200, f"Query failed: {resp.get_json()}"

    data = resp.get_json() or {}
    answer: str = data.get("answer", "").strip()
    sources: list[dict[str, Any]] = data.get("sources", [])
    contexts = [s["text"].strip() for s in sources if isinstance(s, dict) and s.get("text")]

    assert answer, "RAG response answer is empty."
    assert contexts, "RAG pipeline returned no retrieval contexts."
    assert expected_location.lower() in answer.lower(), (
        f"Expected location '{expected_location}' not found in answer: {answer}"
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=ground_truth,
    )

    correctness_metric = GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether actual_output conveys the same core facts as expected_output "
            "without contradicting it."
        ),
        evaluation_steps=[
            "Check if actual_output identifies the entity from input as a skyscraper or building.",
            "Verify that the geographic location stated in expected_output matches actual_output.",
            "Penalize contradictions or missing key facts from expected_output.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.6,
        model=eval_model,
    )

    faithfulness_metric = GEval(
        name="Context Faithfulness",
        criteria=(
            "Evaluate whether all factual claims in actual_output are directly supported "
            "by retrieval_context."
        ),
        evaluation_steps=[
            "Extract the factual claims made in actual_output.",
            "Verify whether each claim is supported by the passages in retrieval_context.",
            "Assign a low score if actual_output introduces external or unsupported facts.",
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.7,
        model=eval_model,
    )

    assert_test(test_case, [correctness_metric, faithfulness_metric])
