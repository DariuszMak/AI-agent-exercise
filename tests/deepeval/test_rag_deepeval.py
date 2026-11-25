from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
import requests
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

if TYPE_CHECKING:
    from flask.testing import FlaskClient


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "gemma:2b",
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
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for model '{self.model_name}'. "
                f"Check installed models with `ollama list`."
            )
        response.raise_for_status()
        return cast("str", response.json()["response"])

    async def a_generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=self._build_payload(prompt),
            )
            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned 404 for model '{self.model_name}'. "
                    f"Check installed models with `ollama list`."
                )
            response.raise_for_status()
            return cast("str", response.json()["response"])

    def get_model_name(self) -> str:
        return f"Ollama ({self.model_name})"


class SimpleOllamaMetric(BaseMetric):
    """Lekka metryka dla małych modeli (np. gemma:2b) ze skalą 0, 1, 2 -> 0.0, 0.5, 1.0."""

    def __init__(
        self,
        name: str,
        instruction: str,
        model: OllamaJudge,
        threshold: float = 0.5,
    ) -> None:
        self.name = name
        self.instruction = instruction
        self.model = model
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

    def _parse_response(self, raw: str) -> tuple[float, str]:
        try:
            data = json.loads(raw)
            rating = int(data.get("rating", 0))
            reason = str(data.get("reason", "")).strip()
        except (ValueError, TypeError, json.JSONDecodeError):
            match = re.search(r"\b([012])\b", raw)
            rating = int(match.group(1)) if match else 0
            reason = f"Fallback parse from raw output: {raw[:200]}"

        rating = max(0, min(2, rating))
        score_map = {0: 0.0, 1: 0.5, 2: 1.0}
        return score_map[rating], reason

    def _build_prompt(self, test_case: LLMTestCase) -> str:
        context_block = "\n".join(test_case.retrieval_context or [])
        return (
            "You are a strict evaluator. Reply ONLY with valid JSON: "
            '{"rating": 0 or 1 or 2, "reason": "short explanation"}.\n'
            "Rating scale:\n"
            "- 2: fully satisfies the criteria\n"
            "- 1: partially satisfies the criteria\n"
            "- 0: fails or contradicts the criteria\n\n"
            f"Criteria: {self.instruction}\n"
            f"Question: {test_case.input}\n"
            f"Expected output: {test_case.expected_output or ''}\n"
            f"Actual output: {test_case.actual_output or ''}\n"
            f"Retrieved context:\n{context_block}\n"
        )

    def measure(self, test_case: LLMTestCase) -> float:
        raw = self.model.generate(self._build_prompt(test_case))
        self.score, self.reason = self._parse_response(raw)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, _show_indicator: bool = True) -> float:
        raw = await self.model.a_generate(self._build_prompt(test_case))
        self.score, self.reason = self._parse_response(raw)
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return self.name


@pytest.fixture(scope="module")
def eval_model() -> OllamaJudge:
    judge = OllamaJudge(model_name="gemma:2b")
    try:
        resp = requests.get(f"{judge.base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        installed = [m["name"] for m in resp.json().get("models", [])]
    except requests.RequestException as exc:
        pytest.fail(f"Ollama is not reachable at {judge.base_url}: {exc}")

    if judge.model_name not in installed:
        matching = [m for m in installed if m.startswith("gemma:2b") or m.startswith("gemma2:2b")]
        if matching:
            judge.model_name = matching[0]
        else:
            pytest.fail(
                f"Model '{judge.model_name}' not found in Ollama. Installed: {installed}"
            )
    return judge


TEST_CASES = [
    (
        "What is the Empire State Building?",
        "The Empire State Building is a skyscraper in Manhattan.",
        "Empire State Building",
        "Manhattan",
    ),
    (
        "What is the Jeddah Tower?",
        "Jeddah Tower is a skyscraper in Saudi Arabia.",
        "Jeddah Tower",
        "Saudi Arabia",
    ),
]


@pytest.mark.deepeval
@pytest.mark.slow
@pytest.mark.parametrize(
    ("question", "ground_truth", "expected_entity", "expected_location"),
    TEST_CASES,
)
def test_rag_accuracy_and_grounding(
    rag_client: FlaskClient,
    eval_model: OllamaJudge,
    question: str,
    ground_truth: str,
    expected_entity: str,
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

    joined_context = "\n".join(contexts).lower()
    assert expected_entity.lower() in joined_context, (
        f"Retriever failure: '{expected_entity}' not found in retrieved contexts: {contexts}"
    )

    abstained = "does not provide any information" in answer.lower()
    assert not abstained, (
        f"Generator abstained despite '{expected_entity}' appearing in retrieved context.\n"
        f"Answer: {answer}\n"
        f"Retrieved contexts ({len(contexts)}): {contexts}"
    )

    assert expected_location.lower() in answer.lower(), (
        f"Generator failure: expected location '{expected_location}' not found in answer: {answer}"
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
        expected_output=ground_truth,
    )

    correctness_metric = SimpleOllamaMetric(
        name="Correctness",
        instruction=(
            "Actual output matches the core meaning and location of expected output "
            "without contradicting it."
        ),
        model=eval_model,
        threshold=0.5,
    )

    faithfulness_metric = SimpleOllamaMetric(
        name="Context Faithfulness",
        instruction=(
            "Actual output uses facts supported by retrieved context and does not "
            "invent unsupported facts."
        ),
        model=eval_model,
        threshold=0.5,
    )

    assert_test(test_case, [correctness_metric, faithfulness_metric])
