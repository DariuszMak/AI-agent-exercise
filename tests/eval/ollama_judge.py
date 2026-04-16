from __future__ import annotations

from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI


class OllamaJudge(DeepEvalBaseLLM):
    """Thin wrapper so DeepEval uses local Ollama as its evaluator/judge."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434/v1") -> None:
        self.model = model
        self._client = OpenAI(api_key="ollama", base_url=base_url)

    def get_model_name(self) -> str:
        return self.model

    def load_model(self) -> Any:
        return self._client

    def generate(self, prompt: str, **_: Any) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""

    async def a_generate(self, prompt: str, **_: Any) -> str:
        return self.generate(prompt)