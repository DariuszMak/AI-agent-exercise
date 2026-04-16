from functools import lru_cache
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self._client = OpenAI(
            api_key="ollama",
            base_url=base_url,
            timeout=timeout,
        )

    def get_model_name(self) -> str:
        return self.model

    def load_model(self) -> Any:
        return self._client

    @lru_cache(maxsize=256)
    def _cached_generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def generate(self, prompt: str, **_: Any) -> str:
        return self._cached_generate(prompt)

    async def a_generate(self, prompt: str, **_: Any) -> str:
        return self.generate(prompt)
