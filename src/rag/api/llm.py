from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

import requests
import structlog

logger = structlog.get_logger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma:2b"


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    done: bool


class OllamaAdapter:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        raw = self._post("/api/chat", payload)

        content: str = raw.get("message", {}).get("content", "")

        return LLMResponse(
            content=content.strip(),
            model=raw.get("model", self._model),
            done=raw.get("done", True),
        )

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    def complete_json(self, prompt: str) -> Any:
        json_prompt = (
            f"{prompt}\n\nOdpowiedz WYŁĄCZNIE poprawnym JSON-em, bez żadnego dodatkowego tekstu ani bloków kodu."
        )

        response = self.complete(
            json_prompt,
            temperature=0.0,
        )

        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLM zwrócił niepoprawny JSON: %s",
                raw[:200],
            )
            raise ValueError(f"Niepoprawny JSON od LLM: {raw[:200]}") from exc

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Nieobsługiwany schemat URL: {parsed.scheme}")

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()

            data = response.json()

            if not isinstance(data, dict):
                raise TypeError(f"Expected dict from Ollama, got {type(data).__name__}")

            return cast("dict[str, Any]", data)

        except requests.RequestException as exc:
            logger.exception(
                "Błąd połączenia z Ollama pod %s",
                url,
            )
            raise ConnectionError(f"Ollama niedostępna: {exc}") from exc
