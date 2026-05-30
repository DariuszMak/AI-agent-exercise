from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    done: bool


class OllamaAdapter:
    """
    Pojedynczy punkt integracji z Ollama.
    Używa wyłącznie stdlib — zero zewnętrznych zależności w tej warstwie.
    Podmień tę klasę na OpenAIAdapter i reszta kodu nie wymaga zmian.
    """

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

    def complete(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    def complete_json(self, prompt: str) -> Any:
        """
        Prosi LLM o odpowiedź w formacie JSON.
        Zwraca sparsowany obiekt Python lub rzuca ValueError.
        """
        json_prompt = (
            f"{prompt}\n\nOdpowiedz WYŁĄCZNIE poprawnym JSON-em, "
            "bez żadnego dodatkowego tekstu ani bloków kodu."
        )
        response = self.complete(json_prompt, temperature=0.0)
        raw = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("LLM zwrócił niepoprawny JSON: %s", raw[:200])
            raise ValueError(f"Niepoprawny JSON od LLM: {raw[:200]}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except URLError as exc:
            logger.exception("Błąd połączenia z Ollama pod %s", url)
            raise ConnectionError(f"Ollama niedostępna: {exc}") from exc
