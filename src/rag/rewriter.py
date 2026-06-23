from __future__ import annotations

from typing import TYPE_CHECKING, cast

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.rag.api.llm import OllamaAdapter

logger = structlog.get_logger(__name__)

_REWRITE_PROMPT = """\
You are an expert in information retrieval.

Original user question:
"{original_query}"

Previous search query (which produced poor results):
"{failed_query}"

Reason for the poor results: {reason}

Generate ONE new, improved search query.
- Use synonyms or alternative wording.
- Be either more specific or more general, depending on the context.
- Respond ONLY with the search query text, without any explanations.
"""


class RAGRewriter:
    def __init__(self, llm: OllamaAdapter | None = None) -> None:
        self._llm = llm

    def rewrite(
        self,
        original_query: str,
        failed_query: str,
        reason: str,
        iteration: int = 1,
    ) -> str:
        if self._llm is not None:
            try:
                return self._rewrite_with_llm(original_query, failed_query, reason)
            except (ConnectionError, ValueError) as exc:
                logger.warning("LLM unavailable for query rewriting: %s", exc)

        return self._rewrite_heuristic(failed_query, iteration)

    def _rewrite_with_llm(
        self,
        original_query: str,
        failed_query: str,
        reason: str,
    ) -> str:
        llm = cast("OllamaAdapter", self._llm)

        prompt = _REWRITE_PROMPT.format(
            original_query=original_query,
            failed_query=failed_query,
            reason=reason,
        )

        response = llm.complete(prompt, temperature=0.3)

        content = str(getattr(response, "content", "")).strip()

        rewritten = content.strip('"').strip("'")

        logger.info("Rewrote query: %r → %r", failed_query, rewritten)
        return rewritten

    def _rewrite_heuristic(self, query: str, iteration: int) -> str:
        strategies: list[Callable[[str], str]] = [
            lambda q: q.lower(),
            lambda q: " ".join(dict.fromkeys(q.split())),
            lambda q: q + " definition",
            lambda q: q.split()[0] if q.split() else q,
        ]

        idx = (iteration - 1) % len(strategies)
        rewritten = strategies[idx](query)

        logger.info("Heuristic query rewrite: %r → %r", query, rewritten)
        return rewritten
