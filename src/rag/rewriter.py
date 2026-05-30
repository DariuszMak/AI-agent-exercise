from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm import OllamaAdapter

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
Jesteś ekspertem od wyszukiwania informacji.

Oryginalne pytanie użytkownika:
"{original_query}"

Poprzednie zapytanie do wyszukiwarki (które dało słabe wyniki):
"{failed_query}"

Przyczyna słabych wyników: {reason}

Wygeneruj JEDNO nowe, ulepszone zapytanie do wyszukiwarki.
- Użyj synonimów lub alternatywnych sformułowań
- Bądź bardziej precyzyjny lub bardziej ogólny (zależnie od kontekstu)
- Odpowiedz TYLKO tekstem zapytania, bez żadnych wyjaśnień
"""


class RAGRewriter:
    """
    Przepisuje zapytanie do RAG gdy poprzednie wyszukiwanie dało słabe wyniki.

    Strategia:
    1. Próbuje LLM-a (semantyczne przepisanie)
    2. Fallback: proste heurystyki bez LLM-a

    Dzięki temu agent nie blokuje się gdy Ollama jest niedostępna.
    """

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
                logger.warning("LLM niedostępny do przepisania zapytania: %s", exc)

        return self._rewrite_heuristic(failed_query, iteration)

    def _rewrite_with_llm(
        self,
        original_query: str,
        failed_query: str,
        reason: str,
    ) -> str:
        prompt = _REWRITE_PROMPT.format(
            original_query=original_query,
            failed_query=failed_query,
            reason=reason,
        )
        response = self._llm.complete(prompt, temperature=0.3)
        rewritten = response.content.strip().strip('"').strip("'")
        logger.info("Przepisano zapytanie: %r → %r", failed_query, rewritten)
        return rewritten

    def _rewrite_heuristic(self, query: str, iteration: int) -> str:
        strategies = [
            lambda q: q.lower(),
            lambda q: " ".join(dict.fromkeys(q.split())),
            lambda q: q + " definicja",
            lambda q: q.split()[0] if q.split() else q,
        ]
        idx = (iteration - 1) % len(strategies)
        rewritten = strategies[idx](query)
        logger.info("Heurystyczne przepisanie zapytania: %r → %r", query, rewritten)
        return rewritten
