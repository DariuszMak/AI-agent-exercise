from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import faiss

logger = structlog.get_logger(__name__)


class AgenticRetriever:
    def __init__(
        self,
        index: faiss.Index,
        documents: list[dict[str, Any]],
    ) -> None:
        self._index = index
        self._documents = documents

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        from src.rag.retriever import search as faiss_search

        results = faiss_search(self._index, self._documents, query, k=k)
        logger.debug("FAISS returned %d results for query=%r", len(results), query[:60])
        return results

    @classmethod
    def from_index_store(cls, store: Any) -> AgenticRetriever:
        if not store.ready:
            raise ValueError("IndexStore is not ready - run store.build() first")
        return cls(index=store.index, documents=store.documents)
