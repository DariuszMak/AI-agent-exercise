from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)


class AgenticRetriever:
    """
    Cienka warstwa adaptera wokół istniejącego kodu FAISS z src/rag/retriever.py.

    Przechowuje referencję do indeksu i dokumentów — agent nie musi
    wiedzieć nic o FAISS, operuje tylko na słownikach Pythona.
    """

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
        logger.debug("FAISS zwrócił %d wyników dla query=%r", len(results), query[:60])
        return results

    @classmethod
    def from_index_store(cls, store: Any) -> AgenticRetriever:
        """
        Fabryczna metoda tworząca retriever z IndexStore z oryginalnego kodu.

        Użycie:
            from src.rag.index import IndexStore
            store = IndexStore.load(index_path, docstore_path)
            retriever = AgenticRetriever.from_index_store(store)
        """
        if not store.ready:
            raise ValueError("IndexStore nie jest gotowy — najpierw wywołaj store.build()")
        return cls(index=store.index, documents=store.documents)
