from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np

logger = logging.getLogger(__name__)

DIM = 384


def create_index() -> faiss.Index:
    return faiss.IndexFlatIP(DIM)


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


@dataclass
class IndexStore:
    """
    Holds the FAISS index and its associated document metadata together.

    Rationale
    ---------
    The original design stored the index inside a Flask app-factory closure.
    That works fine for a single process but breaks under multi-worker
    deployments (Gunicorn pre-fork model) because each worker gets its own
    copy of the closure — and the index is never populated in workers that
    didn't run ``/index``.

    By making IndexStore an explicit object that lives in ``app.config``,
    we gain:

    - Injectability: tests can construct a pre-populated store without HTTP.
    - Mockability: swap the store object for a stub in unit tests.
    - Worker safety: each worker calls ``IndexStore.load()`` in its own
      ``@app.before_request`` or via ``create_app(autoload=True)``, reading
      the same persisted files independently.  FAISS read is thread-safe.
    - Future migration: replacing FAISS with pgvector or Pinecone is a
      one-class change rather than surgery on the app factory.
    """

    index: faiss.Index | None = field(default=None)
    documents: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.documents) > 0

    def build(self, documents: list[dict[str, Any]]) -> None:
        from src.rag.embeddings import embed  # local import avoids eager load

        idx = create_index()
        vectors = np.array([embed(doc["text"]) for doc in documents], dtype="float32")
        idx.add(vectors)
        self.index = idx
        self.documents = documents

    def save(self, index_path: Path, docstore_path: Path) -> None:
        if self.index is None:
            msg = "Cannot save an empty index"
            raise RuntimeError(msg)
        save_index(self.index, index_path)
        with docstore_path.open("w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)
        logger.info("Saved index (%d docs) to %s", len(self.documents), index_path)

    @classmethod
    def load(cls, index_path: Path, docstore_path: Path) -> IndexStore:
        idx = load_index(index_path)
        with docstore_path.open("r", encoding="utf-8") as f:
            docs = json.load(f)
        logger.info("Loaded index (%d docs) from %s", len(docs), index_path)
        return cls(index=idx, documents=docs)