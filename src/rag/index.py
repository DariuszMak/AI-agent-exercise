from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np
import structlog

from src.rag.embeddings import embed

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

DIM = 384


def create_index() -> faiss.Index:
    return faiss.IndexFlatIP(DIM)


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


@dataclass
class IndexStore:
    index: faiss.Index | None = field(default=None)
    documents: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.documents) > 0

    def build(self, documents: list[dict[str, Any]]) -> None:
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
