from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .chunker import chunk_text

if TYPE_CHECKING:
    from pathlib import Path


def load_documents(path: Path) -> list[dict[str, Any]]:

    documents: list[dict[str, Any]] = []

    for f in sorted(path.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f.name,
                    "chunk_id": str(i),
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                }
            )

    return documents
