from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunker import chunk_text


def load_documents(path: Path) -> list[dict[str, Any]]:
    """
    Load all .txt files under *path* and return a flat list of chunk dicts.

    Each dict carries:
    - ``id``          : source filename
    - ``chunk_id``    : zero-based chunk index within the file
    - ``text``        : chunk text
    - ``token_count`` : token count from the chunker (useful for debugging)
    - ``char_start``  : character offset in the original file
    - ``char_end``    : character offset in the original file
    """
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