from pathlib import Path

from .chunker import chunk_text


def load_documents(path: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []

    for f in path.glob("*.txt"):
        text = f.read_text(encoding="utf-8")

        for i, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f.name,
                    "chunk_id": str(i),
                    "text": chunk,
                }
            )

    return documents
