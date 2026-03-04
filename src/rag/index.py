from pathlib import Path

import faiss  # type: ignore[import-untyped]

DIM = 384


def create_index() -> faiss.Index:
    return faiss.IndexFlatIP(DIM)


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
