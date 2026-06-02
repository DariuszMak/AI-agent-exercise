from typing import TYPE_CHECKING

from src.rag.loader import load_documents

if TYPE_CHECKING:
    from pathlib import Path


def test_load_documents_returns_chunks(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "ksef.txt").write_text("KSeF to Krajowy System e-Faktur.", encoding="utf-8")

    documents = load_documents(docs)

    assert len(documents) > 0
    assert documents[0]["id"] == "ksef.txt"
    assert "KSeF" in documents[0]["text"]


def test_load_documents_chunk_metadata(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "test.txt").write_text(
        "First sentence here. Second sentence here. Third sentence here.",
        encoding="utf-8",
    )

    documents = load_documents(docs)

    for doc in documents:
        assert "token_count" in doc
        assert "char_start" in doc
        assert "char_end" in doc
        assert doc["token_count"] > 0
        assert doc["char_start"] >= 0
        assert doc["char_end"] > doc["char_start"]


def test_load_documents_empty_dir(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    assert load_documents(docs) == []


def test_load_documents_multiple_files_sorted(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "b.txt").write_text("File B content.", encoding="utf-8")
    (docs / "a.txt").write_text("File A content.", encoding="utf-8")

    documents = load_documents(docs)
    ids = [d["id"] for d in documents]

    first_a = next(i for i, x in enumerate(ids) if x == "a.txt")
    first_b = next(i for i, x in enumerate(ids) if x == "b.txt")
    assert first_a < first_b
