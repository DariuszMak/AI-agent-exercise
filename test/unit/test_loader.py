from pathlib import Path

from src.rag.loader import load_documents


def test_load_documents_polish_text(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()

    (docs / "ksef.txt").write_text(
        "KSeF to Krajowy System e-Faktur.",
        encoding="utf-8",
    )

    documents = load_documents(docs)

    assert len(documents) > 0
    assert documents[0]["id"] == "ksef.txt"
    assert "KSeF" in documents[0]["text"]
