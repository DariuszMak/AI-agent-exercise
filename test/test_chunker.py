from src.rag.chunker import chunk_text


def test_chunk_text_basic() -> None:
    text = "word " * 1000
    chunks = list(chunk_text(text, chunk_size=200, overlap=50))

    assert len(chunks) > 1
    assert all(len(c.split()) <= 200 for c in chunks)
