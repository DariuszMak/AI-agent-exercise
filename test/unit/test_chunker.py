from src.rag.chunker import chunk_text


def test_chunk_text_basic() -> None:
    text = "word " * 1000
    chunks = list(chunk_text(text, chunk_size=200, overlap=50))

    assert len(chunks) > 1
    assert all(len(c.split()) <= 200 for c in chunks)


def test_chunk_overlap() -> None:
    text = " ".join(str(i) for i in range(100))
    chunks = list(chunk_text(text, chunk_size=50, overlap=10))

    assert chunks[1].split()[0] == "40"
