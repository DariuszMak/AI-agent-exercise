from collections.abc import Iterable


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> Iterable[str]:
    words = text.split()
    start = 0

    while start < len(words):
        end = start + chunk_size
        yield " ".join(words[start:end])
        start = end - overlap
        if start < 0:
            start = 0
