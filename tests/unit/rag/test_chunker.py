from src.rag.chunker import Chunk, chunk_text


def test_returns_chunk_dataclass() -> None:
    chunks = list(chunk_text("Hello world. This is a test sentence."))
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_has_metadata() -> None:
    text = "First sentence. Second sentence. Third sentence."
    chunks = list(chunk_text(text))
    for c in chunks:
        assert c.token_count > 0
        assert c.char_start >= 0
        assert c.char_end <= len(text)
        assert c.char_start < c.char_end


def test_single_short_text_is_one_chunk() -> None:
    text = "KSeF to system do faktur."
    chunks = list(chunk_text(text, chunk_tokens=256, overlap_tokens=32))
    assert len(chunks) == 1
    assert "KSeF" in chunks[0].text


def test_long_text_produces_multiple_chunks() -> None:

    sentence = "This is a moderately long sentence used for testing purposes. "
    text = sentence * 60
    chunks = list(chunk_text(text, chunk_tokens=256, overlap_tokens=32))
    assert len(chunks) > 1


def test_overlap_means_last_sentences_repeated() -> None:

    sentence = "Sentence number {}. "
    text = "".join(sentence.format(i) for i in range(80))
    chunks = list(chunk_text(text, chunk_tokens=64, overlap_tokens=16))
    assert len(chunks) >= 2

    last_in_first = chunks[0].text.split(". ")[-2]
    assert last_in_first in chunks[1].text


def test_token_count_within_budget() -> None:
    sentence = "Each sentence contributes a few tokens. "
    text = sentence * 100
    budget = 128
    chunks = list(chunk_text(text, chunk_tokens=budget, overlap_tokens=16))

    for c in chunks:
        assert c.token_count <= budget * 2


def test_empty_string_yields_nothing() -> None:
    assert list(chunk_text("")) == []


def test_whitespace_only_yields_nothing() -> None:
    assert list(chunk_text("   \n\n   ")) == []


def test_char_offsets_are_non_overlapping() -> None:
    text = "Alpha beta. Gamma delta. Epsilon zeta. " * 30
    chunks = list(chunk_text(text, chunk_tokens=32, overlap_tokens=8))

    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)
