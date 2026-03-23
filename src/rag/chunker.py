from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _token_len(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:  # fallback: approximate 1 word ≈ 1.3 tokens

    def _token_len(text: str) -> int:
        return int(len(text.split()) * 1.3)


# Sentence-boundary split pattern: splits on ". ", "? ", "! ", "\n\n", "\n"
_SENT_RE = re.compile(r"(?<=[.?!])\s+|\n{2,}|\n")


@dataclass(frozen=True)
class Chunk:
    text: str
    token_count: int
    char_start: int
    char_end: int


def chunk_text(
    text: str,
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
) -> Iterator[Chunk]:
    """
    Split *text* into token-bounded chunks that respect sentence boundaries.

    Strategy
    --------
    1. Split the text on sentence boundaries first.
    2. Greedily pack sentences into a window until adding the next sentence
       would exceed *chunk_tokens*.
    3. When the window is full, emit a Chunk and slide forward by
       *chunk_tokens - overlap_tokens* worth of tokens, keeping the tail
       sentences for the next window (semantic overlap, not arbitrary
       character slicing).
    4. A single sentence that exceeds *chunk_tokens* on its own is emitted
       as-is rather than dropped — the caller handles oversized chunks.
    """
    sentences: list[tuple[str, int, int]] = []  # (text, char_start, char_end)
    cursor = 0
    for part in _SENT_RE.split(text):
        part = part.strip()
        if part:
            start = text.find(part, cursor)
            end = start + len(part)
            sentences.append((part, start, end))
            cursor = end

    if not sentences:
        return

    window: list[tuple[str, int, int]] = []
    window_tokens = 0

    def _emit(w: list[tuple[str, int, int]]) -> Chunk:
        combined = " ".join(s for s, _, _ in w)
        return Chunk(
            text=combined,
            token_count=_token_len(combined),
            char_start=w[0][1],
            char_end=w[-1][2],
        )

    for sent, s_start, s_end in sentences:
        sent_tokens = _token_len(sent)

        if window_tokens + sent_tokens > chunk_tokens and window:
            yield _emit(window)

            # slide: drop sentences from the front until we're within overlap budget
            while window and window_tokens > overlap_tokens:
                dropped_tokens = _token_len(window[0][0])
                window.pop(0)
                window_tokens -= dropped_tokens

        window.append((sent, s_start, s_end))
        window_tokens += sent_tokens

    if window:
        yield _emit(window)