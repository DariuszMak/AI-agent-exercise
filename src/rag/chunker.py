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

except ImportError:

    def _token_len(text: str) -> int:
        return int(len(text.split()) * 1.3)


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

    sentences: list[tuple[str, int, int]] = []
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

            while window and window_tokens > overlap_tokens:
                dropped_tokens = _token_len(window[0][0])
                window.pop(0)
                window_tokens -= dropped_tokens

        window.append((sent, s_start, s_end))
        window_tokens += sent_tokens

    if window:
        yield _emit(window)
