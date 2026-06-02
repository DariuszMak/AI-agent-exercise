from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.llm import LLMResponse
from src.rag.rewriter import RAGRewriter


def test_rewrite_uses_llm_when_available() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(content="new improved query", model="x", done=True)
    rewriter = RAGRewriter(llm=llm)
    result = rewriter.rewrite("original", "failed", "no results")
    assert result == "new improved query"


def test_rewrite_strips_quotes() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(content='"quoted query"', model="x", done=True)
    rewriter = RAGRewriter(llm=llm)
    result = rewriter.rewrite("original", "failed", "reason")
    assert result == "quoted query"


def test_rewrite_strips_single_quotes() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(content="'single quoted'", model="x", done=True)
    rewriter = RAGRewriter(llm=llm)
    result = rewriter.rewrite("original", "failed", "reason")
    assert result == "single quoted"


def test_rewrite_falls_back_on_connection_error() -> None:
    llm = MagicMock()
    llm.complete.side_effect = ConnectionError("Ollama down")
    rewriter = RAGRewriter(llm=llm)
    result = rewriter.rewrite("What is KSeF?", "ksef query", "no results", iteration=1)
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_falls_back_on_value_error() -> None:
    llm = MagicMock()
    llm.complete.side_effect = ValueError("bad response")
    rewriter = RAGRewriter(llm=llm)
    result = rewriter.rewrite("original", "failed", "reason", iteration=1)
    assert isinstance(result, str)


def test_rewrite_passes_temperature() -> None:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(content="result", model="x", done=True)
    rewriter = RAGRewriter(llm=llm)
    rewriter.rewrite("original", "failed", "reason")
    call_kwargs = llm.complete.call_args
    assert call_kwargs.kwargs.get("temperature") == pytest.approx(0.3) or (
        len(call_kwargs.args) > 1 and call_kwargs.args[1] == pytest.approx(0.3)
    )


def test_no_llm_uses_heuristic() -> None:
    rewriter = RAGRewriter(llm=None)
    result = rewriter.rewrite("original", "UPPER CASE QUERY", "reason", iteration=1)
    assert result == "upper case query"


def test_iteration_1_lowercases() -> None:
    rewriter = RAGRewriter(llm=None)
    result = rewriter._rewrite_heuristic("What Is KSeF?", 1)
    assert result == "what is ksef?"


def test_iteration_2_deduplicates_words() -> None:
    rewriter = RAGRewriter(llm=None)
    result = rewriter._rewrite_heuristic("ksef ksef system", 2)
    assert result == "ksef system"


def test_iteration_3_appends_definicja() -> None:
    rewriter = RAGRewriter(llm=None)
    result = rewriter._rewrite_heuristic("KSeF", 3)
    assert result == "KSeF definicja"


def test_iteration_4_takes_first_word() -> None:
    rewriter = RAGRewriter(llm=None)
    result = rewriter._rewrite_heuristic("KSeF system invoices", 4)
    assert result == "KSeF"


def test_iteration_4_empty_query_returns_empty() -> None:
    rewriter = RAGRewriter(llm=None)
    result = rewriter._rewrite_heuristic("", 4)
    assert result == ""


def test_iteration_cycles_back() -> None:
    rewriter = RAGRewriter(llm=None)
    result_iter1 = rewriter._rewrite_heuristic("What Is KSeF?", 1)
    result_iter5 = rewriter._rewrite_heuristic("What Is KSeF?", 5)
    assert result_iter1 == result_iter5


def test_iteration_2_preserves_unique_words() -> None:
    rewriter = RAGRewriter(llm=None)
    result = rewriter._rewrite_heuristic("unique words only", 2)
    assert result == "unique words only"
