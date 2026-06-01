from __future__ import annotations

import pytest

from src.rag.evaluator import RAGEvaluator


class TestRAGEvaluatorEdgeCases:
    def test_empty_results_fails(self) -> None:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate("any query", [])
        assert result.passed is False
        assert result.score == 0.0
        assert result.best_chunk == ""

    def test_below_min_results_fails(self) -> None:
        evaluator = RAGEvaluator(min_results=3)
        docs = [{"text": "some text", "score": 0.9}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is False
        assert result.score == 0.1

    def test_score_below_threshold_fails(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.5)
        docs = [{"text": "sufficient length text here that is long enough", "score": 0.3}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is False
        assert result.score == pytest.approx(0.3)

    def test_score_above_threshold_passes(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.5)
        docs = [{"text": "this is a sufficiently long text chunk for evaluation", "score": 0.8}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is True
        assert result.score == pytest.approx(0.8)

    def test_short_best_chunk_fails(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.1)
        docs = [{"text": "short", "score": 0.9}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is False
        assert result.score == pytest.approx(0.45)

    def test_best_chunk_exactly_20_chars_passes(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.1)
        docs = [{"text": "12345678901234567890", "score": 0.9}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is True

    def test_best_chunk_19_chars_fails(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.1)
        docs = [{"text": "1234567890123456789", "score": 0.9}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is False

    def test_picks_highest_score_as_best(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.5)
        docs = [
            {"text": "low score chunk that is long enough to pass", "score": 0.2},
            {"text": "high score chunk that is long enough to pass", "score": 0.9},
        ]
        result = evaluator.evaluate("query", docs)
        assert result.passed is True
        assert result.score == pytest.approx(0.9)
        assert "high score" in result.best_chunk

    def test_reason_contains_threshold(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.45)
        docs = [{"text": "long enough text here for the test evaluation check", "score": 0.3}]
        result = evaluator.evaluate("query", docs)
        assert "0.450" in result.reason or "0.45" in result.reason

    def test_missing_score_key_defaults_to_zero(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.1)
        docs = [{"text": "text with no score key at all defined"}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is False

    def test_min_results_exactly_met(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.1, min_results=2)
        docs = [
            {"text": "long enough text here for the test", "score": 0.8},
            {"text": "another long enough text for the test", "score": 0.6},
        ]
        result = evaluator.evaluate("query", docs)
        assert result.passed is True

    def test_custom_threshold(self) -> None:
        evaluator = RAGEvaluator(relevance_threshold=0.20)
        docs = [{"text": "this is definitely a long enough text chunk", "score": 0.25}]
        result = evaluator.evaluate("query", docs)
        assert result.passed is True
