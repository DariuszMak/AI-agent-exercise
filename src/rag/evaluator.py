from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_RELEVANCE_THRESHOLD = 0.45
DEFAULT_MIN_RESULTS = 1


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    passed: bool
    reason: str
    best_chunk: str


class RAGEvaluator:
    def __init__(
        self,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        min_results: int = DEFAULT_MIN_RESULTS,
    ) -> None:
        self._threshold = relevance_threshold
        self._min_results = min_results

    def evaluate(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> EvaluationResult:
        if not results:
            return EvaluationResult(
                score=0.0,
                passed=False,
                reason="No search results",
                best_chunk="",
            )

        if len(results) < self._min_results:
            return EvaluationResult(
                score=0.1,
                passed=False,
                reason=f"Too few results: {len(results)} < {self._min_results}",
                best_chunk=results[0].get("text", ""),
            )

        best = max(results, key=lambda r: r.get("score", 0.0))
        best_score: float = best.get("score", 0.0)
        best_text: str = best.get("text", "")

        if best_score < self._threshold:
            return EvaluationResult(
                score=best_score,
                passed=False,
                reason=(
                    f"Highest similarity score ({best_score:.3f}) "
                    f"below threshold ({self._threshold:.3f})"
                ),
                best_chunk=best_text,
            )

        if len(best_text.strip()) < 20:
            return EvaluationResult(
                score=best_score * 0.5,
                passed=False,
                reason="Best chunk is too short (< 20 characters)",
                best_chunk=best_text,
            )

        logger.debug(
            "RAG evaluation: score=%.3f, results=%d, query=%r",
            best_score,
            len(results),
            query[:60],
        )

        return EvaluationResult(
            score=best_score,
            passed=True,
            reason=f"Similarity score {best_score:.3f} ≥ threshold {self._threshold:.3f}",
            best_chunk=best_text,
        )