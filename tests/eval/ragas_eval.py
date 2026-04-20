"""
RAGAS-based evaluation for the RAG pipeline.

RAGAS computes continuous 0-1 scores instead of binary pass/fail, giving
much more actionable signal when tuning chunking, retrieval, or the LLM.

Install
-------
    pip install ragas datasets langchain-openai

Known issue: small models (gemma:2b, phi:mini, etc.) cannot reliably follow
RAGAS's internal structured-output prompts, producing NaN scores.
Use a model with at least 7B parameters for reliable results.

Recommended local models (via Ollama):
    ollama pull llama3.1:8b
    ollama pull mistral:7b

Or use an API model by setting:
    RAGAS_MODEL=gpt-4o-mini
    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://api.openai.com/v1
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask.testing import FlaskClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_RAGAS_MODEL = os.environ.get("RAGAS_MODEL", "gemma:2b")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

# Models known to be too small to reliably produce structured RAGAS output
_UNRELIABLE_MODELS = {"gemma:2b", "gemma2:2b", "phi", "phi3:mini", "tinyllama"}


def _warn_if_unreliable() -> None:
    base = _RAGAS_MODEL.split(":")[0].lower()
    if _RAGAS_MODEL in _UNRELIABLE_MODELS or base in _UNRELIABLE_MODELS:
        logger.warning(
            "RAGAS judge model '%s' is likely too small to produce valid scores "
            "(expect NaN). Set RAGAS_MODEL to llama3.1:8b, mistral:7b, or an "
            "API model like gpt-4o-mini for reliable results.",
            _RAGAS_MODEL,
        )


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False


def _require_ragas() -> None:
    if not _RAGAS_AVAILABLE:
        msg = (
            "ragas is not installed. "
            "Run: pip install ragas datasets langchain-openai"
        )
        raise ImportError(msg)


def _build_llm() -> Any:
    return ChatOpenAI(
        model=_RAGAS_MODEL,
        openai_api_key=_API_KEY,
        openai_api_base=_BASE_URL,
        temperature=0.0,
    )


def _build_embeddings() -> Any:
    # Ollama ignores the model name here but langchain requires one
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=_API_KEY,
        openai_api_base=_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ask(client: FlaskClient, question: str) -> tuple[str, list[str]]:
    """Call /ask and return (answer, context_chunks)."""
    resp = client.post("/ask", json={"query": question})
    assert resp.status_code == 200, f"/ask failed: {resp.get_json()}"
    data: dict[str, Any] = resp.get_json()
    return data["answer"], [s["text"] for s in data["sources"]]


def build_ragas_dataset(
    client: FlaskClient,
    questions: list[str],
    ground_truths: list[str] | None = None,
) -> Any:
    """
    Run *questions* through the RAG pipeline and assemble a RAGAS Dataset.

    Parameters
    ----------
    client:
        Flask test client with a built index.
    questions:
        Questions to evaluate.
    ground_truths:
        Optional reference answers (same length as *questions*).
        Required for context_recall; omit to skip that metric.

    Returns
    -------
    datasets.Dataset ready to pass to ``ragas.evaluate``.
    """
    _require_ragas()

    rows: dict[str, list[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
    }
    if ground_truths:
        rows["ground_truth"] = []

    for idx, question in enumerate(questions):
        answer, contexts = ask(client, question)
        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        if ground_truths:
            rows["ground_truth"].append(ground_truths[idx])

    return Dataset.from_dict(rows)


def run_ragas_eval(
    client: FlaskClient,
    questions: list[str],
    ground_truths: list[str] | None = None,
    include_recall: bool = False,
) -> Any:
    """
    Run a full RAGAS evaluation and return a results object.

    Returns
    -------
    EvaluationResult with a ``.to_pandas()`` method.

    Raises
    ------
    ValueError
        If all scores are NaN, which indicates the judge model cannot follow
        RAGAS's internal prompts.  The exception message explains what to do.
    """
    _require_ragas()
    _warn_if_unreliable()

    dataset = build_ragas_dataset(client, questions, ground_truths)

    metrics = [faithfulness, answer_relevancy, context_precision]
    if include_recall and ground_truths:
        metrics.append(context_recall)

    llm = _build_llm()
    embeddings = _build_embeddings()

    results = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    df = results.to_pandas()

    # Detect all-NaN columns — this means the model couldn't produce
    # structured output that RAGAS could parse
    nan_metrics = [col for col in df.columns if df[col].isna().all()]
    if nan_metrics:
        raise ValueError(
            f"RAGAS returned NaN for all rows on metrics: {nan_metrics}.\n\n"
            f"This means the judge model '{_RAGAS_MODEL}' cannot reliably follow "
            f"RAGAS's internal structured-output prompts.\n\n"
            f"Fix options:\n"
            f"  1. Use a larger local model:  RAGAS_MODEL=llama3.1:8b\n"
            f"  2. Use an API model:          RAGAS_MODEL=gpt-4o-mini "
            f"OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.openai.com/v1\n"
            f"  3. Skip RAGAS entirely:       pytest -m 'slow and not ragas'"
        )

    # Warn about partial NaN (some rows failed, not all)
    partial_nan = [col for col in df.columns if df[col].isna().any() and col not in nan_metrics]
    if partial_nan:
        nan_counts = {col: int(df[col].isna().sum()) for col in partial_nan}
        logger.warning(
            "RAGAS produced NaN for some rows on %s. "
            "NaN rows are excluded from aggregate score calculations.",
            nan_counts,
        )

    return results