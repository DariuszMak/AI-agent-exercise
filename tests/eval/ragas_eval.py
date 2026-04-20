"""
RAGAS-based evaluation for the RAG pipeline.

RAGAS computes continuous 0-1 scores instead of binary pass/fail, giving
much more actionable signal when tuning chunking, retrieval, or the LLM.

Install
-------
    pip install ragas datasets langchain-openai

The evaluator re-uses your existing Ollama setup via the OpenAI-compatible
endpoint.  Swap *RAGAS_MODEL* for a stronger model (e.g. "gpt-4o") to get
more reliable judge scores — small models like gemma:2b produce noisy results.

Metrics
-------
faithfulness       : Is every claim in the answer grounded in the context?
answer_relevancy   : Does the answer actually address the question asked?
context_precision  : Are the retrieved chunks relevant to the question?
context_recall     : Were all facts from the ground truth retrieved?
                     (requires ground_truth in the dataset — see golden_dataset.py)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask.testing import FlaskClient

# ---------------------------------------------------------------------------
# Optional imports — fail with a clear message if ragas is not installed
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


_RAGAS_MODEL = os.environ.get("RAGAS_MODEL", "gemma:2b")
_RAGAS_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
_RAGAS_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")


def _require_ragas() -> None:
    if not _RAGAS_AVAILABLE:
        msg = "ragas is not installed. Run: pip install ragas datasets langchain-openai"
        raise ImportError(msg)


def _build_llm() -> Any:
    return ChatOpenAI(
        model=_RAGAS_MODEL,
        openai_api_key=_RAGAS_API_KEY,
        openai_api_base=_RAGAS_BASE_URL,
        temperature=0.0,
    )


def _build_embeddings() -> Any:
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",  # ignored by Ollama but required by ragas
        openai_api_key=_RAGAS_API_KEY,
        openai_api_base=_RAGAS_BASE_URL,
    )


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

    The results object has a ``.to_pandas()`` method for tabular inspection
    and individual metric keys for programmatic access.

    Example
    -------
    >>> results = run_ragas_eval(client, questions, ground_truths)
    >>> print(results)
    >>> df = results.to_pandas()
    >>> print(df[["question", "faithfulness", "answer_relevancy"]])
    """
    _require_ragas()

    dataset = build_ragas_dataset(client, questions, ground_truths)

    metrics = [faithfulness, answer_relevancy, context_precision]
    if include_recall and ground_truths:
        metrics.append(context_recall)

    llm = _build_llm()
    embeddings = _build_embeddings()

    return evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
