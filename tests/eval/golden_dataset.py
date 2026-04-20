"""
Golden dataset for RAG evaluation.

Each entry contains:
- question:       the user query
- expected_doc_id: the .txt filename that should appear in top-k results
- ground_truth:   reference answer used by RAGAS context_recall
"""

from __future__ import annotations

from typing import TypedDict


class GoldenEntry(TypedDict):
    question: str
    expected_doc_id: str
    ground_truth: str


GOLDEN_DATASET: list[GoldenEntry] = [
    {
        "question": "What is KSeF?",
        "expected_doc_id": "ksef.txt",
        "ground_truth": (
            "KSeF (Krajowy System e-Faktur) is Poland's National e-Invoice System "
            "that enables issuing and receiving structured invoices electronically."
        ),
    },
    {
        "question": "What is Camunda?",
        "expected_doc_id": "camunda.txt",
        "ground_truth": (
            "Camunda is an open-source workflow and decision automation platform "
            "used for orchestrating complex business processes with BPMN."
        ),
    },
    {
        "question": "What is Devapo?",
        "expected_doc_id": "devapo.txt",
        "ground_truth": (
            "Devapo is a software company specialising in custom software development, "
            "process automation, and digital transformation."
        ),
    },
]
