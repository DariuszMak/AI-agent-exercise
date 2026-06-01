from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agent.loop import AgentLoop, AgentResult
from src.llm import LLMResponse
from src.mcp_client.mcp_client import MCPClient, MCPToolResult
from src.rag.agentic_retriever import AgenticRetriever
from src.rag.evaluator import RAGEvaluator


def _fake_embed(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.random(384).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_docs(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "id": f"doc_{i}.txt",
            "chunk_id": str(i),
            "text": f"KSeF to system faktur numer {i}. Umożliwia wystawianie faktur.",
            "score": 0.85,
            "token_count": 12,
            "char_start": 0,
            "char_end": 50,
        }
        for i in range(n)
    ]


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(content="KSeF to Krajowy System e-Faktur.", model="gemma:2b", done=True)
    llm.complete_json.return_value = {
        "needs_external_tool": False,
        "tool_name": None,
        "tool_arguments": {},
        "reasoning": "pytanie dotyczy RAG",
    }
    return llm


@pytest.fixture()
def mock_retriever() -> MagicMock:
    retriever = MagicMock(spec=AgenticRetriever)
    retriever.search.return_value = _make_docs()
    return retriever


@pytest.fixture()
def mock_mcp() -> MagicMock:
    mcp = MagicMock(spec=MCPClient)
    mcp.list_tools.return_value = [
        {"name": "log_query", "description": "Loguje zapytanie"},
    ]
    mcp.call_tool.return_value = MCPToolResult(tool_name="log_query", content="OK", is_error=False)
    return mcp


def test_agent_returns_answer_on_good_rag(
    mock_llm: MagicMock,
    mock_retriever: MagicMock,
    mock_mcp: MagicMock,
) -> None:
    evaluator = RAGEvaluator(relevance_threshold=0.5)

    agent = AgentLoop(
        llm=mock_llm,
        retriever=mock_retriever,
        mcp_client=mock_mcp,
        evaluator=evaluator,
        max_iterations=3,
    )

    result = agent.run("Co to jest KSeF?")

    assert isinstance(result, AgentResult)
    assert len(result.answer) > 0
    assert result.total_iterations == 1
    assert result.steps[0].rag_passed is True


def test_agent_retries_on_low_score(
    mock_llm: MagicMock,
    mock_mcp: MagicMock,
) -> None:
    low_score_docs = [
        {
            "id": "x.txt",
            "chunk_id": "0",
            "text": "Nie ma tu żadnych informacji.",
            "score": 0.1,
            "token_count": 5,
            "char_start": 0,
            "char_end": 29,
        }
    ]
    retriever = MagicMock(spec=AgenticRetriever)
    retriever.search.return_value = low_score_docs

    evaluator = RAGEvaluator(relevance_threshold=0.5)
    agent = AgentLoop(
        llm=mock_llm,
        retriever=retriever,
        mcp_client=mock_mcp,
        evaluator=evaluator,
        max_iterations=3,
    )

    result = agent.run("Co to jest KSeF?")

    assert result.total_iterations == 3
    assert retriever.search.call_count == 3


def test_agent_calls_mcp_tool_when_llm_requests(
    mock_llm: MagicMock,
    mock_retriever: MagicMock,
    mock_mcp: MagicMock,
) -> None:
    mock_llm.complete_json.return_value = {
        "needs_external_tool": True,
        "tool_name": "fetch_external_context",
        "tool_arguments": {"topic": "KSeF"},
        "reasoning": "potrzebuję zewnętrznych danych",
    }
    mock_mcp.call_tool.return_value = MCPToolResult(
        tool_name="fetch_external_context",
        content="KSeF działa od 2024.",
        is_error=False,
    )

    evaluator = RAGEvaluator(relevance_threshold=0.5)
    agent = AgentLoop(
        llm=mock_llm,
        retriever=mock_retriever,
        mcp_client=mock_mcp,
        evaluator=evaluator,
        max_iterations=1,
    )

    result = agent.run("Co to jest KSeF?")

    fetch_calls = [call for call in mock_mcp.call_tool.call_args_list if call.args[0] == "fetch_external_context"]
    assert len(fetch_calls) == 1
    assert result.steps[0].tool_called == "fetch_external_context"


def test_agent_handles_mcp_unavailable(
    mock_llm: MagicMock,
    mock_retriever: MagicMock,
) -> None:
    broken_mcp = MagicMock(spec=MCPClient)
    broken_mcp.list_tools.side_effect = ConnectionError("serwer MCP offline")

    evaluator = RAGEvaluator(relevance_threshold=0.5)
    agent = AgentLoop(
        llm=mock_llm,
        retriever=mock_retriever,
        mcp_client=broken_mcp,
        evaluator=evaluator,
        max_iterations=1,
    )

    result = agent.run("Co to jest KSeF?")

    assert isinstance(result, AgentResult)
    assert len(result.answer) > 0
