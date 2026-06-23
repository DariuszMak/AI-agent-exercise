from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import structlog

from src.helpers.logging_setup import logging_setup

logging_setup()

logger = structlog.get_logger(__name__)

INDEX_PATH = Path("storage/index.faiss")
DOCSTORE_PATH = Path("storage/documents/EN.json")
MCP_SERVER_URL = "http://localhost:8765"


def build_agent() -> Any:
    from src.agent.loop import AgentLoop
    from src.mcp_client.mcp_client import MCPClient
    from src.rag.agentic_retriever import AgenticRetriever
    from src.rag.api.llm import OllamaAdapter
    from src.rag.evaluator import RAGEvaluator
    from src.rag.index import IndexStore
    from src.rag.rewriter import RAGRewriter

    logger.info("Load FAISS indexes from %s", INDEX_PATH)
    store = IndexStore.load(INDEX_PATH, DOCSTORE_PATH)
    retriever = AgenticRetriever.from_index_store(store)

    llm = OllamaAdapter(model="gemma:2b")

    mcp = None
    try:
        candidate = MCPClient(server_url=MCP_SERVER_URL)
        candidate.list_tools()
        mcp = candidate
        logger.info("MCP server available: %s", MCP_SERVER_URL)
    except ConnectionError:
        logger.warning("MCP server unavailable — agent works without external tools")

    evaluator = RAGEvaluator(relevance_threshold=0.20)
    rewriter = RAGRewriter(llm=llm)

    return AgentLoop(
        llm=llm,
        retriever=retriever,
        mcp_client=mcp,
        evaluator=evaluator,
        rewriter=rewriter,
        max_iterations=3,
    )


def main() -> None:
    query = " ".join(sys.argv[1:]) or "What is Empire State Building?"
    logger.info("Query: %r", query)

    agent = build_agent()
    result = agent.run(query)

    logger.info("=" * 60)
    logger.info("Answer: %s", result.answer)
    logger.info("=" * 60)
    logger.info("Iterations: %d | Score: %.3f", result.total_iterations, result.final_score)

    for step in result.steps:
        status = "OK" if step.rag_passed else "XX"
        query_preview = step.query_used[:55] + "..." if len(step.query_used) > 55 else step.query_used
        logger.info("[%d] %s score=%.3f query=%r", step.iteration, status, step.rag_score, query_preview)
        if step.tool_called:
            logger.info("[%d] MCP -> %s: %s", step.iteration, step.tool_called, str(step.tool_result)[:80])


if __name__ == "__main__":
    main()
