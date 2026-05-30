from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

INDEX_PATH = Path("storage/index.faiss")
DOCSTORE_PATH = Path("storage/documents/EN.json")
MCP_SERVER_URL = "http://localhost:8765"


def build_agent():
    from src.agent.loop import AgentLoop
    from src.llm import OllamaAdapter
    from src.mcp_client.client import MCPClient
    from src.rag.agentic_retriever import AgenticRetriever
    from src.rag.evaluator import RAGEvaluator
    from src.rag.index import IndexStore
    from src.rag.rewriter import RAGRewriter

    logger.info("Ładuję indeks FAISS z %s", INDEX_PATH)
    store = IndexStore.load(INDEX_PATH, DOCSTORE_PATH)
    retriever = AgenticRetriever.from_index_store(store)

    llm = OllamaAdapter(model="gemma:2b")

    mcp = None
    try:
        candidate = MCPClient(server_url=MCP_SERVER_URL)
        candidate.list_tools()
        mcp = candidate
        logger.info("Serwer MCP dostępny pod %s", MCP_SERVER_URL)
    except ConnectionError:
        logger.warning("Serwer MCP niedostępny — agent działa bez narzędzi zewnętrznych")

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
    query = " ".join(sys.argv[1:]) or "Co to jest KSeF?"
    logger.info("Pytanie: %r", query)

    agent = build_agent()
    result = agent.run(query)

    print("\n" + "=" * 60)
    print(f"ODPOWIEDŹ:\n{result.answer}")
    print("=" * 60)
    print(f"Iteracje: {result.total_iterations}  |  Ostatni score: {result.final_score:.3f}")
    print("\nKroki:")
    for step in result.steps:
        tool_info = f" | MCP: {step.tool_called}" if step.tool_called else ""
        status = "OK" if step.rag_passed else "XX"
        print(
            f"  [{step.iteration}] {status} score={step.rag_score:.3f}"
            f"  query={step.query_used[:50]!r}{tool_info}"
        )


if __name__ == "__main__":
    main()
