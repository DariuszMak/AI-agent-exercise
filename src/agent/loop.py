from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.prompts import ANSWER_PROMPT, THINK_PROMPT
from src.rag.evaluator import RAGEvaluator
from src.rag.rewriter import RAGRewriter

if TYPE_CHECKING:
    from src.llm import OllamaAdapter
    from src.mcp_client.client import MCPClient
    from src.rag.agentic_retriever import AgenticRetriever

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


@dataclass
class AgentStep:
    iteration: int
    query_used: str
    rag_score: float
    rag_passed: bool
    tool_called: str | None
    tool_result: str | None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    final_score: float = 0.0


class AgentLoop:
    def __init__(
        self,
        llm: OllamaAdapter,
        retriever: AgenticRetriever,
        mcp_client: MCPClient | None = None,
        evaluator: RAGEvaluator | None = None,
        rewriter: RAGRewriter | None = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._mcp = mcp_client
        self._evaluator = evaluator or RAGEvaluator()
        self._rewriter = rewriter or RAGRewriter(llm=llm)
        self._max_iter = max_iterations
        self._available_tools: list[dict[str, Any]] = []

    def run(self, query: str) -> AgentResult:
        logger.info("Agent START | query=%r", query)
        self._refresh_tools()

        steps: list[AgentStep] = []
        current_query = query
        context_chunks: list[str] = []

        for iteration in range(1, self._max_iter + 1):
            logger.info("─── Iteracja %d/%d ───", iteration, self._max_iter)

            tool_called: str | None = None
            tool_result: str | None = None

            think_result = self._think(current_query)
            if think_result.get("needs_external_tool") and self._mcp:
                tool_called, tool_result = self._act_mcp(think_result, current_query)
                if tool_result:
                    context_chunks.append(f"[Narzędzie: {tool_called}]\n{tool_result}")

            rag_results = self._act_rag(current_query)
            eval_result = self._evaluator.evaluate(current_query, rag_results)

            steps.append(
                AgentStep(
                    iteration=iteration,
                    query_used=current_query,
                    rag_score=eval_result.score,
                    rag_passed=eval_result.passed,
                    tool_called=tool_called,
                    tool_result=tool_result,
                )
            )

            self._log_via_mcp(current_query, iteration, eval_result.score)

            if eval_result.passed:
                logger.info("Wyniki wystarczające (score=%.3f) — generuję odpowiedź", eval_result.score)
                context_chunks.extend(r["text"] for r in rag_results)
                return self._build_result(query, context_chunks, steps, eval_result.score, iteration)

            if iteration < self._max_iter:
                logger.info("Przepisuję zapytanie (score=%.3f < próg)", eval_result.score)
                current_query = self._rewriter.rewrite(
                    original_query=query,
                    failed_query=current_query,
                    reason=eval_result.reason,
                    iteration=iteration,
                )
            else:
                logger.warning("Osiągnięto limit iteracji — odpowiadam z najlepszym kontekstem")
                context_chunks.extend(r["text"] for r in rag_results)

        return self._build_result(query, context_chunks, steps, 0.0, self._max_iter)

    def _think(self, query: str) -> dict[str, Any]:
        if not self._available_tools:
            return {"needs_external_tool": False, "reasoning": "brak narzędzi MCP"}

        tools_list = "\n".join(f"- {t['name']}: {t.get('description', '')}" for t in self._available_tools)
        prompt = THINK_PROMPT.format(query=query, tools_list=tools_list)

        try:
            decision = self._llm.complete_json(prompt)
            logger.debug("THINK decision: %s", decision)
            return decision
        except (ValueError, ConnectionError) as exc:
            logger.warning("THINK fallback do RAG (błąd LLM): %s", exc)
            return {"needs_external_tool": False, "reasoning": f"fallback: {exc}"}

    def _act_mcp(self, think_result: dict[str, Any], query: str = "") -> tuple[str | None, str | None]:
        tool_name: str | None = think_result.get("tool_name")
        tool_args: dict[str, Any] = think_result.get("tool_arguments") or {}

        if not tool_name or self._mcp is None:
            return None, None

        if tool_name == "fetch_external_context" and "topic" not in tool_args:
            tool_args = {"topic": query[:80]}

        logger.info("ACT MCP | tool=%s args=%s", tool_name, tool_args)
        result = self._mcp.call_tool(tool_name, tool_args)

        if result.is_error:
            logger.warning("Narzędzie MCP zwróciło błąd: %s", result.error_message)
            return tool_name, None

        return tool_name, str(result.content)

    def _act_rag(self, query: str) -> list[dict[str, Any]]:
        logger.info("ACT RAG | query=%r", query[:80])
        return self._retriever.search(query, k=5)

    def _log_via_mcp(self, query: str, iteration: int, score: float) -> None:
        if self._mcp is None:
            return
        with contextlib.suppress(Exception):
            self._mcp.call_tool(
                "log_query",
                {"query": query, "iteration": iteration, "score": score},
            )

    def _refresh_tools(self) -> None:
        if self._mcp is None:
            return
        try:
            self._available_tools = self._mcp.list_tools()
        except ConnectionError as exc:
            logger.warning("Nie udało się pobrać listy narzędzi MCP: %s", exc)
            self._available_tools = []

    def _build_result(
        self,
        original_query: str,
        context_chunks: list[str],
        steps: list[AgentStep],
        final_score: float,
        iterations: int,
    ) -> AgentResult:
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "Brak kontekstu."
        prompt = ANSWER_PROMPT.format(context=context, question=original_query)

        try:
            response = self._llm.complete(prompt, temperature=0.0)
            answer = response.content
        except ConnectionError as exc:
            answer = f"[Błąd LLM: {exc}]"

        logger.info("Agent DONE | iterations=%d, score=%.3f", iterations, final_score)
        return AgentResult(
            answer=answer,
            steps=steps,
            total_iterations=iterations,
            final_score=final_score,
        )
