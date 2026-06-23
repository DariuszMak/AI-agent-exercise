from __future__ import annotations

THINK_PROMPT = """\
You are an AI agent analyzing the user's question.

Question: "{query}"

Available external tools (MCP):
{tools_list}

Decide whether external tools are required to answer this question.

Respond ONLY in JSON format (without code blocks):
{{
  "needs_external_tool": true/false,
  "tool_name": "tool_name or null",
  "tool_arguments": {{}},
  "reasoning": "brief justification"
}}
"""

ANSWER_PROMPT = """\
Answer exclusively based on the context provided below.
If the context does not contain the answer, say: "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""
