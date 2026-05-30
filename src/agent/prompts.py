from __future__ import annotations

THINK_PROMPT = """\
Jesteś agentem AI analizującym pytania użytkownika.

Pytanie: "{query}"

Dostępne narzędzia zewnętrzne (MCP):
{tools_list}

Zdecyduj, czy do odpowiedzi na to pytanie potrzebne są narzędzia zewnętrzne.

Odpowiedz WYŁĄCZNIE w formacie JSON (bez bloków kodu):
{{
  "needs_external_tool": true/false,
  "tool_name": "nazwa_narzędzia lub null",
  "tool_arguments": {{}},
  "reasoning": "krótkie uzasadnienie"
}}
"""

ANSWER_PROMPT = """\
Odpowiadaj wyłącznie na podstawie poniższego kontekstu.
Jeżeli kontekst nie zawiera odpowiedzi, powiedz: "Nie wiem".

KONTEKST:
{context}

PYTANIE:
{question}

Odpowiedź:
"""
