from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, TypeVar, cast

import structlog

from src.helpers.logging_setup import logging_setup

logger = structlog.get_logger(__name__)

_LOG_FILE = Path("storage/query_log.jsonl")

F = TypeVar("F", bound=Callable[..., Any])


class MCPServer:
    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._tools: dict[str, dict[str, Any]] = {}

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            self._tools[name] = {
                "name": name,
                "description": description,
                "inputSchema": input_schema,
                "_fn": fn,
            }
            logger.info("Registered MCP tool: %s", name)
            return fn

        return decorator

    def run(self) -> None:
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode())
                response = server_ref._dispatch(cast("dict[str, Any]", body))

                payload = json.dumps(response).encode()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _log_format: str, *args: object) -> None:
                pass

        httpd = HTTPServer((self._host, self._port), Handler)
        logger.info("MCP server started at %s", self.url)
        httpd.serve_forever()

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method == "tools/list":
                result = {"tools": [{k: v for k, v in t.items() if k != "_fn"} for t in self._tools.values()]}
            elif method == "tools/call":
                result = self._call_tool(cast("dict[str, Any]", params))
            else:
                return _error(req_id, -32601, f"Unknown method: {method}")
        except Exception as exc:
            logger.exception("Error while handling method %s", method)
            return _error(req_id, -32603, str(exc))

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not isinstance(arguments, dict):
            arguments = {}

        if name not in self._tools:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"Unknown tool: {name}",
                    }
                ],
            }

        tool = self._tools[name]
        fn = cast("Callable[..., Any]", tool["_fn"])

        required = tool.get("inputSchema", {}).get("required", [])
        missing = [r for r in required if r not in arguments]

        if missing:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"Missing arguments: {missing}",
                    }
                ],
            }

        try:
            output = fn(**arguments)
        except TypeError as exc:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"Invocation error: {exc}",
                    }
                ],
            }

        return {
            "isError": False,
            "content": [{"type": "text", "text": str(output)}],
        }


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


server = MCPServer()


@server.tool(
    name="log_query",
    description="Stores an agent query in a JSONL log file.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query text",
            },
            "iteration": {
                "type": "integer",
                "description": "Agent iteration number",
            },
            "score": {
                "type": "number",
                "description": "RAG quality score (0.0-1.0)",
            },
        },
        "required": ["query"],
    },
)
def log_query(
    query: str,
    iteration: int = 0,
    score: float = 0.0,
) -> str:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "query": query,
        "iteration": iteration,
        "score": score,
    }

    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return f"Log saved: iteration={iteration}, score={score:.2f}"


@server.tool(
    name="fetch_external_context",
    description="Retrieves additional context from an external source (mock).",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic to search for",
            },
        },
        "required": ["topic"],
    },
)
def fetch_external_context(topic: str) -> str:
    external_db: dict[str, str] = {
        "empire_state_building": (
            "Empire State Building is a 102-story Art Deco skyscraper in Manhattan, New York City, and one of "
            "the world's most famous landmarks and observation towers."
        ),
        "jeddah_tower": (
            "Jeddah Tower is a supertall skyscraper under construction in Saudi Arabia, designed to "
            "become the world's tallest building once completed."
        ),
    }

    key = topic.lower()

    for k, v in external_db.items():
        if k in key:
            return v

    return f"No data available for topic: {topic}"


if __name__ == "__main__":
    logging_setup()
    server.run()
