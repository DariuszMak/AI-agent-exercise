from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, cast

import requests
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MCPToolResult:
    tool_name: str
    content: Any
    is_error: bool = False
    error_message: str = ""


@dataclass
class MCPClient:
    server_url: str
    timeout: float = 10.0
    _session_id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._rpc("tools/list", {})
        tools: list[dict[str, Any]] = result.get("tools", [])
        logger.debug("Available MCP tools: %s", [t.get("name") for t in tools])
        return tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        try:
            result = self._rpc(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
            )

            content = result.get("content", [])

            if result.get("isError"):
                error_text = _extract_text(content)
                logger.warning(
                    "Tool %s returned an error: %s",
                    tool_name,
                    error_text,
                )
                return MCPToolResult(
                    tool_name=tool_name,
                    content=None,
                    is_error=True,
                    error_message=error_text,
                )

            return MCPToolResult(
                tool_name=tool_name,
                content=_extract_text(content),
            )

        except (ConnectionError, TimeoutError, ValueError) as exc:
            logger.debug(
                "Tool invocation %s failed: %s",
                tool_name,
                exc,
            )
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=str(exc),
            )

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        logger.debug("MCP RPC → %s %s", method, params)

        try:
            response = requests.post(
                self.server_url,
                json=cast("Any", payload),
                headers={
                    "Content-Type": "application/json",
                    "X-Session-Id": self._session_id,
                },
                timeout=self.timeout,
            )

            response.raise_for_status()
            body: dict[str, Any] = response.json()

        except requests.RequestException as exc:
            raise ConnectionError(f"MCP server unavailable ({self.server_url}): {exc}") from exc

        if "error" in body:
            err = body["error"]
            raise ValueError(f"JSON-RPC error {err.get('code')}: {err.get('message')}")

        result = body.get("result", {})
        return result if isinstance(result, dict) else {}


def _extract_text(content: list[dict[str, Any]] | Any) -> str:
    if not isinstance(content, list):
        return str(content)

    texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]

    return "\n".join(texts)
