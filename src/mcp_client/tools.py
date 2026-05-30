from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """
    Lokalna reprezentacja narzędzia MCP.
    Pobierana z serwera przez MCPClient.list_tools() i cachowana tutaj,
    żeby agent nie odpytywał serwera przy każdej iteracji.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_prompt_line(self) -> str:
        return f"- {self.name}: {self.description}"

    @classmethod
    def from_mcp_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_schema=data.get("inputSchema", {}),
        )


class ToolRegistry:
    """
    Rejestr narzędzi MCP dostępnych dla agenta.

    Odpowiedzialność:
    - cache listy narzędzi pobranej z serwera MCP
    - walidacja czy narzędzie istnieje przed wywołaniem
    - formatowanie listy narzędzi do promptu fazy THINK

    Użycie:
        registry = ToolRegistry()
        registry.load_from_server(mcp_client)

        if registry.has("log_query"):
            args = registry.get("log_query").input_schema
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def load_from_server(self, mcp_client: Any) -> None:
        raw_tools: list[dict[str, Any]] = mcp_client.list_tools()
        self._tools = {
            t["name"]: ToolDefinition.from_mcp_dict(t)
            for t in raw_tools
        }

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Narzędzie '{name}' nie istnieje w rejestrze")
        return self._tools[name]

    def all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def to_prompt_block(self) -> str:
        if not self._tools:
            return "(brak dostępnych narzędzi)"
        return "\n".join(t.to_prompt_line() for t in self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)