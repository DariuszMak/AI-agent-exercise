from __future__ import annotations

import pytest

from src.mcp_client.tools import ToolDefinition, ToolRegistry


class TestToolDefinition:
    def test_from_mcp_dict_full(self) -> None:
        data = {
            "name": "my_tool",
            "description": "does stuff",
            "inputSchema": {"type": "object", "properties": {}},
        }
        td = ToolDefinition.from_mcp_dict(data)
        assert td.name == "my_tool"
        assert td.description == "does stuff"
        assert td.input_schema == {"type": "object", "properties": {}}

    def test_from_mcp_dict_missing_description(self) -> None:
        data = {"name": "tool_x", "inputSchema": {}}
        td = ToolDefinition.from_mcp_dict(data)
        assert td.description == ""

    def test_from_mcp_dict_missing_schema(self) -> None:
        data = {"name": "tool_y", "description": "d"}
        td = ToolDefinition.from_mcp_dict(data)
        assert td.input_schema == {}

    def test_to_prompt_line(self) -> None:
        td = ToolDefinition(name="search", description="searches stuff", input_schema={})
        line = td.to_prompt_line()
        assert line == "- search: searches stuff"


class TestToolRegistry:
    def test_empty_registry(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.all() == []

    def test_register_and_has(self) -> None:
        registry = ToolRegistry()
        td = ToolDefinition(name="t1", description="d", input_schema={})
        registry.register(td)
        assert registry.has("t1")
        assert not registry.has("other")

    def test_get_existing(self) -> None:
        registry = ToolRegistry()
        td = ToolDefinition(name="fetch", description="gets data", input_schema={})
        registry.register(td)
        result = registry.get("fetch")
        assert result is td

    def test_get_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="missing_tool"):
            registry.get("missing_tool")

    def test_all_returns_all_registered(self) -> None:
        registry = ToolRegistry()
        t1 = ToolDefinition(name="a", description="A", input_schema={})
        t2 = ToolDefinition(name="b", description="B", input_schema={})
        registry.register(t1)
        registry.register(t2)
        all_tools = registry.all()
        assert len(all_tools) == 2
        assert t1 in all_tools
        assert t2 in all_tools

    def test_len(self) -> None:
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(ToolDefinition(name="x", description="", input_schema={}))
        assert len(registry) == 1

    def test_to_prompt_block_empty(self) -> None:
        registry = ToolRegistry()
        block = registry.to_prompt_block()
        assert block == "(brak dostępnych narzędzi)"

    def test_to_prompt_block_with_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="tool_a", description="does a", input_schema={}))
        registry.register(ToolDefinition(name="tool_b", description="does b", input_schema={}))
        block = registry.to_prompt_block()
        assert "tool_a" in block
        assert "tool_b" in block
        assert "\n" in block

    def test_load_from_server(self) -> None:
        from unittest.mock import MagicMock
        registry = ToolRegistry()
        mcp_client = MagicMock()
        mcp_client.list_tools.return_value = [
            {"name": "srv_tool", "description": "from server", "inputSchema": {"type": "object"}},
        ]
        registry.load_from_server(mcp_client)
        assert registry.has("srv_tool")
        assert registry.get("srv_tool").description == "from server"

    def test_register_overwrites_existing(self) -> None:
        registry = ToolRegistry()
        t1 = ToolDefinition(name="dup", description="first", input_schema={})
        t2 = ToolDefinition(name="dup", description="second", input_schema={})
        registry.register(t1)
        registry.register(t2)
        assert registry.get("dup").description == "second"
        assert len(registry) == 1
