from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.mcp_client.server import MCPServer, _error, fetch_external_context, log_query

if TYPE_CHECKING:
    from pathlib import Path


class TestMCPServerDispatch:
    def setup_method(self) -> None:
        self.server = MCPServer()

        @self.server.tool(
            name="echo",
            description="Returns the input",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        def echo(text: str) -> str:
            return text

    def test_tools_list_returns_registered_tools(self) -> None:
        req = {"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}}
        resp = self.server._dispatch(req)
        tools = resp["result"]["tools"]
        assert any(t["name"] == "echo" for t in tools)

    def test_tools_list_excludes_fn_key(self) -> None:
        req = {"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}}
        resp = self.server._dispatch(req)
        for tool in resp["result"]["tools"]:
            assert "_fn" not in tool

    def test_tools_call_success(self) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello"}},
        }
        resp = self.server._dispatch(req)
        assert resp["result"]["isError"] is False
        assert resp["result"]["content"][0]["text"] == "hello"

    def test_tools_call_unknown_tool(self) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        }
        resp = self.server._dispatch(req)
        assert resp["result"]["isError"] is True
        assert "Nieznane narzędzie" in resp["result"]["content"][0]["text"]

    def test_tools_call_missing_required_arg(self) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": "4",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {}},
        }
        resp = self.server._dispatch(req)
        assert resp["result"]["isError"] is True
        assert "Brakujące argumenty" in resp["result"]["content"][0]["text"]

    def test_unknown_method_returns_error(self) -> None:
        req = {"jsonrpc": "2.0", "id": "5", "method": "nonexistent/method", "params": {}}
        resp = self.server._dispatch(req)
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_tools_call_non_dict_arguments_handled(self) -> None:
        req = {
            "jsonrpc": "2.0",
            "id": "6",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": None},
        }
        resp = self.server._dispatch(req)
        assert resp["result"]["isError"] is True

    def test_dispatch_preserves_request_id(self) -> None:
        req = {"jsonrpc": "2.0", "id": "my-id-99", "method": "tools/list", "params": {}}
        resp = self.server._dispatch(req)
        assert resp["id"] == "my-id-99"


class TestMCPServerToolDecorator:
    def test_decorator_registers_tool(self) -> None:
        server = MCPServer()

        @server.tool(name="t1", description="desc", input_schema={})
        def t1() -> str:
            return "ok"

        assert "t1" in server._tools

    def test_decorator_preserves_fn(self) -> None:
        server = MCPServer()

        @server.tool(name="t2", description="d", input_schema={})
        def t2() -> str:
            return "result"

        assert server._tools["t2"]["_fn"]() == "result"

    def test_tool_raises_type_error_on_bad_call(self) -> None:
        server = MCPServer()

        @server.tool(name="typed", description="d", input_schema={"required": []})
        def typed(x: int) -> str:
            return str(x)

        req = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/call",
            "params": {"name": "typed", "arguments": {"wrong": "arg"}},
        }
        resp = server._dispatch(req)
        assert resp["result"]["isError"] is True


class TestBuiltinTools:
    def test_log_query_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.mcp_client.server as srv_module

        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(srv_module, "_LOG_FILE", log_path)

        result = log_query(query="test query", iteration=1, score=0.75)

        assert log_path.exists()
        assert "iteration=1" in result
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["query"] == "test query"
        assert entry["iteration"] == 1
        assert entry["score"] == pytest.approx(0.75)

    def test_log_query_appends_multiple_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.mcp_client.server as srv_module

        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(srv_module, "_LOG_FILE", log_path)

        log_query(query="first", iteration=1)
        log_query(query="second", iteration=2)

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_log_query_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.mcp_client.server as srv_module

        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(srv_module, "_LOG_FILE", log_path)

        log_query(query="minimal")

        entry = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert entry["iteration"] == 0
        assert entry["score"] == pytest.approx(0.0)

    def test_fetch_external_context_ksef(self) -> None:
        result = fetch_external_context("ksef")
        assert "KSeF" in result

    def test_fetch_external_context_camunda(self) -> None:
        result = fetch_external_context("Camunda platform")
        assert "Camunda" in result

    def test_fetch_external_context_unknown(self) -> None:
        result = fetch_external_context("something completely unknown")
        assert "Brak danych" in result

    def test_fetch_external_context_case_insensitive(self) -> None:
        result_lower = fetch_external_context("ksef invoices")
        result_upper = fetch_external_context("KSEF INVOICES")
        assert result_lower == result_upper


class TestErrorHelper:
    def test_error_structure(self) -> None:
        result = _error("req-1", -32601, "not found")
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req-1"
        assert result["error"]["code"] == -32601
        assert result["error"]["message"] == "not found"
