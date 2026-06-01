from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.mcp_client.client import MCPClient, MCPToolResult, _extract_text


def _rpc_response(result: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"jsonrpc": "2.0", "id": "test-id", "result": result}
    return resp


def _error_response(code: int, message: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"jsonrpc": "2.0", "id": "test-id", "error": {"code": code, "message": message}}
    return resp


class TestMCPClientListTools:
    def test_list_tools_returns_tools(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        tools = [{"name": "tool_a"}, {"name": "tool_b"}]
        with patch("requests.post", return_value=_rpc_response({"tools": tools})):
            result = client.list_tools()
        assert result == tools

    def test_list_tools_empty(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        with patch("requests.post", return_value=_rpc_response({})):
            result = client.list_tools()
        assert result == []

    def test_list_tools_raises_on_connection_error(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        with (
            patch("requests.post", side_effect=requests.ConnectionError("refused")),
            pytest.raises(ConnectionError, match="Serwer MCP niedostępny"),
        ):
            client.list_tools()


class TestMCPClientCallTool:
    def test_call_tool_success(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        result_body = {"isError": False, "content": [{"type": "text", "text": "result text"}]}
        with patch("requests.post", return_value=_rpc_response(result_body)):
            result = client.call_tool("my_tool", {"arg": "val"})
        assert isinstance(result, MCPToolResult)
        assert result.is_error is False
        assert result.content == "result text"
        assert result.tool_name == "my_tool"

    def test_call_tool_is_error_true(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        result_body = {"isError": True, "content": [{"type": "text", "text": "something failed"}]}
        with patch("requests.post", return_value=_rpc_response(result_body)):
            result = client.call_tool("broken_tool", {})
        assert result.is_error is True
        assert result.error_message == "something failed"
        assert result.content is None

    def test_call_tool_on_connection_error_returns_error_result(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        with patch("requests.post", side_effect=requests.ConnectionError("down")):
            result = client.call_tool("any_tool", {})
        assert result.is_error is True
        assert result.content is None

    def test_call_tool_on_value_error_returns_error_result(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        with patch("requests.post", return_value=_error_response(-32601, "method not found")):
            result = client.call_tool("missing", {})
        assert result.is_error is True

    def test_call_tool_multiple_text_blocks_joined(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        result_body = {"isError": False, "content": content}
        with patch("requests.post", return_value=_rpc_response(result_body)):
            result = client.call_tool("t", {})
        assert result.content == "first\nsecond"

    def test_call_tool_ignores_non_text_blocks(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        content = [{"type": "image", "data": "xyz"}, {"type": "text", "text": "ok"}]
        result_body = {"isError": False, "content": content}
        with patch("requests.post", return_value=_rpc_response(result_body)):
            result = client.call_tool("t", {})
        assert result.content == "ok"


class TestMCPClientRpc:
    def test_rpc_raises_value_error_on_json_rpc_error(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        with(
            patch("requests.post", return_value=_error_response(-32600, "invalid request")),
            pytest.raises(ValueError, match="-32600"),
        ):
            client._rpc("tools/list", {})

    def test_rpc_raises_connection_error_on_http_error(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("503")
        with patch("requests.post", return_value=resp), pytest.raises(ConnectionError):
            client._rpc("tools/list", {})

    def test_rpc_returns_empty_dict_for_non_dict_result(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"jsonrpc": "2.0", "id": "x", "result": ["not", "a", "dict"]}
        with patch("requests.post", return_value=resp):
            result = client._rpc("tools/list", {})
        assert result == {}

    def test_rpc_sends_correct_headers(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        with patch("requests.post", return_value=_rpc_response({})) as mock_post:
            client._rpc("tools/list", {})
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        assert "X-Session-Id" in headers

    def test_session_id_consistent_across_calls(self) -> None:
        client = MCPClient(server_url="http://localhost:8765")
        captured: list[str] = []
        with patch("requests.post", return_value=_rpc_response({})) as mock_post:
            client._rpc("tools/list", {})
            captured.append(mock_post.call_args.kwargs["headers"]["X-Session-Id"])
            client._rpc("tools/list", {})
            captured.append(mock_post.call_args.kwargs["headers"]["X-Session-Id"])
        assert captured[0] == captured[1]


class TestExtractText:
    def test_extracts_text_from_list(self) -> None:
        content = [{"type": "text", "text": "hello"}]
        assert _extract_text(content) == "hello"

    def test_joins_multiple_texts(self) -> None:
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _extract_text(content) == "a\nb"

    def test_skips_non_text_blocks(self) -> None:
        content = [{"type": "image", "data": "..."}, {"type": "text", "text": "x"}]
        assert _extract_text(content) == "x"

    def test_non_list_returns_str(self) -> None:
        assert _extract_text("raw string") == "raw string"
        assert _extract_text(42) == "42"

    def test_empty_list_returns_empty_string(self) -> None:
        assert _extract_text([]) == ""
