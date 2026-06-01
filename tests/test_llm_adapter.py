from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.llm import LLMResponse, OllamaAdapter


def _make_response(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = body
    return resp


class TestOllamaAdapterChat:
    def test_chat_returns_llm_response(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {"content": " hello "}, "model": "gemma:2b", "done": True}
        with patch("requests.post", return_value=_make_response(body)):
            result = adapter.chat([{"role": "user", "content": "hi"}])
        assert isinstance(result, LLMResponse)
        assert result.content == "hello"
        assert result.model == "gemma:2b"
        assert result.done is True

    def test_chat_strips_content(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {"content": "  stripped  "}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)):
            result = adapter.chat([{"role": "user", "content": "q"}])
        assert result.content == "stripped"

    def test_chat_missing_content_key(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)):
            result = adapter.chat([{"role": "user", "content": "q"}])
        assert result.content == ""

    def test_chat_missing_model_key_falls_back(self) -> None:
        adapter = OllamaAdapter(model="mymodel")
        body = {"message": {"content": "hi"}}
        with patch("requests.post", return_value=_make_response(body)):
            result = adapter.chat([{"role": "user", "content": "q"}])
        assert result.model == "mymodel"

    def test_chat_done_defaults_to_true(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {"content": "hi"}, "model": "x"}
        with patch("requests.post", return_value=_make_response(body)):
            result = adapter.chat([{"role": "user", "content": "q"}])
        assert result.done is True


class TestOllamaAdapterComplete:
    def test_complete_delegates_to_chat(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {"content": "answer"}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)) as mock_post:
            result = adapter.complete("some prompt")
        assert result.content == "answer"
        call_payload = mock_post.call_args.kwargs["json"]
        assert call_payload["messages"][0]["role"] == "user"
        assert call_payload["messages"][0]["content"] == "some prompt"

    def test_complete_passes_temperature(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {"content": "x"}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)) as mock_post:
            adapter.complete("q", temperature=0.7)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == pytest.approx(0.7)


class TestOllamaAdapterCompleteJson:
    def test_complete_json_parses_valid_json(self) -> None:
        adapter = OllamaAdapter()
        payload = {"key": "value", "num": 42}
        body = {"message": {"content": json.dumps(payload)}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)):
            result = adapter.complete_json("return json")
        assert result == payload

    def test_complete_json_strips_markdown_fences(self) -> None:
        adapter = OllamaAdapter()
        raw = '```json\n{"a": 1}\n```'
        body = {"message": {"content": raw}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)):
            result = adapter.complete_json("return json")
        assert result == {"a": 1}

    def test_complete_json_raises_on_invalid_json(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {"content": "not json at all!!"}, "model": "x", "done": True}
        with(
             patch("requests.post", return_value=_make_response(body)),
             pytest.raises(ValueError, match="Niepoprawny JSON"),
        ):
            adapter.complete_json("return json")


class TestOllamaAdapterPost:
    def test_post_raises_connection_error_on_request_exception(self) -> None:
        adapter = OllamaAdapter()
        with(
            patch("requests.post", side_effect=requests.ConnectionError("refused")),
            pytest.raises(ConnectionError, match="Ollama niedostępna"),
        ):
            adapter.complete("hi")

    def test_post_raises_on_non_dict_response(self) -> None:
        adapter = OllamaAdapter()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = ["not", "a", "dict"]
        with patch("requests.post", return_value=resp), pytest.raises(TypeError):
            adapter.complete("hi")

    def test_post_raises_on_invalid_scheme(self) -> None:
        adapter = OllamaAdapter(base_url="ftp://evil.example.com")
        with pytest.raises(ValueError, match="Nieobsługiwany schemat URL"):
            adapter.complete("hi")

    def test_post_raises_on_http_error(self) -> None:
        adapter = OllamaAdapter()
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch("requests.post", return_value=resp), pytest.raises(ConnectionError):
            adapter.complete("hi")

    def test_post_uses_correct_url(self) -> None:
        adapter = OllamaAdapter(base_url="http://myhost:1234")
        body = {"message": {"content": "ok"}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)) as mock_post:
            adapter.complete("q")
        url = mock_post.call_args.args[0]
        assert url == "http://myhost:1234/api/chat"

    def test_post_sends_stream_false(self) -> None:
        adapter = OllamaAdapter()
        body = {"message": {"content": "ok"}, "model": "x", "done": True}
        with patch("requests.post", return_value=_make_response(body)) as mock_post:
            adapter.complete("q")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["stream"] is False
