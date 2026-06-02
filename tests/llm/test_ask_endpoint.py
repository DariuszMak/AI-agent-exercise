from typing import TYPE_CHECKING, Any

import pytest

from src.rag.api.app import create_app

if TYPE_CHECKING:
    from pathlib import Path

    from flask import Flask
    from flask.testing import FlaskClient


@pytest.fixture()
def app_with_docs(tmp_path: Path) -> Flask:
    docs = tmp_path / "documents"
    docs.mkdir()

    (docs / "ksef.txt").write_text(
        "KSeF umożliwia wystawianie i odbieranie faktur ustrukturyzowanych w Polsce.",
        encoding="utf-8",
    )

    app = create_app(documents_path=docs)
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(app_with_docs: Flask) -> Any:
    return app_with_docs.test_client()


def test_ask_without_index(client: FlaskClient) -> None:
    response = client.post("/ask", json={"query": "Do czego służy KSeF?"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "index not built"


@pytest.mark.slow
def test_ask_happy_path_polish(client: FlaskClient) -> None:
    response = client.post("/index")
    assert response.status_code == 200
    assert response.get_json()["indexed"] > 0

    response = client.post(
        "/ask",
        json={"query": "Do czego służy KSeF?"},
    )

    assert response.status_code == 200

    data = response.get_json()
    assert data is not None

    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0

    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0

    assert any("faktur" in s["text"].lower() for s in data["sources"])
