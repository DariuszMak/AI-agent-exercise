from pathlib import Path

from flask.testing import FlaskClient


def test_query_without_index(client: FlaskClient) -> None:
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 400


def test_index_and_query(client: FlaskClient, tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()

    (docs_path / "a.txt").write_text("Flask is a web framework. Flask is written in Python.")

    # monkeypatch DOCUMENTS_PATH
    from src import app

    app.DOCUMENTS_PATH = docs_path

    response = client.post("/index")
    assert response.status_code == 200
    assert response.json["indexed"] > 0

    response = client.post("/query", json={"query": "What is Flask?"})
    assert response.status_code == 200
    assert len(response.json) > 0
