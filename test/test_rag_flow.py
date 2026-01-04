from pathlib import Path

from flask.testing import FlaskClient

from src import app


def test_query_without_index(client: FlaskClient) -> None:
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 400


def test_index_and_query(client: FlaskClient, tmp_path: Path) -> None:
    docs_path = tmp_path / "documents"
    docs_path.mkdir()

    (docs_path / "a.txt").write_text("Flask is a web framework. Flask is written in Python.")

    app.DOCUMENTS_PATH = docs_path

    response = client.post("/index")
    assert response.status_code == 200

    data = response.get_json()
    assert data is not None
    assert data["indexed"] > 0

    response = client.post("/query", json={"query": "What is Flask?"})
    assert response.status_code == 200

    results = response.get_json()
    assert results is not None
    assert len(results) > 0
