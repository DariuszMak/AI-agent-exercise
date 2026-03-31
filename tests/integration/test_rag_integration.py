
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from flask.testing import FlaskClient


def test_index_and_query(client: FlaskClient, tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir(exist_ok=True)

    (docs / "a.txt").write_text("Flask is a web framework. Flask is written in Python.")

    client.post("/index")

    response = client.post("/query", json={"query": "What is Flask?"})
    assert response.status_code == 200

    results = response.get_json()
    assert results is not None
    assert len(results) > 0


@pytest.mark.slow
def test_ksef_semantic_polish(client: FlaskClient, tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir(parents=True, exist_ok=True)

    (docs / "ksef.txt").write_text(
        "KSeF umożliwia wystawianie i odbieranie faktur ustrukturyzowanych.",
        encoding="utf-8",
    )

    client.post("/index")

    response = client.post(
        "/query",
        json={"query": "Do czego służy KSeF?"},
    )

    results = response.get_json()
    assert results is not None
    assert isinstance(results, list)
    assert any("faktur" in r["text"] for r in results)
