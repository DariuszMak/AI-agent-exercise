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
def test_empire_state_building_semantic(client: FlaskClient, tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir(parents=True, exist_ok=True)

    (docs / "empire_state_building.txt").write_text(
        "Empire State Building is a 102-story Art Deco skyscraper in Manhattan, New York City, and one of the world's most famous landmarks and observation towers.",
        encoding="utf-8",
    )

    client.post("/index")

    response = client.post(
        "/query",
        json={"query": "What is Empire State Building?"},
    )

    results = response.get_json()
    assert results is not None
    assert isinstance(results, list)
    assert any("building" in r["text"] for r in results)
