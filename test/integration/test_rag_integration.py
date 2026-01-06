from pathlib import Path

from flask.testing import FlaskClient


# test/test_rag_integration.py
def test_index_and_query_polish(client: FlaskClient, tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()

    (docs / "ksef.txt").write_text(
        "KSeF umożliwia wystawianie i odbieranie faktur.",
        encoding="utf-8",
    )

    client.post("/index")

    response = client.post("/query", json={"query": "Do czego służy KSeF?"})
    assert response.status_code == 200

    results = response.get_json()
    assert any("faktur" in r["text"].lower() for r in results)
