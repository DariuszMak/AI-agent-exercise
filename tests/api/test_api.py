from flask.testing import FlaskClient


def test_query_without_index(client: FlaskClient) -> None:
    assert client.post("/query", json={"query": "x"}).status_code == 400


def test_ask_without_index(client: FlaskClient) -> None:
    assert client.post("/ask", json={"query": "x"}).status_code == 400


def test_query_invalid_payload(client: FlaskClient) -> None:
    client.post("/index")
    assert client.post("/query", json={}).status_code == 400
