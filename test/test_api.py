from flask.testing import FlaskClient


def test_query_without_index(client: FlaskClient) -> None:
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 400


def test_query_invalid_payload(client: FlaskClient) -> None:
    client.post("/index")
    response = client.post("/query", json={})
    assert response.status_code == 400
