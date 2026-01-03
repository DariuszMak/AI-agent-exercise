from flask.testing import FlaskClient


def test_pivot_file() -> None:
    assert True


def test_index_endpoint(client: FlaskClient) -> None:
    response = client.post("/index")
    assert response.status_code == 200
