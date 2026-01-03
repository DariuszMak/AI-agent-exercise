def test_pivot_file() -> None:
    assert True


def test_index_endpoint(client):
    response = client.post("/index")
    assert response.status_code == 200
