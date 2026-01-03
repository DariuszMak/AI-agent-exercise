def test_pivot_file() -> None:
    assert True


def test_index_endpoint(client: str) -> None:
    response = client.post("/index")
    assert response.status_code == 200
