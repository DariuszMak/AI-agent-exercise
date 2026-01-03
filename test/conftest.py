import pytest

from src.app import create_app


@pytest.fixture()
def client() -> str:
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()
