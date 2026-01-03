import pytest
from flask.testing import FlaskClient

from src.app import create_app


@pytest.fixture()
def client() -> FlaskClient:
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()
