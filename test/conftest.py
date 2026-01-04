from pathlib import Path

import pytest
from flask.testing import FlaskClient

from src.app import create_app


@pytest.fixture()
def client(tmp_path: Path) -> FlaskClient:
    docs = tmp_path / "documents"
    docs.mkdir()
    app = create_app(docs)
    app.config["TESTING"] = True
    return app.test_client()
