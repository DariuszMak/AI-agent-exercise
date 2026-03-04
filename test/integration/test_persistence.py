from pathlib import Path
from typing import TYPE_CHECKING

from src.app import create_app

if TYPE_CHECKING:
    from flask.testing import FlaskClient


def test_index_persistence(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()

    
    (docs / "a.txt").write_text(
        "KSeF to system do faktur.",
        encoding="utf-8",
    )

    index_path = tmp_path / "index.faiss"
    docstore_path = tmp_path / "documents.pkl"

    
    
    
    app1 = create_app(
        documents_path=docs,
        index_path=index_path,
        docstore_path=docstore_path,
        autoload=False,  
    )
    app1.config["TESTING"] = True
    client1: FlaskClient = app1.test_client()

    response = client1.post("/index")
    assert response.status_code == 200
    assert response.get_json()["indexed"] > 0

    
    assert index_path.exists()
    assert docstore_path.exists()

    
    
    
    app2 = create_app(
        documents_path=docs,
        index_path=index_path,
        docstore_path=docstore_path,
        autoload=True,  
    )
    app2.config["TESTING"] = True
    client2: FlaskClient = app2.test_client()

    
    response = client2.post("/query", json={"query": "Czym jest KSeF?"})
    assert response.status_code == 200

    results = response.get_json()
    assert results is not None
    assert len(results) > 0
    assert "KSeF" in results[0]["text"]
