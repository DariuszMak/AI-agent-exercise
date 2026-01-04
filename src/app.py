from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, make_response, request

from src.rag.embeddings import embed
from src.rag.index import create_index
from src.rag.loader import load_documents
from src.rag.retriever import search

DEFAULT_DOCUMENTS_PATH = Path("storage/documents")


def create_app(documents_path: Path = DEFAULT_DOCUMENTS_PATH) -> Flask:
    app = Flask(__name__)

    documents = []
    index = None

    @app.post("/index")
    def build_index() -> tuple[dict[str, object], int] | dict[str, int]:
        nonlocal documents, index

        documents = load_documents(documents_path)

        if not documents:
            return {"indexed": 0, "warning": "no documents found"}, 200

        index = create_index()
        vectors = np.array([embed(doc["text"]) for doc in documents], dtype="float32")
        index.add(vectors)

        return {"indexed": len(documents)}

    @app.post("/query")
    def query() -> Response:
        if index is None:
            return make_response(jsonify({"error": "index not built"}), 400)

        data = request.get_json(silent=True)
        if not data or "query" not in data or not isinstance(data["query"], str):
            return make_response(jsonify({"error": "invalid query"}), 400)

        results = search(index, documents, data["query"])
        return jsonify(results)

    return app
