from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from src.rag.embeddings import embed
from src.rag.index import create_index
from src.rag.loader import load_documents
from src.rag.retriever import search

DOCUMENTS_PATH = Path("storage/documents")


def create_app() -> Flask:
    app = Flask(__name__)

    documents = []
    index = None

    @app.post("/index")
    def build_index():
        nonlocal documents, index

        documents = load_documents(DOCUMENTS_PATH)

        if not documents:
            return {"indexed": 0, "warning": "no documents found"}, 200

        index = create_index()

        vectors = np.array([embed(doc["text"]) for doc in documents], dtype="float32")

        index.add(vectors)

        return {"indexed": len(documents)}

    @app.post("/query")
    def query() -> dict:
        data = request.json
        results = search(index, documents, data["query"])
        return jsonify(results)

    return app
