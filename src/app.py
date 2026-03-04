import pickle
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, make_response, request

from src.rag.embeddings import embed
from src.rag.index import create_index, load_index, save_index
from src.rag.llm import generate_answer
from src.rag.loader import load_documents
from src.rag.retriever import search

DEFAULT_DOCUMENTS_PATH = Path("storage/documents")
DEFAULT_INDEX_PATH = Path("storage/index.faiss")
DEFAULT_DOCSTORE_PATH = Path("storage/documents.pkl")


def create_app(
    documents_path: Path = DEFAULT_DOCUMENTS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    docstore_path: Path = DEFAULT_DOCSTORE_PATH,
    autoload: bool = False,  # <- explicit flag
) -> Flask:
    app = Flask(__name__)

    documents = []
    index = None

    # Auto-load persisted index and documents only if autoload=True
    if autoload and index_path.exists() and docstore_path.exists():
        index = load_index(index_path)
        with docstore_path.open("rb") as f:
            documents = pickle.load(f)

    @app.post("/index")
    def build_index():
        nonlocal documents, index

        documents = load_documents(documents_path)

        if not documents:
            return {"indexed": 0, "warning": "no documents found"}, 200

        index = create_index()
        vectors = np.array([embed(doc["text"]) for doc in documents], dtype="float32")
        index.add(vectors)

        # ✅ persist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        save_index(index, index_path)

        with docstore_path.open("wb") as f:
            pickle.dump(documents, f)

        return {"indexed": len(documents)}

    @app.post("/query")
    def query() -> Response:
        if index is None:
            return make_response(jsonify({"error": "index not built"}), 400)

        data = request.get_json(silent=True)
        if not data or "query" not in data:
            return make_response(jsonify({"error": "invalid query"}), 400)

        results = search(index, documents, data["query"])
        return jsonify(results)

    @app.post("/ask")
    def ask() -> Response:
        if index is None:
            return make_response(jsonify({"error": "index not built"}), 400)

        data = request.get_json(silent=True)
        if not data or "query" not in data:
            return make_response(jsonify({"error": "invalid question"}), 400)

        question = data["query"]

        retrieved = search(index, documents, question, k=5)
        context_chunks = [r["text"] for r in retrieved]

        answer = generate_answer(question, context_chunks)

        return jsonify(
            {
                "question": question,
                "answer": answer,
                "sources": retrieved,
            }
        )

    return app
