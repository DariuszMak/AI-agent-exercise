from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, make_response, request

from src.rag.index import IndexStore
from src.rag.llm import generate_answer
from src.rag.loader import load_documents
from src.rag.retriever import search

DEFAULT_DOCUMENTS_PATH = Path("storage/documents")
DEFAULT_INDEX_PATH = Path("storage/index.faiss")
DEFAULT_DOCSTORE_PATH = Path("storage/documents.json")

_STORE_KEY = "INDEX_STORE"


def create_app(
    documents_path: Path = DEFAULT_DOCUMENTS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    docstore_path: Path = DEFAULT_DOCSTORE_PATH,
    autoload: bool = False,
) -> Flask:
    app = Flask(__name__)

    # Store lives in app.config — injectable, mockable, no closure magic.
    # Each Gunicorn worker calls create_app(autoload=True) independently
    # and reads the persisted files into its own IndexStore.  FAISS reads
    # are safe to do concurrently from separate processes.
    store = IndexStore()
    if autoload and index_path.exists() and docstore_path.exists():
        store = IndexStore.load(index_path, docstore_path)

    app.config[_STORE_KEY] = store

    def _store() -> IndexStore:
        return app.config[_STORE_KEY]  # type: ignore[return-value]

    @app.post("/index")
    def build_index() -> tuple[dict[str, object], int] | dict[str, int]:
        documents = load_documents(documents_path)
        if not documents:
            return {"indexed": 0, "warning": "no documents found"}, 200

        s = IndexStore()
        s.build(documents)
        s.save(index_path, docstore_path)
        app.config[_STORE_KEY] = s

        return {"indexed": len(documents)}

    @app.post("/query")
    def query() -> Any:
        s = _store()
        if not s.ready:
            return make_response(jsonify({"error": "index not built"}), 400)

        data = request.get_json(silent=True)
        if not data or "query" not in data:
            return make_response(jsonify({"error": "invalid query"}), 400)

        assert s.index is not None
        results = search(s.index, s.documents, data["query"])
        return jsonify(results)

    @app.post("/ask")
    def ask() -> Any:
        s = _store()
        if not s.ready:
            return make_response(jsonify({"error": "index not built"}), 400)

        data = request.get_json(silent=True)
        if not data or "query" not in data:
            return make_response(jsonify({"error": "invalid question"}), 400)

        question = data["query"]
        assert s.index is not None
        retrieved = search(s.index, s.documents, question, k=5)
        context_chunks = [r["text"] for r in retrieved]

        answer = generate_answer(question, context_chunks)
        return jsonify({"question": question, "answer": answer, "sources": retrieved})

    return app
