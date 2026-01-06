import numpy as np

from src.rag.embeddings import embed
from src.rag.index import create_index
from src.rag.retriever import search


def test_faiss_retrieval_order() -> None:
    documents = [
        {"id": "1", "chunk_id": "0", "text": "KSeF to system faktur"},
        {"id": "2", "chunk_id": "0", "text": "Kot siedzi na dachu"},
    ]

    index = create_index()
    vectors = np.array([embed(d["text"]) for d in documents], dtype="float32")
    index.add(vectors)

    results = search(index, documents, "Czym jest KSeF?", k=1)

    assert results[0]["id"] == "1"
