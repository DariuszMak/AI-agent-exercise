import faiss

DIM = 384


def create_index() -> faiss.Index:
    return faiss.IndexFlatIP(DIM)
