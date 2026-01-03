import faiss

DIM = 384


def create_index():
    return faiss.IndexFlatIP(DIM)
