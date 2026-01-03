from sentence_transformers import SentenceTransformer
from torch import Tensor

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> Tensor:
    return _model.encode(text)
