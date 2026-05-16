"""Sentence-transformer embedding model (loaded once at startup)."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def set_embedder(model: SentenceTransformer) -> None:
    global _model
    _model = model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()
