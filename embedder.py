"""Local sentence-transformers embedding model - avoids Gemini rate limits."""

import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_model = None

def get_embedder():
    """Load the sentence-transformers model lazily."""
    global _model
    if _model is None:
        # all-MiniLM-L6-v2 is small (20MB) and fast, perfect for free-tier servers
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def set_embedder(model) -> None:
    """Set the model explicitly if needed."""
    global _model
    _model = model

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using local sentence-transformers."""
    model = get_embedder()
    # model.encode returns a numpy array, we convert it to a list of lists
    embeddings = model.encode(texts)
    return embeddings.tolist()