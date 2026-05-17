"""Gemini embedding model - zero RAM, runs on Google's servers."""

import time
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Confirmed available via genai.list_models() on this API key
GEMINI_MODEL = "models/gemini-embedding-001"   # stable; upgrade to gemini-embedding-2 if needed
_configured = False


def _configure():
    global _configured
    if not _configured:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _configured = True


def get_embedder():
    """No-op — kept for compatibility with build_index.py."""
    _configure()
    return None


def set_embedder(model) -> None:
    """No-op — kept for compatibility."""
    pass


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Gemini API."""
    _configure()
    embeddings = []
    for i, text in enumerate(texts):
        try:
            result = genai.embed_content(
                model=GEMINI_MODEL,
                content=text[:2000]  # cap length
            )
            embeddings.append(result["embedding"])
        except Exception as e:
            print(f"Embedding failed for text {i}: {e}")
            # Return zero vector as fallback (768 dims for text-embedding-004)
            embeddings.append([0.0] * 768)
        time.sleep(0.05)  # avoid rate limiting
    return embeddings