"""One-time script to embed catalog.json into ChromaDB (idempotent)."""

import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

from catalog_loader import build_embed_text, load_catalog, normalize_test_type
from embedder import embed_texts, get_embedder

COLLECTION_NAME = "shl_assessments"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
BATCH_SIZE = 20  # smaller batch — Gemini API rate limit friendly


def build_index() -> int:
    catalog = load_catalog()
    print(f"Loaded {len(catalog)} assessments from catalog.json")

    get_embedder()  # just configures API key

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Removed existing collection '{COLLECTION_NAME}' for rebuild")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(catalog)
    for start in range(0, total, BATCH_SIZE):
        batch = catalog[start : start + BATCH_SIZE]
        texts = [build_embed_text(item) for item in batch]

        print(f"Embedding batch {start}-{min(start+BATCH_SIZE, total)}...")
        embeddings = embed_texts(texts)

        ids = [f"assessment_{start + i}" for i in range(len(batch))]

        metadatas = []
        for item in batch:
            duration = item.get("duration_minutes")
            metadatas.append(
                {
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": normalize_test_type(str(item["test_type"])),
                    "duration_minutes": int(duration) if duration is not None else -1,
                    "remote_testing": bool(item.get("remote_testing", False)),
                    "description": item["description"][:8000],
                }
            )

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        done = min(start + BATCH_SIZE, total)
        print(f"Indexed {done}/{total} assessments")
        time.sleep(1)  # pause between batches for rate limit

    count = collection.count()
    print(f"Done. ChromaDB collection '{COLLECTION_NAME}' contains {count} assessments.")
    return count


if __name__ == "__main__":
    try:
        build_index()
    except Exception as exc:
        print(f"build_index failed: {exc}", file=sys.stderr)
        sys.exit(1)