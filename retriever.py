"""ChromaDB vector store retrieval for SHL assessments."""

import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from catalog_loader import (
    build_embed_text,
    catalog_by_name,
    item_to_recommendation,
    load_catalog,
    normalize_test_type,
)
from embedder import embed_texts, get_embedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "shl_assessments"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"

VECTOR_CANDIDATES = 40
DEFAULT_TOP_K = 15

TEST_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "A": (
        "cognitive",
        "ability",
        "reasoning",
        "numerical",
        "inductive",
        "deductive",
        "verify",
        "aptitude",
        "general ability",
    ),
    "P": (
        "personality",
        "opq",
        "behaviour",
        "behavior",
        "temperament",
        "motivation",
        "leadership style",
    ),
    "K": (
        "technical",
        "knowledge",
        "skill test",
        "programming",
        "java",
        "python",
        "software",
        "coding",
        ".net",
    ),
    "S": ("situational", "judgment", "judgement", "simulation"),
    "B": ("biodata", "biographical"),
    "C": ("competency", "competencies", "ucf"),
}

REFINE_KEYWORDS = re.compile(
    r"\b(add|remove|only|exclude|without|instead|focus on|prioriti[sz]e|"
    r"remote|personality|technical|cognitive|fewer|more|narrow|broaden|"
    r"update the list|change the list|refine)\b",
    re.IGNORECASE,
)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_catalog: list[dict[str, Any]] | None = None
_name_lookup: dict[str, dict[str, Any]] | None = None


def init_retriever(
    chroma_dir: Path | None = None,
    preload_embedder: bool = True,
) -> None:
    """Initialize Chroma client, collection, and catalog (call once at startup)."""
    global _client, _collection, _catalog, _name_lookup

    if preload_embedder:
        get_embedder()

    _catalog = load_catalog()
    _name_lookup = catalog_by_name(_catalog)

    persist_dir = str(chroma_dir or CHROMA_DIR)
    _client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "Retriever ready: collection '%s' has %d documents",
        COLLECTION_NAME,
        _collection.count(),
    )


def _ensure_ready() -> chromadb.Collection:
    if _collection is None:
        init_retriever()
    assert _collection is not None
    return _collection


def _vector_search(query: str, n_results: int) -> list[dict[str, Any]]:
    collection = _ensure_ready()
    if not query.strip():
        return []

    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection is empty; run build_index.py first")
        return []

    query_embedding = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, count),
        include=["metadatas", "documents", "distances"],
    )

    items: list[dict[str, Any]] = []
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]

    for meta, distance in zip(metadatas[0], distances[0]):
        if not meta:
            continue
        items.append(
            {
                "name": meta.get("name", ""),
                "url": meta.get("url", ""),
                "test_type": meta.get("test_type", "K"),
                "duration_minutes": meta.get("duration_minutes"),
                "remote_testing": meta.get("remote_testing")
                in (True, "True", "true", 1),
                "description": meta.get("description", ""),
                "distance": float(distance),
            }
        )
    return items


def _test_type_boost(query: str, item: dict[str, Any]) -> float:
    q = query.lower()
    boost = 0.0
    item_type = normalize_test_type(str(item.get("test_type", "K")))
    for letter, keywords in TEST_TYPE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            if item_type == letter:
                boost += 0.12
            text = f"{item.get('name', '')} {item.get('description', '')}".lower()
            if any(kw in text for kw in keywords):
                boost += 0.04
    return boost


def _keyword_overlap_boost(query: str, item: dict[str, Any]) -> float:
    q_words = {w for w in re.findall(r"[a-z0-9+#.]{3,}", query.lower())}
    if not q_words:
        return 0.0
    text = f"{item.get('name', '')} {item.get('description', '')}".lower()
    hits = sum(1 for w in q_words if w in text)
    return min(0.25, hits * 0.025)


def _remote_boost(query: str, item: dict[str, Any]) -> float:
    if re.search(r"\bremote\b", query, re.IGNORECASE):
        if item.get("remote_testing"):
            return 0.15
        return -0.05
    return 0.0


def rerank(query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-score vector hits with lexical and constraint boosts."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        dist = item.get("distance", 1.0)
        base = max(0.0, 1.0 - dist)
        score = (
            base
            + _keyword_overlap_boost(query, item)
            + _test_type_boost(query, item)
            + _remote_boost(query, item)
        )
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


def _merge_by_url(lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for items in lists:
        for item in items:
            url = item.get("url", "")
            if not url:
                continue
            existing = merged.get(url)
            if existing is None or item.get("distance", 99) < existing.get(
                "distance", 99
            ):
                merged[url] = item
    return list(merged.values())


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Semantic search with re-ranking."""
    if not query.strip():
        return []
    candidates = _vector_search(query, VECTOR_CANDIDATES)
    ranked = rerank(query, candidates)
    return ranked[:top_k]


def retrieve_merged(queries: list[str], top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Run multiple queries, merge, re-rank on combined text."""
    combined_query = " ".join(q.strip() for q in queries if q.strip())
    if not combined_query:
        return []

    lists: list[list[dict[str, Any]]] = []
    for q in queries:
        if q.strip():
            lists.append(_vector_search(q.strip(), VECTOR_CANDIDATES))

    merged = _merge_by_url(lists)
    ranked = rerank(combined_query, merged)
    return ranked[:top_k]


def recommendations_from_results(
    results: list[dict[str, Any]], limit: int = 10
) -> list[dict[str, str]]:
    """Build canonical recommendations directly from ranked retrieval."""
    recs: list[dict[str, str]] = []
    for item in results[:limit]:
        recs.append(item_to_recommendation(item))
    return recs


def get_by_name(name: str) -> dict[str, Any] | None:
    """Exact or case-insensitive full-name lookup."""
    if _name_lookup is None:
        init_retriever()

    assert _name_lookup is not None
    key = name.strip().lower()
    if key in _name_lookup:
        return _name_lookup[key]

    for catalog_name, item in _name_lookup.items():
        if key in catalog_name or catalog_name in key:
            return item
    return None


def find_names_in_text(text: str, max_names: int = 4) -> list[dict[str, Any]]:
    """Find catalog assessments whose names appear in the given text."""
    if _catalog is None or _name_lookup is None:
        init_retriever()

    assert _catalog is not None
    lowered = text.lower()
    matches: list[tuple[int, dict[str, Any]]] = []

    for item in _catalog:
        name = item["name"]
        name_lower = name.lower()
        if name_lower in lowered:
            matches.append((len(name), item))

    matches.sort(key=lambda x: x[0], reverse=True)
    seen_urls: set[str] = set()
    results: list[dict[str, Any]] = []
    for _, item in matches:
        if item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])
        results.append(item)
        if len(results) >= max_names:
            break
    return results
