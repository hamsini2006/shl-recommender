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

ACRONYMS: dict[str, str] = {
    "opq": "Occupational Personality Questionnaire",
    "gsa": "Global Skills Assessment",
    "ucf": "Universal Competency Framework",
    "verify": "Verify",
}

ROLE_EXPANSION: dict[str, list[str]] = {
    "ui": ["photoshop", "front end", "html", "css", "interface", "web", "design", "user interface"],
    "ux": ["photoshop", "front end", "html", "css", "user interface", "web", "design", "user experience"],
    "design": ["photoshop", "creative", "graphics", "interface", "layout"],
    "designer": ["photoshop", "creative", "graphics", "interface", "layout"],
    "data scientist": ["python", "statistics", "sql", "data science", "machine learning", "r ", "scientist"],
    "data analyst": ["excel", "sql", "tableau", "data analysis", "numbers", "reporting", "analyst"],
    "cloud engineer": ["cloud", "aws", "azure", "gcp", "infrastructure", "devops", "linux", "networking"],
    "cloud": ["cloud", "aws", "azure", "gcp", "infrastructure", "linux", "networking", "devops"],
    "devops": ["docker", "kubernetes", "jenkins", "aws", "linux", "automation", "cloud", "devops", "ci cd"],
    "security engineer": ["cybersecurity", "information security", "network security", "linux", "risk"],
    "security": ["cybersecurity", "information security", "network security", "linux", "risk"],
    "backend": ["java", "python", "node.js", "c#", "sql", "api", "server", "architecture"],
    "frontend": ["javascript", "reactjs", "angular", "html", "css", "front end", "web"],
    "hr": ["personality", "behavior", "leadership", "competency", "recruitment", "interpersonal", "hr"],
    "sales": ["sales", "negotiation", "customer", "influence", "selling"],
    "marketing": ["marketing", "digital", "branding", "social media", "advertising", "market"],
    "accountant": ["accounting", "financial", "excel", "bookkeeping", "tax", "audit", "accountant"],
    "finance": ["finance", "accounting", "financial", "excel", "economics"],
    "manager": ["leadership", "management", "supervisor", "team", "planning", "strategy", "manager"],
    "lead": ["leadership", "management", "supervisor", "team", "lead"],
    "project manager": ["project management", "pjm", "planning", "scheduling", "coordination"],
    "customer support": ["customer", "service", "communication", "interpersonal", "support"],
    "ai": ["python", "data science", "machine learning", "ai skills", "algorithms"],
    "ml": ["python", "data science", "machine learning", "algorithms"],
    "developer": ["programming", "software", "coding", "development", "engineer", "developing"],
    "engineer": ["programming", "software", "coding", "development", "engineer", "engineering"],
}

DOMAIN_CONFLICTS: dict[str, list[str]] = {
    "software": ["automotive", "civil", "mechanical", "polymer", "chemical", "petroleum", "metallurgical", "ceramic", "aeronautical", "mechatronics", "sales", "retail", "nursing", "customer service", "aerospace", "geoinformatics", "hardware", "electrical", "biology", "healthcare", "data entry"],
    "ai": ["automotive", "civil", "mechanical", "polymer", "chemical", "petroleum", "metallurgical", "ceramic", "aeronautical", "mechatronics", "sales", "retail", "nursing", "customer service", "technical support", "data entry", "aerospace", "geoinformatics"],
    "ml": ["automotive", "civil", "mechanical", "polymer", "chemical", "petroleum", "metallurgical", "ceramic", "aeronautical", "mechatronics", "sales", "retail", "nursing", "customer service", "technical support", "data entry", "aerospace", "geoinformatics"],
    "it": ["automotive", "civil", "mechanical", "polymer", "chemical", "petroleum", "metallurgical", "ceramic", "aeronautical", "mechatronics", "sales", "retail", "nursing", "customer service", "aerospace", "geoinformatics"],
    "developer": ["automotive", "civil", "mechanical", "polymer", "chemical", "petroleum", "metallurgical", "ceramic", "aeronautical", "mechatronics", "sales", "retail", "nursing", "customer service", "aerospace", "geoinformatics"],
    "java": ["javascript", "automotive", "civil", "mechanical", "polymer", "chemical", "petroleum", "metallurgical", "ceramic", "aeronautical", "mechatronics", "sales", "retail", "nursing", "customer service", "aerospace", "geoinformatics"],
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
    q_lower = query.lower()
    # Explicit intent detection for Technical vs Behavioral
    is_tech_q = any(
        w in q_lower
        for w in [
            "technical", "skill", "programming", "knowledge", "expert", "coding",
            "software", "engineer", "developer", "scientist", "analyst", "ai", "ml", "devops"
        ]
    )
    is_beh_q = any(
        w in q_lower 
        for w in ["personality", "behavior", "leadership", "competency", "soft skills", "interpersonal", "manager"]
    )
    
    test_type = item.get("test_type_normalized") or normalize_test_type(str(item.get("test_type", "K")))
    
    boost = 0.0
    if is_tech_q and test_type in ["K", "S"]:
        boost += 0.6  # High priority for technical tests in tech roles
    if is_beh_q and test_type == "P":
        boost += 0.6  # High priority for personality tests in behavioral queries
        
    # Generic keyword match from mapping
    for letter, keywords in TEST_TYPE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            if test_type == letter:
                boost += 0.2
    return boost


def _keyword_overlap_boost(query: str, item: dict[str, Any]) -> float:
    q_words = set(re.findall(r"\b\w+\b", query.lower()))
    if not q_words:
        return 0.0

    # Expand query with role-specific keywords
    role_terms = set()
    query_lower = query.lower()
    for role, terms in ROLE_EXPANSION.items():
        if role in query_lower:
            role_terms.update(terms)

    text = f"{item.get('name', '')} {item.get('description', '')}".lower()

    # Boost for exact matches in name
    name_lower = item.get("name", "").lower()
    name_words = set(re.findall(r"\b\w+\b", name_lower))
    name_boost = 0.0
    for word in q_words:
        if word in name_words:
            # Penalize generic keywords in name boost if they are alone
            if word in ["entry", "level", "senior", "junior", "solution", "new"]:
                name_boost += 0.2
            elif word in ["java", "python", "sql", "c++", "c#", "aws", "react", "angular", "node", "ruby", "php"]:
                name_boost += 2.5  # Huge boost for specific technology stack match
            else:
                name_boost += 0.5

    # Boost for role-specific terms (HIGHER WEIGHT)
    role_hits = sum(1 for w in role_terms if w in text)
    role_boost = min(2.0, role_hits * 1.0)  # 1.0 per hit!

    # Boost for original query words in text (LOWER WEIGHT for generic words)
    # Exclude very common filler/seniority words from this part
    filtered_q_words = q_words - {"entry", "level", "senior", "junior", "hiring", "need", "assessment", "test", "role", "position", "solution", "shl"}
    generic_hits = sum(1 for w in filtered_q_words if w in text)
    generic_boost = min(0.8, generic_hits * 0.2)

    return name_boost + role_boost + generic_boost


def _seniority_boost(query: str, item: dict[str, Any]) -> float:
    q_lower = query.lower()
    desc = item.get("description", "").lower()
    
    is_senior_q = any(w in q_lower for w in ["senior", "lead", "expert", "manager", "director"])
    is_junior_q = any(w in q_lower for w in ["junior", "entry", "fresh", "graduate"])
    
    if is_senior_q:
        if "professional" in desc or "manager" in desc or "mid-professional" in desc:
            return 0.3
        if "entry-level" in desc or "graduate" in desc:
            return -0.2
    elif is_junior_q:
        if "entry-level" in desc or "graduate" in desc:
            return 0.3
        if "professional" in desc or "manager" in desc:
            return -0.1
            
    return 0.0


def _remote_boost(query: str, item: dict[str, Any]) -> float:
    if re.search(r"\bremote\b", query, re.IGNORECASE):
        if item.get("remote_testing"):
            return 0.15
        return -0.05
    return 0.0


def _domain_penalty(query: str, item: dict[str, Any]) -> float:
    q_lower = query.lower()
    name_lower = item.get("name", "").lower()
    penalty = 0.0
    for trigger, forbidden_list in DOMAIN_CONFLICTS.items():
        if re.search(r"\b" + re.escape(trigger) + r"\b", q_lower):
            for forbidden in forbidden_list:
                if forbidden in name_lower:
                    penalty -= 2.0  # Heavier penalty for domain mismatch

    # Special case: 'engineer' in query matching generic 'Engineering' tests
    if re.search(r"\b(engineer|software|ai|ml|java)\b", q_lower):
        if "engineering" in name_lower or "engineer" in name_lower:
            # If it's a tech role, but the test isn't explicitly software/data/front-end/java
            if not any(w in name_lower for w in ["software", "front end", "web", "data", "ai", "java"]):
                penalty -= 1.5  # Discourage generic industrial/civil engineering
                
    # Special case for Javascript when Java is asked for
    if re.search(r"\bjava\b", q_lower) and not re.search(r"\bjavascript\b", q_lower):
        if "javascript" in name_lower or "node" in name_lower:
            penalty -= 3.0
            
    # Penalize non-software engineering fields if software engineering is requested
    if "software" in q_lower or "developer" in q_lower or "programmer" in q_lower:
        if any(w in name_lower for w in ["aerospace", "geoinformatics", "mechanical", "civil", "electrical"]):
            penalty -= 3.0

    return penalty


def rerank(query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score and sort items based on several heuristic boosts."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        dist = item.get("distance", 1.0)
        base = max(0.0, 1.0 - dist)
        score = (
            base
            + _keyword_overlap_boost(query, item)
            + _test_type_boost(query, item)
            + _remote_boost(query, item)
            + _seniority_boost(query, item)
            + _domain_penalty(query, item)
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
    # Use the full context query (the last one) for reranking
    context_q = queries[-1] if queries else combined_query
    ranked = rerank(context_q, merged)
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
        # Cleaned catalog name for easier partial matching
        clean_name = re.sub(r'\s*\([^)]*\)', '', name_lower).strip()
        
        # Check for exact name, cleaned name, or if user mentioned an acronym for this item
        is_match = False
        if name_lower in lowered or (len(clean_name) > 3 and clean_name in lowered):
            is_match = True
        else:
            # Check acronyms
            for acr, full in ACRONYMS.items():
                if acr in re.findall(r"\b\w+\b", lowered) and full.lower() in name_lower:
                    is_match = True
                    break
        
        if is_match:
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
