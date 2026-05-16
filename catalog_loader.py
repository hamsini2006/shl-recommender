"""Load and validate catalog.json; provide lookup helpers."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"

VALID_TEST_TYPES = frozenset({"A", "P", "B", "K", "S", "C"})

TEST_TYPE_FULL_NAMES = {
    "A": "Ability",
    "P": "Personality",
    "B": "Biodata",
    "K": "Knowledge",
    "S": "Situational Judgment",
    "C": "Competency",
}

REQUIRED_FIELDS = ("name", "url", "test_type", "description")


def normalize_test_type(raw: str) -> str:
    """Return a single valid test-type letter from catalog test_type string."""
    if not raw:
        return "K"
    cleaned = raw.replace(" ", "").upper()
    for char in cleaned:
        if char in VALID_TEST_TYPES:
            return char
    if cleaned:
        return cleaned[0] if cleaned[0] in VALID_TEST_TYPES else "K"
    return "K"


def test_type_full_name(test_type_letter: str) -> str:
    return TEST_TYPE_FULL_NAMES.get(test_type_letter.upper(), test_type_letter)


def _validate_item(item: dict[str, Any], index: int) -> None:
    for field in REQUIRED_FIELDS:
        if field not in item or item[field] is None:
            raise ValueError(f"Catalog item at index {index} missing field: {field}")
        if field != "description" and str(item[field]).strip() == "":
            raise ValueError(f"Catalog item at index {index} missing or empty field: {field}")
    if not str(item.get("description", "")).strip():
        item["description"] = f"SHL assessment: {item['name']}"
    if not str(item["url"]).startswith("https://www.shl.com/"):
        raise ValueError(
            f"Catalog item at index {index} has invalid URL: {item.get('url')}"
        )


def load_catalog(path: Path | None = None) -> list[dict[str, Any]]:
    """Load catalog.json and validate every entry."""
    catalog_path = path or CATALOG_PATH
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(catalog_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("catalog.json must be a JSON array")

    normalized: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Catalog item at index {i} is not an object")
        _validate_item(item, i)
        entry = dict(item)
        entry["test_type_normalized"] = normalize_test_type(str(item["test_type"]))
        normalized.append(entry)

    logger.info("Loaded %d assessments from catalog", len(normalized))
    return normalized


def get_valid_urls(catalog: list[dict[str, Any]] | None = None) -> set[str]:
    items = catalog if catalog is not None else load_catalog()
    return {item["url"] for item in items}


def build_embed_text(item: dict[str, Any]) -> str:
    letter = item.get("test_type_normalized") or normalize_test_type(str(item["test_type"]))
    full_name = test_type_full_name(letter)
    return f"{item['name']} | {full_name} | {item['description']}"


def catalog_by_name(catalog: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    items = catalog if catalog is not None else load_catalog()
    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        lookup[item["name"].lower()] = item
    return lookup


def catalog_by_url(catalog: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    items = catalog if catalog is not None else load_catalog()
    return {item["url"]: item for item in items}


def item_to_recommendation(item: dict[str, Any]) -> dict[str, str]:
    """Canonical recommendation dict from a catalog entry."""
    letter = item.get("test_type_normalized") or normalize_test_type(str(item["test_type"]))
    return {
        "name": item["name"],
        "url": item["url"],
        "test_type": letter,
    }
