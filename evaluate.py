"""
Evaluation harness for the SHL Assessment Recommender.

Measures:
  - Retrieval Recall@K (vector search + re-ranking)
  - Agent recommendation Recall@K (final shortlist vs labeled relevant URLs)
  - Catalog validity (all recommendation URLs in catalog.json)
  - Name groundedness (recommendation name matches canonical catalog name for URL)
  - Schema compliance (response shape)
  - Behavior probe pass rate (vague, refuse, compare, refine, etc.)

Usage:
  python evaluate.py                  # retrieval + agent (needs GEMINI_API_KEY for agent)
  python evaluate.py --retrieval-only # fast; no LLM
  python evaluate.py --output report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import agent
import retriever
from agent import build_full_user_query, build_search_query
from catalog_loader import catalog_by_url, load_catalog

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_PATH = Path(__file__).resolve().parent / "eval_cases.json"
DEFAULT_K = 10


@dataclass
class CaseResult:
    case_id: str
    case_type: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    retrieval_recall_at_k: float = 0.0
    agent_recall_at_k: float = 0.0
    catalog_validity_rate: float = 0.0
    name_groundedness_rate: float = 0.0
    schema_compliance_rate: float = 0.0
    behavior_probe_pass_rate: float = 0.0
    case_results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_recall_at_k": round(self.retrieval_recall_at_k, 4),
            "agent_recall_at_k": round(self.agent_recall_at_k, 4),
            "catalog_validity_rate": round(self.catalog_validity_rate, 4),
            "name_groundedness_rate": round(self.name_groundedness_rate, 4),
            "schema_compliance_rate": round(self.schema_compliance_rate, 4),
            "behavior_probe_pass_rate": round(self.behavior_probe_pass_rate, 4),
            "cases": [
                {
                    "id": c.case_id,
                    "type": c.case_type,
                    "passed": c.passed,
                    **c.details,
                }
                for c in self.case_results
            ],
        }


def load_eval_cases(path: Path = EVAL_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = DEFAULT_K) -> float:
    """Fraction of relevant items present in top-K retrieved URLs."""
    if not relevant:
        return 1.0
    top = retrieved[:k]
    hits = sum(1 for url in top if url in relevant)
    return hits / len(relevant)


def check_schema(response: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key in ("reply", "recommendations", "end_of_conversation"):
        if key not in response:
            errors.append(f"missing key: {key}")
    if "reply" in response and not isinstance(response["reply"], str):
        errors.append("reply must be string")
    recs = response.get("recommendations")
    if recs is None:
        errors.append("recommendations must not be null")
    elif not isinstance(recs, list):
        errors.append("recommendations must be list")
    elif len(recs) > 10:
        errors.append("recommendations exceeds 10 items")
    else:
        for i, rec in enumerate(recs):
            if not isinstance(rec, dict):
                errors.append(f"recommendation[{i}] not an object")
                continue
            for field_name in ("name", "url", "test_type"):
                if field_name not in rec:
                    errors.append(f"recommendation[{i}] missing {field_name}")
    if "end_of_conversation" in response and not isinstance(
        response["end_of_conversation"], bool
    ):
        errors.append("end_of_conversation must be bool")
    return len(errors) == 0, errors


def catalog_validity(
    recommendations: list[dict[str, Any]], url_map: dict[str, dict[str, Any]]
) -> float:
    if not recommendations:
        return 1.0
    valid = sum(1 for r in recommendations if r.get("url") in url_map)
    return valid / len(recommendations)


def name_groundedness(
    recommendations: list[dict[str, Any]], url_map: dict[str, dict[str, Any]]
) -> float:
    if not recommendations:
        return 1.0
    grounded = 0
    for rec in recommendations:
        url = rec.get("url", "")
        item = url_map.get(url)
        if item and rec.get("name", "").strip() == item["name"]:
            grounded += 1
    return grounded / len(recommendations)


def run_retrieval_eval(
    case: dict[str, Any], k: int = DEFAULT_K
) -> CaseResult:
    query = case.get("search_query", "")
    if not query and case.get("messages"):
        msgs = case["messages"]
        query = build_search_query(msgs)
        full = build_full_user_query(msgs)
        results = retriever.retrieve_merged([query, full], top_k=k) if full else retriever.retrieve(query, top_k=k)
    else:
        results = retriever.retrieve(query, top_k=k)

    retrieved_urls = [r["url"] for r in results]
    relevant = set(case.get("relevant_urls", []))
    score = recall_at_k(retrieved_urls, relevant, k=k)

    return CaseResult(
        case_id=case["id"],
        case_type="retrieval_only",
        passed=score >= 0.5 if relevant else True,
        details={
            "recall_at_k": round(score, 4),
            "k": k,
            "retrieved_count": len(retrieved_urls),
            "relevant_count": len(relevant),
            "hits": [u for u in retrieved_urls[:k] if u in relevant],
        },
    )


def run_behavior_probe(
    case: dict[str, Any], response: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    probe = case.get("probe", "")
    recs = response.get("recommendations") or []
    reply = (response.get("reply") or "").lower()

    if probe == "no_recommendations_turn1_vague":
        ok = len(recs) == 0 and len(reply) > 10
        return ok, {"recommendation_count": len(recs)}

    if probe == "refuse_off_topic":
        ok = (
            len(recs) == 0
            and ("shl" in reply or "assessment" in reply)
            and ("salary" not in reply[:80] or "only" in reply or "can't" in reply or "cannot" in reply)
        )
        return ok, {"recommendation_count": len(recs)}

    if probe == "compare_no_recommendations":
        ok = len(recs) == 0
        return ok, {"recommendation_count": len(recs)}

    if probe == "has_recommendations":
        ok = 1 <= len(recs) <= 10
        return ok, {"recommendation_count": len(recs)}

    if probe == "refine_updates_shortlist":
        ok = 1 <= len(recs) <= 10
        types = {r.get("test_type", "").upper() for r in recs}
        has_personality = "P" in types
        return ok and has_personality, {
            "recommendation_count": len(recs),
            "test_types": sorted(types),
            "has_personality": has_personality,
        }

    return True, {"probe": probe or "none"}


def run_agent_eval(
    case: dict[str, Any],
    url_map: dict[str, dict[str, Any]],
    k: int = DEFAULT_K,
) -> CaseResult:
    messages = case.get("messages", [])
    response = agent.run(messages)

    schema_ok, schema_errors = check_schema(response)
    recs = response.get("recommendations") or []
    rec_urls = [r.get("url", "") for r in recs if isinstance(r, dict)]
    relevant = set(case.get("relevant_urls", []))
    recall = recall_at_k(rec_urls, relevant, k=k) if relevant else None

    cat_val = catalog_validity(recs, url_map)
    name_gr = name_groundedness(recs, url_map)

    behavior_ok = True
    behavior_details: dict[str, Any] = {}
    if case.get("probe"):
        behavior_ok, behavior_details = run_behavior_probe(case, response)

    passed = schema_ok and cat_val == 1.0 and name_gr == 1.0 and behavior_ok
    if recall is not None and case.get("type") == "recommendation":
        passed = passed and recall >= 0.2

    return CaseResult(
        case_id=case["id"],
        case_type=case.get("type", "agent"),
        passed=passed,
        details={
            "schema_ok": schema_ok,
            "schema_errors": schema_errors,
            "recall_at_k": round(recall, 4) if recall is not None else None,
            "catalog_validity": round(cat_val, 4),
            "name_groundedness": round(name_gr, 4),
            "behavior_probe": case.get("probe"),
            "behavior_passed": behavior_ok,
            "behavior_details": behavior_details,
            "recommendation_count": len(recs),
            "reply_preview": (response.get("reply") or "")[:120],
        },
    )


def run_evaluation(
    retrieval_only: bool = False,
    k: int = DEFAULT_K,
) -> EvalReport:
    retriever.init_retriever()
    url_map = catalog_by_url(load_catalog())
    cases = load_eval_cases()
    report = EvalReport()

    retrieval_scores: list[float] = []
    agent_recall_scores: list[float] = []
    catalog_scores: list[float] = []
    grounded_scores: list[float] = []
    schema_scores: list[float] = []
    behavior_results: list[bool] = []

    for case in cases:
        if case.get("type") == "retrieval_only":
            result = run_retrieval_eval(case, k=k)
            report.case_results.append(result)
            retrieval_scores.append(result.details.get("recall_at_k", 0.0))
            continue

        if retrieval_only:
            msgs = case.get("messages", [])
            pseudo = {
                **case,
                "type": "retrieval_only",
                "search_query": build_search_query(msgs) if msgs else "",
            }
            result = run_retrieval_eval(pseudo, k=k)
            result.case_id = case["id"]
            result.details["note"] = "agent skipped (--retrieval-only)"
            report.case_results.append(result)
            retrieval_scores.append(result.details.get("recall_at_k", 0.0))
            continue

        result = run_agent_eval(case, url_map, k=k)
        report.case_results.append(result)

        if result.details.get("recall_at_k") is not None:
            agent_recall_scores.append(result.details["recall_at_k"])
        catalog_scores.append(result.details.get("catalog_validity", 0.0))
        grounded_scores.append(result.details.get("name_groundedness", 0.0))
        schema_scores.append(1.0 if result.details.get("schema_ok") else 0.0)
        if case.get("probe"):
            behavior_results.append(result.details.get("behavior_passed", False))

    if retrieval_scores:
        report.retrieval_recall_at_k = sum(retrieval_scores) / len(retrieval_scores)
    if agent_recall_scores:
        report.agent_recall_at_k = sum(agent_recall_scores) / len(agent_recall_scores)
    if catalog_scores:
        report.catalog_validity_rate = sum(catalog_scores) / len(catalog_scores)
    if grounded_scores:
        report.name_groundedness_rate = sum(grounded_scores) / len(grounded_scores)
    if schema_scores:
        report.schema_compliance_rate = sum(schema_scores) / len(schema_scores)
    if behavior_results:
        report.behavior_probe_pass_rate = sum(behavior_results) / len(behavior_results)

    return report


def print_report(report: EvalReport) -> None:
    d = report.to_dict()
    print("\n=== SHL Recommender Evaluation Report ===\n")
    print(f"  Retrieval Recall@{DEFAULT_K} (mean):  {d['retrieval_recall_at_k']:.2%}")
    print(f"  Agent Recall@{DEFAULT_K} (mean):       {d['agent_recall_at_k']:.2%}")
    print(f"  Catalog validity rate:               {d['catalog_validity_rate']:.2%}")
    print(f"  Name groundedness rate:              {d['name_groundedness_rate']:.2%}")
    print(f"  Schema compliance rate:              {d['schema_compliance_rate']:.2%}")
    print(f"  Behavior probe pass rate:            {d['behavior_probe_pass_rate']:.2%}")
    print("\n--- Per-case results ---\n")
    for case in d["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        print(f"  [{status}] {case['id']} ({case['type']})")
        for key, val in case.items():
            if key in ("id", "type", "passed"):
                continue
            print(f"         {key}: {val}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SHL Assessment Recommender")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip agent/LLM calls; evaluate retrieval only",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"K for Recall@K (default {DEFAULT_K})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Write JSON report to this path",
    )
    args = parser.parse_args()

    try:
        report = run_evaluation(retrieval_only=args.retrieval_only, k=args.k)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Evaluation failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(report)

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report written to {out_path}")

    failed = sum(1 for c in report.case_results if not c.passed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
