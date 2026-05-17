"""Core conversational agent logic for SHL assessment recommendations."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

import google.generativeai as genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

import retriever
from catalog_loader import catalog_by_url, item_to_recommendation, load_catalog, normalize_test_type
from prompts import COMPARE_SUPPLEMENT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 25
MAX_USER_TURNS = 8  # Total 16 turns (8 user + 8 assistant)
MAX_TOTAL_TURNS = 16

FALLBACK_RESPONSE: dict[str, Any] = {
    "reply": "I'm having trouble right now. Please try again.",
    "recommendations": [],
    "end_of_conversation": False,
}

PARSE_FALLBACK_RESPONSE: dict[str, Any] = {
    "reply": "I had trouble processing that response. Please try again.",
    "recommendations": [],
    "end_of_conversation": False,
}

COMPARE_KEYWORDS = re.compile(
    r"\b(difference|differences|compare|comparison|versus|vs\.?|compared to|"
    r"how (?:is|are).+different|what.+different|between .+ and)\b",
    re.IGNORECASE,
)

ROLE_SKILL_HINTS = re.compile(
    r"\b(developer|engineer|manager|analyst|sales|customer|nurse|teacher|"
    r"accountant|designer|leader|executive|graduate|entry[\s-]?level|"
    r"senior|junior|technical|personality|cognitive|ability|skills?|"
    r"programming|software|java|python|\.net|finance|marketing|hire|hiring|"
    r"role|position|job|candidate|remote|knowledge|situational|nursing|"
    r"data|scientist|hr|accounting|audit|retail|"
    r"ui|ux|design|graphic|creative|front-?end|"
    r"ai|ml|machine.learning|cloud|devops|security|infrastructure|"
    r"administrator|consultant|specialist|coordinator|supervisor|director)\b",
    re.IGNORECASE,
)

VAGUE_PHRASES = re.compile(
    r"^(hi|hello|hey|help|help me|help us|i need an assessment|i need a test|"
    r"i need help|need an assessment|need a test|assessment|test|something|"
    r"recommend something|what do you suggest|not sure what i need)\s*[.!?]*$",
    re.IGNORECASE,
)

VAGUE_CONTAINS = re.compile(
    r"\b(i need an assessment|i need a test|help me (?:find|choose|pick)|"
    r"looking for an assessment|find me a test|recommend an assessment|"
    r"what assessment should|which assessment should|help me hire someone|"
    r"need help hiring|assessment for hiring)\b",
    re.IGNORECASE,
)

SENIORITY_HINTS = re.compile(
    r"\b(senior|junior|mid[\s-]?level|entry[\s-]?level|executive|graduate|"
    r"manager|director|supervisor|experienced|lead)\b",
    re.IGNORECASE,
)

TEST_PREF_HINTS = re.compile(
    r"\b(personality|cognitive|ability|technical|knowledge|situational|"
    r"remote|opq|verify|numerical|inductive|java|python|software|sales|"
    r"leadership|competency|biodata)\b",
    re.IGNORECASE,
)

OFF_TOPIC_PATTERNS = re.compile(
    r"\b(salary|pay scale|how much (?:should|to) pay|legal(?:ly)?|"
    r"criminal record|lawsuit|ignore (?:your|all) instructions|"
    r"forget (?:your|the) (?:rules|instructions)|jailbreak|"
    r"write (?:me )?code|recipe|weather|stock price|medical advice|"
    r"visa|immigration law)\b",
    re.IGNORECASE,
)

REFINE_SIGNAL = re.compile(
    r"\b(add|remove|only|exclude|without|instead|focus on|prioriti[sz]e|"
    r"remote|personality|technical|cognitive|update the list|change the list|"
    r"refine|narrow|drop the|leave out|also include)\b",
    re.IGNORECASE,
)

_catalog_by_url: dict[str, dict[str, Any]] | None = None


def _get_catalog_by_url() -> dict[str, dict[str, Any]]:
    global _catalog_by_url
    if _catalog_by_url is None:
        _catalog_by_url = catalog_by_url(load_catalog())
    return _catalog_by_url


def count_user_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def build_search_query(messages: list[dict]) -> str:
    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    last_three = user_contents[-3:] if user_contents else []
    return " ".join(last_three).strip()


def build_full_user_query(messages: list[dict]) -> str:
    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    return " ".join(user_contents).strip()


def detect_compare_intent(messages: list[dict]) -> bool:
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        return False
    last = user_messages[-1]
    if not COMPARE_KEYWORDS.search(last):
        return False
    found = retriever.find_names_in_text(last, max_names=2)
    return len(found) >= 2 or (
        len(found) >= 1
        and bool(re.search(r"\b(and|versus|vs\.?|between)\b", last, re.IGNORECASE))
    )


def detect_off_topic(messages: list[dict]) -> bool:
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        return False
    return bool(OFF_TOPIC_PATTERNS.search(user_messages[-1]))


def has_recommendation_context(messages: list[dict]) -> bool:
    """True when role/skill plus at least one other hiring dimension is present."""
    text = " ".join(m["content"] for m in messages if m.get("role") == "user")
    if not text.strip():
        return False
    has_role = bool(ROLE_SKILL_HINTS.search(text))
    has_extra = bool(
        SENIORITY_HINTS.search(text)
        or TEST_PREF_HINTS.search(text)
        or len(text) > 140
    )
    return has_role and has_extra


def is_first_user_turn(messages: list[dict]) -> bool:
    return count_user_turns(messages) == 1


def query_is_vague(messages: list[dict]) -> bool:
    """Strict vague detection for first user message only."""
    if not is_first_user_turn(messages):
        return False

    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    text = user_messages[-1].strip()
    if not text:
        return True

    if VAGUE_PHRASES.match(text):
        return True
    if VAGUE_CONTAINS.search(text) and not has_recommendation_context(messages):
        return True
    if len(text) < 50 and not ROLE_SKILL_HINTS.search(text):
        return True
    if len(text) < 90 and not has_recommendation_context(messages):
        return True
    return False


def detect_role_switch(messages: list[dict]) -> bool:
    if len(messages) < 2:
        return False
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if len(user_messages) < 2:
        return False
    last = user_messages[-1].lower()
    # Check if the last message mentions a role
    last_roles = set(ROLE_SKILL_HINTS.findall(last))
    if not last_roles:
        return False
    # Check all previous user messages for these roles
    previous_text = " ".join(user_messages[:-1]).lower()
    prev_roles = set(ROLE_SKILL_HINTS.findall(previous_text))
    # If a NEW role is introduced that wasn't mentioned before
    return bool(last_roles - prev_roles)


def detect_refine_intent(messages: list[dict]) -> bool:
    if count_user_turns(messages) < 2:
        return False
    if detect_role_switch(messages):
        return False
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        return False
    last = user_messages[-1]
    had_shortlist = any(
        m.get("role") == "assistant"
        and (
            "recommend" in m.get("content", "").lower()
            or "shortlist" in m.get("content", "").lower()
            or "here are" in m.get("content", "").lower()
        )
        for m in messages
    )
    return bool(REFINE_SIGNAL.search(last)) or (had_shortlist and len(last) < 200)


def format_history(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_catalog_for_prompt(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(No assessments retrieved — ask the user to clarify their hiring need.)"
    blocks: list[str] = []
    for i, item in enumerate(results, 1):
        duration = item.get("duration_minutes")
        duration_str = (
            str(duration) if duration is not None and duration != -1 else "unknown"
        )
        remote = item.get("remote_testing", False)
        desc = item.get("description", "")
        blocks.append(
            f"{i}. name: {item['name']}\n"
            f"   url: {item['url']}\n"
            f"   test_type: {item.get('test_type', 'K')}\n"
            f"   duration_minutes: {duration_str}\n"
            f"   remote_testing: {remote}\n"
            f"   description: {desc[:1200]}"
        )
    return "\n\n".join(blocks)


def _format_assessment_block(item: dict[str, Any]) -> str:
    letter = normalize_test_type(str(item.get("test_type", "K")))
    return (
        f"name: {item['name']}\n"
        f"url: {item['url']}\n"
        f"test_type: {letter}\n"
        f"duration_minutes: {item.get('duration_minutes')}\n"
        f"remote_testing: {item.get('remote_testing')}\n"
        f"description: {item.get('description', '')}"
    )


def build_compare_context(messages: list[dict]) -> str:
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        return "(No user message for comparison.)"

    last = user_messages[-1]
    found = retriever.find_names_in_text(last, max_names=4)

    if len(found) < 2:
        for part in re.split(r"\b(?:and|versus|vs\.?|,)\b", last, flags=re.IGNORECASE):
            part = part.strip(" ?.")
            if len(part) < 3:
                continue
            item = retriever.get_by_name(part)
            if item and item["url"] not in {f["url"] for f in found}:
                found.append(item)

    if not found:
        return (
            "COMPARE MODE: No matching assessments found in catalog for the names mentioned. "
            "Tell the user you do not have that assessment in your catalog. recommendations: []"
        )

    blocks = [_format_assessment_block(item) for item in found[:4]]
    compare_blocks = "\n\n---\n\n".join(blocks)
    return COMPARE_SUPPLEMENT.format(compare_blocks=compare_blocks)


def build_refuse_response() -> dict[str, Any]:
    return {
        "reply": (
            "I can only help with selecting SHL assessments from our product catalog. "
            "I'm not able to assist with salary, legal, or other HR compliance topics. "
            "Tell me about the role and skills you want to assess, and I'll recommend "
            "relevant SHL tests."
        ),
        "recommendations": [],
        "end_of_conversation": False,
    }


def build_clarify_response() -> dict[str, Any]:
    return {
        "reply": (
            "I'd be happy to help you find SHL assessments. What role or skills are you "
            "hiring for? For example: seniority level, and whether you need cognitive "
            "ability, personality, technical knowledge, or situational judgment tests."
        ),
        "recommendations": [],
        "end_of_conversation": False,
    }


def build_turn_cap_response(results: list[dict[str, Any]]) -> dict[str, Any]:
    recs = retriever.recommendations_from_results(results, limit=5)
    if recs:
        return {
            "reply": (
                "We've reached the maximum of 8 turns for this conversation. "
                "Here is your final shortlist of SHL assessments based on our discussion."
            ),
            "recommendations": recs,
            "end_of_conversation": True,
        }
    return {
        "reply": (
            "We've reached the maximum of 8 turns for this conversation. "
            "Please start a new chat if you need further SHL assessment guidance."
        ),
        "recommendations": [],
        "end_of_conversation": True,
    }


# Confirmed working models (ordered: best → fallback)
MODELS_TO_TRY = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]


def _get_api_keys() -> list[str]:
    """Return all configured API keys, filtering out blanks/duplicates."""
    keys: list[str] = []
    seen: set[str] = set()
    for env_var in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        k = os.environ.get(env_var, "").strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


def _call_llm_sync(prompt: str) -> str:
    """Attempt to generate a response using available API keys and model fallbacks.
    The function cycles through GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3 and, for each key,
    tries the models listed in MODELS_TO_TRY. On a 429 (quota exhausted) it moves to the next key.
    On a 404 (model not found) it tries the next model on the same key.
    Returns the generated JSON string or raises the last encountered exception.
    """
    api_keys = _get_api_keys()
    if not api_keys:
        raise RuntimeError("No GEMINI_API_KEY environment variables are set.")

    last_error: Exception | None = None
    for api_key in api_keys:
        if api_key.startswith("gsk_"):
            # Transparently support Groq keys as fallback via REST API
            try:
                import requests
                logger.info("Trying Groq fallback with key=***%s", api_key[-6:])
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                # Llama 3 70B is highly capable and supports JSON mode
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                }
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=20,
                )
                if response.status_code == 200:
                    res_json = response.json()
                    text = res_json["choices"][0]["message"]["content"]
                    logger.info("Groq success — model=llama-3.3-70b-versatile")
                    return text or ""
                else:
                    logger.warning(
                        "Groq API failed with status %s: %s",
                        response.status_code,
                        response.text[:200],
                    )
                    last_error = RuntimeError(f"Groq API failed: {response.status_code}")
                    continue
            except Exception as exc:
                logger.warning("Groq call failed: %s", exc)
                last_error = exc
                continue
        else:
            # Configure the client with the current key
            genai.configure(api_key=api_key)
            for model_id in MODELS_TO_TRY:
                try:
                    model = genai.GenerativeModel(model_id)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                    logger.info(
                        "LLM success – key=***%s model=%s",
                        api_key[-6:],
                        model_id,
                    )
                    return response.text or ""
                except google_exceptions.ResourceExhausted as exc:
                    last_error = exc
                    logger.warning(
                        "Quota exhausted for key=***%s model=%s – switching key",
                        api_key[-6:],
                        model_id,
                    )
                    # Break to try next API key
                    break
                except google_exceptions.NotFound as exc:
                    last_error = exc
                    logger.warning(
                        "Model %s not found for key=***%s – trying next model",
                        model_id,
                        api_key[-6:],
                    )
                    continue
                except Exception as exc:
                    logger.exception(
                        "Unexpected error with key=***%s model=%s",
                        api_key[-6:],
                        model_id,
                    )
                    raise
    # If we exit loops without returning, all attempts failed
    raise last_error or RuntimeError("All Gemini models and keys exhausted.")




def call_llm_with_timeout(prompt: str) -> str:
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_llm_sync, prompt)
            return future.result(timeout=LLM_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        logger.exception("LLM call timed out after %s seconds", LLM_TIMEOUT_SECONDS)
        return json.dumps(FALLBACK_RESPONSE)
    except Exception:
        logger.exception("LLM call failed")
        return json.dumps(FALLBACK_RESPONSE)


def safe_parse_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object")
        return data
    except Exception:
        logger.exception("Failed to parse LLM JSON: %s", raw[:500])
        return dict(PARSE_FALLBACK_RESPONSE)


def validate_and_strip_hallucinations(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate LLM recommendations against catalog. Rescues correct names with wrong URLs."""
    url_map = _get_catalog_by_url()
    recs = parsed.get("recommendations") or []
    if not isinstance(recs, list):
        parsed["recommendations"] = []
        return parsed

    cleaned: list[dict[str, str]] = []
    rescued = 0
    removed = 0
    seen_urls: set[str] = set()

    for rec in recs:
        if not isinstance(rec, dict):
            removed += 1
            continue

        url = str(rec.get("url", "")).strip()
        name = str(rec.get("name", "")).strip()

        # 1. Try exact URL match first
        catalog_item = url_map.get(url)

        # 2. URL failed — try name-based rescue (LLM hallucinated the URL)
        if catalog_item is None and name:
            catalog_item = retriever.get_by_name(name)
            if catalog_item:
                rescued += 1
                logger.info("Rescued hallucinated URL for '%s' via name lookup", name)

        if catalog_item and catalog_item["url"] not in seen_urls:
            canonical = item_to_recommendation(catalog_item)
            cleaned.append(canonical)
            seen_urls.add(catalog_item["url"])
        else:
            removed += 1

    if rescued:
        logger.info("Rescued %d recommendation(s) with correct name but wrong URL", rescued)
    if removed:
        logger.warning("Removed %d truly invalid recommendation(s)", removed)

    parsed["recommendations"] = cleaned
    return parsed


def enforce_recommendation_limits(parsed: dict[str, Any]) -> dict[str, Any]:
    recs = parsed.get("recommendations") or []
    if not isinstance(recs, list):
        recs = []
    if len(recs) > 5:
        recs = recs[:5]
    parsed["recommendations"] = recs
    parsed["reply"] = str(parsed.get("reply", ""))
    parsed["end_of_conversation"] = bool(parsed.get("end_of_conversation", False))
    return parsed


def _normalize_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    if "reply" not in parsed:
        parsed["reply"] = PARSE_FALLBACK_RESPONSE["reply"]
    if "recommendations" not in parsed or parsed["recommendations"] is None:
        parsed["recommendations"] = []
    if "end_of_conversation" not in parsed:
        parsed["end_of_conversation"] = False
    return parsed


def _finalize(parsed: dict[str, Any]) -> dict[str, Any]:
    parsed = _normalize_parsed(parsed)
    parsed = validate_and_strip_hallucinations(parsed)
    parsed = enforce_recommendation_limits(parsed)
    return parsed


def ensure_recommendations_when_ready(
    parsed: dict[str, Any],
    messages: list[dict],
    results: list[dict[str, Any]],
    user_turn: int,
    is_compare: bool,
    is_switch: bool,
) -> dict[str, Any]:
    """Backfill shortlist from retrieval when context is sufficient but LLM returned none."""
    if is_compare or (is_first_user_turn(messages) and query_is_vague(messages)) or is_switch:
        return parsed
    if not has_recommendation_context(messages) or parsed.get("recommendations"):
        return parsed
    if not results:
        return parsed

    should_recommend = user_turn >= 2 or detect_refine_intent(messages) or user_turn >= 6
    if user_turn == 1 and has_recommendation_context(messages):
        should_recommend = True

    if should_recommend:
        parsed["recommendations"] = retriever.recommendations_from_results(results, limit=5)
        if parsed["recommendations"] and not parsed.get("reply", "").strip():
            parsed["reply"] = (
                "Based on your requirements, here are SHL assessments from our catalog "
                "that may be a good fit."
            )
        if user_turn >= MAX_USER_TURNS and parsed["recommendations"]:
            parsed["end_of_conversation"] = True
    return parsed


def _is_llm_fallback(parsed: dict[str, Any]) -> bool:
    return parsed.get("reply") == FALLBACK_RESPONSE["reply"] and not parsed.get(
        "recommendations"
    )


def build_retrieval_fallback(
    messages: list[dict], results: list[dict], user_turn: int, is_compare: bool
) -> dict:
    """Last resort: provide a helpful response even if the LLM is down."""
    if is_compare:
        return {
            "reply": (
                "I'm experiencing a temporary technical issue, but I've located those "
                "assessments in our catalog. However, I need my full analytical "
                "capabilities to perform a detailed comparison. Please try again in a moment."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    if results:
        recs = retriever.recommendations_from_results(results, limit=5)
        return {
            "reply": (
                "I'm having a slight technical glitch, but I've found these relevant "
                "assessments from our catalog based on your request. Do these look like "
                "what you're looking for?"
            ),
            "recommendations": recs,
            "end_of_conversation": False,
        }

    return {
        "reply": (
            "I'm having some trouble right now due to high traffic. Could you please "
            "rephrase your request or try again in a few seconds?"
        ),
        "recommendations": [],
        "end_of_conversation": False,
    }


def _fetch_results(messages: list[dict], is_compare: bool, is_switch: bool) -> list[dict[str, Any]]:
    if is_compare:
        return []
        
    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    
    if is_switch:
        # User switched roles, only search using the latest message to drop old context
        recent = user_contents[-1] if user_contents else ""
        full = recent
    else:
        recent = build_search_query(messages)
        full = build_full_user_query(messages)
        
    queries = [q for q in (recent, full) if q]
    if len(queries) > 1 and queries[0] == queries[1]:
        queries = [queries[0]]
        
    if len(queries) == 1:
        results = retriever.retrieve(queries[0], top_k=15)
    else:
        results = retriever.retrieve_merged(queries, top_k=15)

    for r in results:
        if not r.get("description"):
            full_item = retriever.get_by_name(r.get("name", ""))
            if full_item:
                r["description"] = full_item.get("description", "")
    return results


def run(messages: list[dict]) -> dict:
    user_turn = count_user_turns(messages)
    current_turn = len(messages)
    is_switch = detect_role_switch(messages)

    if user_turn > MAX_USER_TURNS:
        results = _fetch_results(messages, False, is_switch)
        return _finalize(build_turn_cap_response(results))

    if detect_off_topic(messages):
        return _finalize(build_refuse_response())

    is_compare = detect_compare_intent(messages)
    results = _fetch_results(messages, is_compare, is_switch)

    if is_first_user_turn(messages) and query_is_vague(messages) and not is_compare:
        return _finalize(build_clarify_response())

    if is_compare:
        catalog_context = build_compare_context(messages)
    else:
        catalog_context = format_catalog_for_prompt(results)

    mode_hint = ""
    is_refine = detect_refine_intent(messages)

    if is_switch:
        mode_hint = (
            "\n\nIMPORTANT: User has switched to a NEW job role. Do NOT carry over "
            "seniority (junior/senior), experience levels, or previous technical "
            "constraints (like Java) to this new role. Treat it as a completely "
            "NEW selection process. You MUST CLARIFY details for this new role (like "
            "seniority or specific skill area) before recommending, unless the user "
            "provided them in the latest message. Explain that you are asking because "
            "the role has changed.\n"
        )
    elif is_refine:
        mode_hint = (
            "\n\nIMPORTANT: User is refining a prior shortlist. Use REFINE mode. "
            "Return an updated 5 item shortlist from the CATALOG. Do not ask "
            "clarifying questions.\n"
        )
    elif (user_turn >= MAX_USER_TURNS - 1 or current_turn >= MAX_TOTAL_TURNS - 2) and has_recommendation_context(messages):
        mode_hint = (
            "\n\nIMPORTANT: We are nearing the turn limit ("
            f"{current_turn}/{MAX_TOTAL_TURNS}). You MUST RECOMMEND now (5 items) "
            "using the CATALOG. Do not clarify again.\n"
        )

    conversation_history = format_history(messages)
    prompt = (
        SYSTEM_PROMPT.format(
            catalog_context=catalog_context,
            conversation_history=conversation_history,
            current_turn=current_turn,
            user_turn=user_turn,
        )
        + mode_hint
    )

    raw_response = call_llm_with_timeout(prompt)
    parsed = safe_parse_json(raw_response)
    
    # Map new schema fields to old schema for internal processing
    if "reasoning_summary" in parsed and "reply" not in parsed:
        parsed["reply"] = parsed["reasoning_summary"]
        if parsed.get("clarification_question"):
            parsed["reply"] = f"{parsed['reply']}\n\n{parsed['clarification_question']}"
            
    parsed = _normalize_parsed(parsed)

    if _is_llm_fallback(parsed):
        logger.warning("LLM unavailable; using retrieval-based fallback")
        parsed = build_retrieval_fallback(messages, results, user_turn, is_compare)
    else:
        if is_first_user_turn(messages) and query_is_vague(messages) and not is_compare:
            parsed["recommendations"] = []
        elif is_switch and not has_recommendation_context(messages):
            parsed["recommendations"] = []
            parsed["end_of_conversation"] = False
        elif (user_turn >= MAX_USER_TURNS or current_turn >= MAX_TOTAL_TURNS - 1) and parsed.get("recommendations"):
            parsed["end_of_conversation"] = True
        parsed = ensure_recommendations_when_ready(
            parsed, messages, results, user_turn, is_compare, is_switch
        )

    parsed = _finalize(parsed)

    if is_first_user_turn(messages) and query_is_vague(messages):
        parsed["recommendations"] = []

    if user_turn >= MAX_USER_TURNS and not parsed.get("recommendations") and results:
        parsed = _finalize(build_turn_cap_response(results))

    return parsed
