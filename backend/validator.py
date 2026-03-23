"""Field validators for each screening stage.

Each validator receives the LLM-extracted value and returns a
``(is_valid, normalized_value, should_disqualify)`` tuple. The ``VALIDATORS``
dict at the bottom maps stage names to their corresponding validator function.
"""

import json
import os
import re
from pathlib import Path

import dateparser
from rapidfuzz import fuzz

from backend.logging_config import logger

# Load service areas at module level
_SERVICE_AREAS_PATH = Path(__file__).parent.parent / "data" / "service_areas.json"
with open(_SERVICE_AREAS_PATH, "r", encoding="utf-8") as f:
    _SERVICE_AREAS = json.load(f)

CITIES = _SERVICE_AREAS["cities"]
SYNONYMS = _SERVICE_AREAS["synonyms"]
logger.debug("Loaded {} service area cities and {} synonyms", len(CITIES), len(SYNONYMS))

# Dynamic length cap: longest known city/synonym * 1.3, used to detect gibberish
_MAX_CITY_LENGTH = int(
    max(len(c) for c in (*CITIES, *SYNONYMS.keys())) * 1.3
)

# Vague phrases that are clearly not city names — retry instead of disqualifying
_VAGUE_CITY_PATTERNS = re.compile(
    r"^(somewhere|anywhere|here|around here|close by|nearby|near|"
    r"i don'?t know|not sure|idk|dunno|no s[eé]|"
    r"no lo s[eé]|por ah[ií]|cerca|aqui|aquí|"
    r"wherever|wherever you need|where you want|donde sea)$",
    re.IGNORECASE,
)

VALID_AVAILABILITY = {"full-time", "part-time", "weekends"}
VALID_SCHEDULE = {"morning", "afternoon", "evening", "flexible"}

FUZZY_THRESHOLD = 80


def _is_gibberish_name(name: str) -> bool:
    """Detect gibberish or repetitive input that is not a plausible name."""
    # Excessively long names are suspicious
    if len(name) > 80:
        return True
    # Very low ratio of unique words to total words suggests repetition
    words = name.lower().split()
    if len(words) >= 4:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            return True
    # Name should contain mostly alphabetic characters (allow spaces, hyphens, apostrophes)
    alpha_chars = sum(1 for c in name if c.isalpha())
    if len(name) > 0 and alpha_chars / len(name) < 0.5:
        return True
    return False


def validate_name(llm_value: object) -> tuple[bool, str | None, bool]:
    """Validate a candidate name extracted by the LLM.

    Args:
        llm_value: Name string from the LLM, or ``None`` if unclear.

    Returns:
        Tuple of ``(is_valid, normalized_value, should_disqualify)``.
        Names shorter than 2 characters or gibberish are treated as invalid.
        Non-string types (booleans, lists, etc.) are rejected as invalid.
    """
    if llm_value is None:
        return (False, None, False)
    if not isinstance(llm_value, str):
        logger.debug("Name validation: rejected non-string type {}", type(llm_value).__name__)
        return (False, None, False)
    name = llm_value.strip()
    if len(name) < 2:
        return (False, None, False)
    if _is_gibberish_name(name):
        logger.debug("Name validation: rejected gibberish input '{}'", name[:40])
        return (False, None, False)
    if len(name.split()) < 2:
        logger.debug("Name validation: need full name (first + last), got '{}'", name)
        return (False, None, False)
    return (True, name, False)


def validate_license(llm_value: object) -> tuple[bool, bool | None, bool]:
    """Validate the driver's licence answer.

    Args:
        llm_value: ``True``, ``False``, or ``None`` from the LLM JSON.

    Returns:
        Tuple of ``(is_valid, value, should_disqualify)``. A ``False``
        answer triggers disqualification.
    """
    if llm_value is True:
        return (True, True, False)
    if llm_value is False:
        return (True, False, True)  # DISQUALIFY
    return (False, None, False)  # unclear, retry


def _find_city_in_tokens(text: str) -> str | None:
    """Check if any service-area city appears as a token/substring in *text*.

    Handles compound phrases like "Pozuelo near Madrid" or "a neighbourhood
    of Madrid" by splitting on common delimiters and checking each token
    against the city list (exact) and synonym table.

    Returns the canonical city name if found, otherwise ``None``.
    """
    import re

    tokens = re.split(r"[\s,;.\-/()]+", text)
    lower_text = text.lower()

    # Check if a full city name (including multi-word like "Mexico City")
    # appears as a substring
    for city in CITIES:
        if city.lower() in lower_text:
            return city
    for synonym, canonical in SYNONYMS.items():
        if synonym.lower() in lower_text:
            return canonical

    # Check individual tokens against single-word cities
    for token in tokens:
        token_lower = token.lower()
        for city in CITIES:
            if token_lower == city.lower():
                return city
        for synonym, canonical in SYNONYMS.items():
            if token_lower == synonym.lower():
                return canonical

    return None


def validate_city(llm_value: object) -> tuple[bool, str | None, bool]:
    """Validate the candidate's city against the service-area list.

    Resolution order: synonym lookup → exact match → token/substring
    match → fuzzy match (rapidfuzz score ≥ 80). Cities not in the
    service area trigger disqualification.

    Args:
        llm_value: City name string from the LLM, or ``None``.

    Returns:
        Tuple of ``(is_valid, canonical_city_or_None, should_disqualify)``.
        Non-string types (booleans, lists, etc.) are rejected as invalid
        rather than risking a false disqualification.
    """
    if llm_value is None:
        logger.trace("City validation: value is None, requesting retry")
        return (False, None, False)  # retry
    if not isinstance(llm_value, str):
        logger.debug("City validation: rejected non-string type {}", type(llm_value).__name__)
        return (False, None, False)  # retry, don't disqualify on bad types

    city_str = llm_value.strip()
    if not city_str:
        return (False, None, False)

    # Check synonyms first (case-insensitive)
    for synonym, canonical in SYNONYMS.items():
        if city_str.lower() == synonym.lower():
            logger.debug("City '{}' matched synonym -> '{}'", city_str, canonical)
            return (True, canonical, False)

    # Exact match (case-insensitive)
    for city in CITIES:
        if city_str.lower() == city.lower():
            logger.debug("City '{}' exact match -> '{}'", city_str, city)
            return (True, city, False)

    # Token/substring match — handles compound phrases like
    # "Pozuelo near Madrid" or "a neighbourhood of Madrid"
    token_match = _find_city_in_tokens(city_str)
    if token_match:
        logger.debug("City '{}' token/substring matched -> '{}'", city_str, token_match)
        return (True, token_match, False)

    # Fuzzy match
    best_score = 0
    best_match = None
    for city in CITIES:
        score = fuzz.ratio(city_str.lower(), city.lower())
        if score > best_score:
            best_score = score
            best_match = city

    if best_score >= FUZZY_THRESHOLD:
        logger.debug("City '{}' fuzzy matched -> '{}' (score={})", city_str, best_match, best_score)
        return (True, best_match, False)

    # Before disqualifying, check if the input looks like a genuine city name
    # rather than a vague/evasive answer that could cause a false disqualification.
    if _VAGUE_CITY_PATTERNS.match(city_str):
        logger.debug("City '{}' is a vague phrase, requesting retry", city_str)
        return (False, None, False)
    if len(city_str) > _MAX_CITY_LENGTH:
        logger.debug("City '{}' exceeds max plausible length ({}), requesting retry", city_str[:30], _MAX_CITY_LENGTH)
        return (False, None, False)

    # LLM gave a clear city name but it's not in our service area -> DISQUALIFY
    logger.info("City '{}' not in service area (best fuzzy: '{}' score={}) — DISQUALIFY", city_str, best_match, best_score)
    return (True, None, True)


def validate_availability(llm_value: object) -> tuple[bool, str | None, bool]:
    """Validate the candidate's availability preference.

    Args:
        llm_value: One of ``"full-time"``, ``"part-time"``, ``"weekends"``,
            or ``None``.

    Returns:
        Tuple of ``(is_valid, normalized_value, should_disqualify)``.
        Never disqualifies.
    """
    if llm_value is None or not isinstance(llm_value, str):
        return (False, None, False)
    val = llm_value.strip().lower()
    if val in VALID_AVAILABILITY:
        return (True, val, False)
    return (False, None, False)


def validate_schedule(llm_value: object) -> tuple[bool, str | None, bool]:
    """Validate the candidate's preferred schedule.

    Args:
        llm_value: One of ``"morning"``, ``"afternoon"``, ``"evening"``,
            ``"flexible"``, or ``None``.

    Returns:
        Tuple of ``(is_valid, normalized_value, should_disqualify)``.
        Never disqualifies.
    """
    if llm_value is None or not isinstance(llm_value, str):
        return (False, None, False)
    val = llm_value.strip().lower()
    if val in VALID_SCHEDULE:
        return (True, val, False)
    return (False, None, False)


def validate_experience_years(llm_value: object) -> tuple[bool, float | None, bool]:
    """Validate the candidate's years of delivery experience.

    Args:
        llm_value: A non-negative number (int or float), or ``None`` if unclear.

    Returns:
        Tuple of ``(is_valid, years_float_or_None, should_disqualify)``.
        Never disqualifies.
    """
    if llm_value is None:
        return (False, None, False)
    try:
        val = float(llm_value)
        if val >= 0:
            return (True, round(val, 2), False)
        return (False, None, False)
    except (ValueError, TypeError):
        return (False, None, False)


# Vague answers that don't name specific platforms — triggers a retry
_VAGUE_PLATFORM_PATTERNS = re.compile(
    r"^(many|several|some|a lot|a few|various|todos|varias|muchas|muchos|"
    r"bastantes|diferentes|multiple|lots|enough|plenty|"
    r"many ones|all of them|todas|all|most)$",
    re.IGNORECASE,
)


def _is_vague_platform(value: str) -> bool:
    """Return True if the string is a vague non-answer rather than a platform name."""
    return bool(_VAGUE_PLATFORM_PATTERNS.match(value.strip()))


def validate_experience_platforms(llm_value: object) -> tuple[bool, list[str] | None, bool]:
    """Validate the list of delivery platforms the candidate has worked with.

    Args:
        llm_value: A list of platform name strings, a single string,
            ``"none"``/``"ninguno"``, or ``None``.

    Returns:
        Tuple of ``(is_valid, platform_list, should_disqualify)``.
        Returns invalid for vague answers (e.g. "many", "several") to
        trigger a retry asking for specific platform names.
        Never disqualifies.
    """
    if llm_value is None:
        return (False, None, False)
    if isinstance(llm_value, str):
        if llm_value.lower() in ("none", "no", "ninguno", "ninguna"):
            return (True, [], False)
        if _is_vague_platform(llm_value):
            logger.debug("Platform validation: vague answer '{}', requesting retry", llm_value)
            return (False, None, False)
        return (True, [llm_value], False)
    if isinstance(llm_value, list):
        platforms = [str(p).strip() for p in llm_value if p]
        # If all entries are vague, retry
        if platforms and all(_is_vague_platform(p) for p in platforms):
            logger.debug("Platform validation: all entries vague {}, requesting retry", platforms)
            return (False, None, False)
        # Filter out vague entries, keep specific ones
        specific = [p for p in platforms if not _is_vague_platform(p)]
        if not specific:
            return (False, None, False)
        return (True, specific, False)
    return (False, None, False)


def validate_start_date(llm_value: object) -> tuple[bool, str | None, bool]:
    """Validate the candidate's desired start date.

    Accepts ISO 8601 dates, ASAP variants (EN/ES), and relative
    expressions parsed by ``dateparser``.

    Args:
        llm_value: An ISO date string, ``"ASAP"`` variant, a natural-language
            date expression, or ``None``.

    Returns:
        Tuple of ``(is_valid, normalized_date_or_ASAP_or_None, should_disqualify)``.
        Never disqualifies.
    """
    if llm_value is None or not isinstance(llm_value, str):
        return (False, None, False)

    val = llm_value.strip()

    # Accept ASAP variants
    if val.upper() in ("ASAP", "INMEDIATAMENTE", "IMMEDIATELY", "YA", "NOW"):
        logger.debug("Start date '{}' -> ASAP", val)
        return (True, "ASAP", False)

    # Try ISO format first
    try:
        from datetime import datetime
        datetime.fromisoformat(val)
        logger.debug("Start date '{}' parsed as ISO date", val)
        return (True, val, False)
    except ValueError:
        pass

    # Try dateparser for relative dates
    parsed = dateparser.parse(val, settings={"PREFER_DATES_FROM": "future"})
    if parsed:
        result = parsed.strftime("%Y-%m-%d")
        logger.debug("Start date '{}' parsed via dateparser -> {}", val, result)
        return (True, result, False)

    logger.debug("Start date '{}' could not be parsed", val)
    return (False, None, False)


# Map stage names to their validator functions
VALIDATORS = {
    "name": validate_name,
    "license": validate_license,
    "city": validate_city,
    "availability": validate_availability,
    "schedule": validate_schedule,
    "experience_years": validate_experience_years,
    "experience_platform": validate_experience_platforms,
    "start_date": validate_start_date,
}
