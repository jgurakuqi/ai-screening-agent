"""Unit tests for validators and config helpers."""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.validator import (
    validate_name,
    validate_license,
    validate_city,
    validate_availability,
    validate_schedule,
    validate_experience_years,
    validate_experience_platforms,
    validate_start_date,
)
from backend.config import ConversationStage, get_next_stage
from backend.agent import parse_llm_response


# --- Name ---

def test_validate_name_valid():
    assert validate_name("María García") == (True, "María García", False)


def test_validate_name_short():
    assert validate_name("M") == (False, None, False)


def test_validate_name_null():
    assert validate_name(None) == (False, None, False)


def test_validate_name_whitespace():
    assert validate_name("  Juan Pérez  ") == (True, "Juan Pérez", False)


def test_validate_name_single_word():
    assert validate_name("Pablo") == (False, None, False)


# --- License ---

def test_validate_license_true():
    assert validate_license(True) == (True, True, False)


def test_validate_license_false():
    assert validate_license(False) == (True, False, True)  # disqualify!


def test_validate_license_null():
    assert validate_license(None) == (False, None, False)


# --- City ---

def test_validate_city_match():
    assert validate_city("Madrid") == (True, "Madrid", False)


def test_validate_city_case_insensitive():
    assert validate_city("madrid") == (True, "Madrid", False)


def test_validate_city_synonym():
    assert validate_city("CDMX") == (True, "Mexico City", False)


def test_validate_city_synonym_lowercase():
    assert validate_city("cdmx") == (True, "Mexico City", False)


def test_validate_city_fuzzy():
    assert validate_city("Madriz") == (True, "Madrid", False)


def test_validate_city_no_match():
    is_valid, value, disqualify = validate_city("Tokyo")
    assert is_valid is True  # LLM gave a clear city
    assert value is None
    assert disqualify is True  # not in service area


def test_validate_city_null():
    assert validate_city(None) == (False, None, False)


def test_validate_city_empty():
    assert validate_city("") == (False, None, False)


def test_validate_city_vague_somewhere():
    """Vague answers should retry, not disqualify."""
    assert validate_city("somewhere") == (False, None, False)


def test_validate_city_vague_idk():
    assert validate_city("I don't know") == (False, None, False)


def test_validate_city_vague_spanish():
    assert validate_city("no sé") == (False, None, False)
    assert validate_city("por ahí") == (False, None, False)


def test_validate_city_vague_nearby():
    assert validate_city("close by") == (False, None, False)
    assert validate_city("around here") == (False, None, False)


def test_validate_city_gibberish_long():
    """Excessively long input should retry, not disqualify."""
    assert validate_city("a" * 50) == (False, None, False)


def test_validate_city_real_unknown_still_disqualifies():
    """A real city name outside the service area should still disqualify."""
    is_valid, value, disqualify = validate_city("Tokyo")
    assert is_valid is True
    assert disqualify is True


# --- Availability ---

def test_validate_availability_full_time():
    assert validate_availability("full-time") == (True, "full-time", False)


def test_validate_availability_part_time():
    assert validate_availability("part-time") == (True, "part-time", False)


def test_validate_availability_weekends():
    assert validate_availability("weekends") == (True, "weekends", False)


def test_validate_availability_null():
    assert validate_availability(None) == (False, None, False)


def test_validate_availability_invalid():
    assert validate_availability("sometimes") == (False, None, False)


# --- Schedule ---

def test_validate_schedule_morning():
    assert validate_schedule("morning") == (True, "morning", False)


def test_validate_schedule_flexible():
    assert validate_schedule("flexible") == (True, "flexible", False)


def test_validate_schedule_null():
    assert validate_schedule(None) == (False, None, False)


# --- Experience Years ---

def test_validate_experience_zero():
    assert validate_experience_years(0) == (True, 0.0, False)


def test_validate_experience_two():
    assert validate_experience_years(2) == (True, 2.0, False)


def test_validate_experience_fractional():
    assert validate_experience_years(0.42) == (True, 0.42, False)


def test_validate_experience_half_year():
    assert validate_experience_years(0.5) == (True, 0.5, False)


def test_validate_experience_null():
    assert validate_experience_years(None) == (False, None, False)


def test_validate_experience_negative():
    assert validate_experience_years(-1) == (False, None, False)


# --- Experience Platforms ---

def test_validate_platforms_list():
    assert validate_experience_platforms(["Glovo", "Uber Eats"]) == (True, ["Glovo", "Uber Eats"], False)


def test_validate_platforms_none_string():
    assert validate_experience_platforms("none") == (True, [], False)


def test_validate_platforms_null():
    assert validate_experience_platforms(None) == (False, None, False)


def test_validate_platforms_empty_list():
    assert validate_experience_platforms([]) == (False, None, False)


def test_validate_platforms_vague_answer():
    """Vague answers like 'many' or 'several' should trigger a retry."""
    assert validate_experience_platforms("many") == (False, None, False)
    assert validate_experience_platforms("several") == (False, None, False)
    assert validate_experience_platforms("many ones") == (False, None, False)
    assert validate_experience_platforms("todas") == (False, None, False)


def test_validate_platforms_vague_list():
    """A list of only vague entries should trigger a retry."""
    assert validate_experience_platforms(["many", "several"]) == (False, None, False)


def test_validate_platforms_mixed_list():
    """A list with some real platforms and some vague entries keeps the real ones."""
    assert validate_experience_platforms(["Glovo", "many"]) == (True, ["Glovo"], False)


def test_validate_platforms_unknown_platform():
    """Unknown but specific platform names should be accepted."""
    assert validate_experience_platforms("Local Courier Co") == (True, ["Local Courier Co"], False)


# --- Name (gibberish) ---

def test_validate_name_gibberish_repetitive():
    """Repetitive nonsense strings should be rejected."""
    gibberish = "towards " * 20
    assert validate_name(gibberish.strip()) == (False, None, False)


def test_validate_name_too_long():
    assert validate_name("a" * 100) == (False, None, False)


def test_validate_name_normal_long():
    """Legitimate multi-part names should still pass."""
    assert validate_name("María del Carmen García López") == (True, "María del Carmen García López", False)


# --- Start Date ---

def test_validate_date_iso():
    assert validate_start_date("2026-04-01") == (True, "2026-04-01", False)


def test_validate_date_asap():
    assert validate_start_date("ASAP") == (True, "ASAP", False)


def test_validate_date_asap_lowercase():
    assert validate_start_date("asap") == (True, "ASAP", False)


def test_validate_date_inmediatamente():
    assert validate_start_date("inmediatamente") == (True, "ASAP", False)


def test_validate_date_null():
    assert validate_start_date(None) == (False, None, False)


# --- get_next_stage ---

def test_get_next_stage_normal():
    assert get_next_stage(ConversationStage.LICENSE, {}) == ConversationStage.CITY


def test_get_next_stage_skip_platform():
    assert get_next_stage(
        ConversationStage.EXPERIENCE_YEARS, {"experience_years": 0}
    ) == ConversationStage.START_DATE


def test_get_next_stage_no_skip():
    assert get_next_stage(
        ConversationStage.EXPERIENCE_YEARS, {"experience_years": 2}
    ) == ConversationStage.EXPERIENCE_PLATFORM


def test_get_next_stage_no_skip_fractional():
    assert get_next_stage(
        ConversationStage.EXPERIENCE_YEARS, {"experience_years": 0.42}
    ) == ConversationStage.EXPERIENCE_PLATFORM


def test_get_next_stage_closing():
    assert get_next_stage(ConversationStage.CLOSING, {}) is None


def test_get_next_stage_name_to_license():
    assert get_next_stage(ConversationStage.NAME, {}) == ConversationStage.LICENSE


def test_get_next_stage_start_date_to_closing():
    assert get_next_stage(ConversationStage.START_DATE, {}) == ConversationStage.CLOSING


# --- LLM Response Parsing ---

def test_parse_response_dict():
    """Function calling mode returns a dict — parsed directly."""
    result = parse_llm_response({
        "field_value": True,
        "response": "¡Perfecto!",
        "detected_language": "es",
        "exit_intent": False,
        "is_offensive": False,
        "sentiment": "neutral",
    })
    assert result.field_value is True
    assert result.response == "¡Perfecto!"
    assert result.sentiment == "neutral"


def test_parse_response_json_string():
    """Content mode fallback — valid JSON string."""
    result = parse_llm_response('{"field_value": null, "response": "¿Tienes carnet?"}')
    assert result.field_value is None
    assert result.response == "¿Tienes carnet?"


def test_parse_response_malformed_string():
    """Non-JSON string falls back gracefully."""
    result = parse_llm_response("Sure, do you have a license?")
    assert result.field_value is None


def test_parse_response_list_value():
    result = parse_llm_response({
        "field_value": ["Glovo", "Uber Eats"],
        "response": "Great!",
        "detected_language": "es",
        "exit_intent": False,
        "is_offensive": False,
        "sentiment": "positive",
    })
    assert result.field_value == ["Glovo", "Uber Eats"]


def test_parse_response_integer_value():
    result = parse_llm_response({
        "field_value": 3,
        "response": "Nice experience!",
        "detected_language": "es",
        "exit_intent": False,
        "is_offensive": False,
        "sentiment": "neutral",
    })
    assert result.field_value == 3
