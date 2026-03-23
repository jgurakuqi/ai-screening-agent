"""Scenario tests for the agent state machine with mocked LLM."""
import sys
import os
import json
import asyncio
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import AsyncMock, patch

import backend.agent as agent_module
from backend import storage
from backend.config import (
    ConversationStage, ConversationStatus,
    generate_conversation_id,
)
from backend.agent import (
    call_llm, process_message, generate_greeting, _detect_response_language,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("backend.storage.DB_PATH", db_path)
    monkeypatch.setattr("backend.config.DB_PATH", db_path)
    storage.init_db()
    yield


def _create_conversation() -> str:
    """Helper to create a conversation and store greeting."""
    conv_id = generate_conversation_id()
    storage.create_conversation(conv_id)
    greeting = generate_greeting("es")
    storage.save_message(conv_id, "assistant", greeting, "es")
    storage.update_conversation(conv_id, stage=ConversationStage.NAME.value)
    return conv_id


def _mock_llm_response(field_value, response_text, sentiment="neutral"):
    """Create a dict mimicking function calling tool call output."""
    return {
        "field_value": field_value,
        "response": response_text,
        "detected_language": "es",
        "exit_intent": False,
        "is_offensive": False,
        "sentiment": sentiment,
    }


# Patch both call_llm and finalize_conversation for tests that trigger finalization.
def _patch_agent(llm_responses):
    """Return a context manager that patches call_llm and finalize_conversation."""
    mock_llm = AsyncMock(side_effect=llm_responses)
    mock_finalize = AsyncMock()
    return (
        patch("backend.agent.call_llm", mock_llm),
        patch("backend.agent.finalize_conversation", mock_finalize),
        mock_llm,
        mock_finalize,
    )


# --- Response Language Detection (Lingua) ---

def test_detect_response_language_spanish():
    assert _detect_response_language("¡Genial, gracias!") == "es"


def test_detect_response_language_spanish_no_special_chars():
    assert _detect_response_language("Perfecto, Madrid encaja muy bien!") == "es"


def test_detect_response_language_english():
    assert _detect_response_language("Great, thanks! What city do you live in?") == "en"


def test_detect_response_language_long_spanish():
    assert _detect_response_language("Hola, me llamo Juan y vivo en Madrid") == "es"


def test_detect_response_language_long_english():
    assert _detect_response_language("Hello, my name is John and I live in London") == "en"


# --- Greeting ---

def test_greeting_spanish():
    greeting = generate_greeting("es")
    assert "FreshRoute" in greeting
    assert "nombre" in greeting or "llamas" in greeting


def test_greeting_english():
    greeting = generate_greeting("en")
    assert "FreshRoute" in greeting
    assert "name" in greeting


# --- Happy Path ---

@pytest.mark.asyncio
async def test_happy_path():
    """Full screening flow with all valid answers -> qualified."""
    conv_id = _create_conversation()

    responses = [
        _mock_llm_response("María García", "¡Hola María! ¿Tienes carnet de conducir?"),
        _mock_llm_response(True, "¡Perfecto! ¿En qué ciudad vives?"),
        _mock_llm_response("Madrid", "¡Genial, Madrid está en nuestra zona! ¿Qué disponibilidad tienes?"),
        _mock_llm_response("full-time", "Excelente. ¿Qué horario prefieres?"),
        _mock_llm_response("morning", "¡Bien! ¿Cuántos años de experiencia en reparto tienes?"),
        _mock_llm_response(2, "¿En qué plataformas has trabajado?"),
        _mock_llm_response(["Glovo"], "¡Genial! ¿Cuándo podrías empezar?"),
        _mock_llm_response("ASAP", "¡Perfecto!"),
    ]

    p1, p2, mock_llm, mock_finalize = _patch_agent(responses)
    with p1, p2:
        messages = [
            "Me llamo María García",
            "Sí, tengo carnet",
            "Vivo en Madrid",
            "Tiempo completo",
            "Por la mañana",
            "2 años",
            "Glovo",
            "Lo antes posible",
        ]

        for msg in messages:
            result = await process_message(conv_id, msg)

        assert result["status"] in (ConversationStatus.QUALIFIED, "qualified")
        mock_finalize.assert_called_once()

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"]["full_name"] == "María García"
    assert conv["extracted_data"]["driver_license"] is True
    assert conv["extracted_data"]["city_zone"] == "Madrid"
    assert conv["extracted_data"]["availability"] == "full-time"
    assert conv["extracted_data"]["preferred_schedule"] == "morning"
    assert conv["extracted_data"]["experience_years"] == 2
    assert conv["extracted_data"]["experience_platforms"] == ["Glovo"]
    assert conv["extracted_data"]["start_date"] == "ASAP"


# --- Sentiment Tracking ---

@pytest.mark.asyncio
async def test_sentiment_recorded():
    """Sentiment values are appended to sentiment_history in extracted_data."""
    conv_id = _create_conversation()

    responses = [
        _mock_llm_response("Test User", "¡Hola! ¿Tienes carnet?", sentiment="positive"),
        _mock_llm_response(True, "¿En qué ciudad vives?", sentiment="neutral"),
    ]

    with patch("backend.agent.call_llm", new_callable=AsyncMock, side_effect=responses):
        await process_message(conv_id, "Me llamo Test User")
        await process_message(conv_id, "Sí")

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("sentiment_history") == ["positive", "neutral"]


# --- Early Disqualification (No License) ---

@pytest.mark.asyncio
async def test_disqualified_no_license():
    """Candidate has no license -> disqualified at stage 2."""
    conv_id = _create_conversation()

    responses = [
        _mock_llm_response("Juan Pérez", "¡Hola Juan! ¿Tienes carnet de conducir?"),
        _mock_llm_response(False, "Gracias por tu interés, Juan. El carnet es requisito imprescindible."),
    ]

    p1, p2, mock_llm, mock_finalize = _patch_agent(responses)
    with p1, p2:
        result = await process_message(conv_id, "Me llamo Juan Pérez")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")

        result = await process_message(conv_id, "No, no tengo")
        assert result["status"] in (ConversationStatus.DISQUALIFIED, "disqualified")
        mock_finalize.assert_called_once()

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("disqualification_reason") == "no_license"
    assert conv["extracted_data"]["driver_license"] is False


# --- Disqualification (Wrong City) ---

@pytest.mark.asyncio
async def test_disqualified_wrong_city():
    """Candidate lives outside service area -> disqualified."""
    conv_id = _create_conversation()

    responses = [
        _mock_llm_response("Ana López", "¡Hola Ana! ¿Tienes carnet de conducir?"),
        _mock_llm_response(True, "¡Perfecto! ¿En qué ciudad vives?"),
        _mock_llm_response("Tokyo", "Lo siento, Tokyo no está en nuestra zona de servicio."),
    ]

    p1, p2, mock_llm, mock_finalize = _patch_agent(responses)
    with p1, p2:
        await process_message(conv_id, "Soy Ana López")
        await process_message(conv_id, "Sí, tengo carnet")
        result = await process_message(conv_id, "Vivo en Tokyo")

        assert result["status"] in (ConversationStatus.DISQUALIFIED, "disqualified")
        mock_finalize.assert_called_once()

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("disqualification_reason") == "outside_area"


# --- Experience Platform Skip (0 years) ---

@pytest.mark.asyncio
async def test_experience_skip_zero_years():
    """If experience_years == 0, skip the platform question."""
    conv_id = _create_conversation()

    responses = [
        _mock_llm_response("Carlos Ruiz", "¡Hola Carlos! ¿Tienes carnet?"),
        _mock_llm_response(True, "¿En qué ciudad vives?"),
        _mock_llm_response("Barcelona", "¡Barcelona está en nuestra zona! ¿Disponibilidad?"),
        _mock_llm_response("weekends", "¿Horario preferido?"),
        _mock_llm_response("flexible", "¿Cuántos años de experiencia?"),
        _mock_llm_response(0, "¿Cuándo podrías empezar?"),
        _mock_llm_response("2026-05-01", "¡Perfecto!"),
    ]

    p1, p2, mock_llm, mock_finalize = _patch_agent(responses)
    with p1, p2:
        messages = [
            "Soy Carlos Ruiz", "Sí", "Barcelona",
            "Fines de semana", "Flexible", "Ninguna", "1 de mayo 2026",
        ]
        for msg in messages:
            result = await process_message(conv_id, msg)

        assert result["status"] in (ConversationStatus.QUALIFIED, "qualified")

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"]["experience_years"] == 0


# --- Unclear Field -> Retry ---

@pytest.mark.asyncio
async def test_unclear_field_retry():
    """Unclear name triggers retry, then succeeds."""
    conv_id = _create_conversation()

    responses = [
        _mock_llm_response(None, "No entendí tu nombre. ¿Podrías repetirlo?"),
        _mock_llm_response("Pedro Martín", "¡Hola Pedro! ¿Tienes carnet de conducir?"),
    ]

    with patch("backend.agent.call_llm", new_callable=AsyncMock, side_effect=responses):
        result = await process_message(conv_id, "asdfghjkl")
        assert result["stage"] == ConversationStage.NAME.value  # still on name

        result = await process_message(conv_id, "Me llamo Pedro Martín")
        assert result["stage"] == ConversationStage.LICENSE.value  # advanced


# --- JSON Parse Failure (content mode fallback) ---

@pytest.mark.asyncio
async def test_json_parse_failure():
    """LLM returns non-JSON string -> graceful fallback, triggers retry."""
    conv_id = _create_conversation()

    responses = [
        "This is not JSON at all!",  # malformed string (content mode fallback)
        _mock_llm_response("Laura Sánchez", "¡Hola Laura! ¿Tienes carnet?"),
    ]

    with patch("backend.agent.call_llm", new_callable=AsyncMock, side_effect=responses):
        result = await process_message(conv_id, "hola")
        assert result["stage"] == ConversationStage.NAME.value  # retry

        result = await process_message(conv_id, "Soy Laura Sánchez")
        assert result["stage"] == ConversationStage.LICENSE.value  # advanced


# --- Empty Content Fallback (OpenRouter provider-level tests) ---

@pytest.mark.asyncio
async def test_openrouter_retries_when_primary_returns_empty_content():
    """A 200 response with null content should fall through to the next model."""
    from backend.llm.openrouter_provider import OpenRouterProvider

    empty_response = SimpleNamespace(
        model="openai/gpt-5-nano-2025-08-07",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="length",
            )
        ],
    )
    good_content = json.dumps(_mock_llm_response("Juan Garcia", "Hola Juan. Tienes carnet?"))
    good_response = SimpleNamespace(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=good_content),
                finish_reason="stop",
            )
        ],
    )
    create = AsyncMock(side_effect=[empty_response, good_response])

    provider = OpenRouterProvider(
        api_key="test-key",
        models=["model-a", "model-b"],
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    raw = await provider.generate([{"role": "user", "content": "Hola"}])
    assert raw == good_content
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_openrouter_skips_recent_empty_content_model():
    """After an empty-content response, later calls should skip that model during cooldown."""
    from backend.llm.openrouter_provider import OpenRouterProvider

    empty_response = SimpleNamespace(
        model="openai/gpt-5-nano-2025-08-07",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="length",
            )
        ],
    )
    good_content = json.dumps(_mock_llm_response("Juan Garcia", "Hola Juan. Tienes carnet?"))
    good_response = SimpleNamespace(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=good_content),
                finish_reason="stop",
            )
        ],
    )
    create = AsyncMock(side_effect=[empty_response, good_response, good_response])

    provider = OpenRouterProvider(
        api_key="test-key",
        models=["model-a", "model-b"],
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )

    raw1 = await provider.generate([{"role": "user", "content": "Hola"}])
    raw2 = await provider.generate([{"role": "user", "content": "Hola otra vez"}])

    assert raw1 == good_content
    assert raw2 == good_content
    assert create.await_count == 3


# --- ASAP Start Date ---

@pytest.mark.asyncio
async def test_asap_start_date():
    """ASAP is accepted as a valid start_date."""
    conv_id = _create_conversation()
    # Fast-forward to start_date stage
    storage.update_conversation(conv_id, stage=ConversationStage.START_DATE.value)
    storage.update_extracted_field(conv_id, "full_name", "Test User")
    storage.update_extracted_field(conv_id, "driver_license", True)
    storage.update_extracted_field(conv_id, "city_zone", "Madrid")
    storage.update_extracted_field(conv_id, "availability", "full-time")
    storage.update_extracted_field(conv_id, "preferred_schedule", "morning")
    storage.update_extracted_field(conv_id, "experience_years", 1)
    storage.update_extracted_field(conv_id, "experience_platforms", ["Glovo"])

    responses = [
        _mock_llm_response("ASAP", "¡Perfecto!"),
    ]

    p1, p2, mock_llm, mock_finalize = _patch_agent(responses)
    with p1, p2:
        result = await process_message(conv_id, "Lo antes posible")
        assert result["status"] in (ConversationStatus.QUALIFIED, "qualified")

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"]["start_date"] == "ASAP"


# --- Language Switch ---

@pytest.mark.asyncio
async def test_language_switch():
    """Agent detects language switch from Spanish to English."""
    conv_id = _create_conversation()

    responses = [
        {
            "field_value": "John Smith",
            "response": "Hi John! Do you have a driver's license?",
            "detected_language": "en",
            "exit_intent": False,
            "is_offensive": False,
            "sentiment": "neutral",
        },
    ]

    with patch("backend.agent.call_llm", new_callable=AsyncMock, side_effect=responses):
        result = await process_message(conv_id, "My name is John Smith")

    conv = storage.get_conversation(conv_id)
    assert conv["language"] == "en"


# --- Terminal Conversation Rejects Messages ---

@pytest.mark.asyncio
async def test_terminal_conversation():
    """Messages to a completed conversation are rejected."""
    conv_id = _create_conversation()
    storage.update_conversation(conv_id, status=ConversationStatus.QUALIFIED.value)

    result = await process_message(conv_id, "Hello again")
    assert "already ended" in result["response"]


@pytest.mark.asyncio
async def test_terminal_withdrawn_conversation():
    """Messages to a withdrawn conversation are rejected."""
    conv_id = _create_conversation()
    storage.update_conversation(conv_id, status=ConversationStatus.WITHDRAWN.value)

    result = await process_message(conv_id, "Hello again")
    assert "already ended" in result["response"]
    assert result["status"] in (ConversationStatus.WITHDRAWN, "withdrawn")


# --- LLM-detected exit intent ---

@pytest.mark.asyncio
async def test_exit_intent_via_llm():
    """LLM detects exit intent -> confirmation requested."""
    conv_id = _create_conversation()

    responses = [
        {
            "field_value": None,
            "response": "¿Seguro que quieres terminar?",
            "detected_language": "es",
            "exit_intent": True,
            "is_offensive": False,
            "sentiment": "neutral",
        },
    ]

    with patch("backend.agent.call_llm", new_callable=AsyncMock, side_effect=responses):
        result = await process_message(conv_id, "No me interesa")

    assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")
    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("_pending_withdrawal") is True


# --- LLM-detected offensive content ---

@pytest.mark.asyncio
async def test_offensive_via_llm():
    """LLM detects offensive content -> warning issued."""
    conv_id = _create_conversation()

    responses = [
        {
            "field_value": None,
            "response": "Por favor, mantén un tono respetuoso.",
            "detected_language": "es",
            "exit_intent": False,
            "is_offensive": True,
            "sentiment": "frustrated",
        },
    ]

    with patch("backend.agent.call_llm", new_callable=AsyncMock, side_effect=responses):
        result = await process_message(conv_id, "something offensive")

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("_offensive_strikes") == 1
