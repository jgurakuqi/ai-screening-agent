"""Tests for guardrails: exit intent, offensive content, and pipeline integration."""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import AsyncMock, patch

import backend.agent as agent_module
from backend import storage
from backend.config import (
    ConversationStage, ConversationStatus,
    generate_conversation_id,
)
from backend.agent import process_message, generate_greeting
from backend.guardrails import (
    detect_exit_intent,
    detect_offensive_content,
    detect_confirmation,
    detect_denial,
    FRONTEND_STOP_SIGNAL,
)


# ---- Fixtures ----

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


def _mock_llm_response(field_value, response_text, exit_intent=False, is_offensive=False, sentiment="neutral"):
    """Create a dict mimicking function calling tool call output."""
    return {
        "field_value": field_value,
        "response": response_text,
        "detected_language": "es",
        "exit_intent": exit_intent,
        "is_offensive": is_offensive,
        "sentiment": sentiment,
    }


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


# ======================================================================
# Unit tests — detection functions
# ======================================================================

class TestDetectExitIntent:
    def test_english_stop(self):
        assert detect_exit_intent("I want to stop") is True

    def test_english_quit(self):
        assert detect_exit_intent("quit") is True

    def test_english_bye(self):
        assert detect_exit_intent("goodbye") is True

    def test_english_done(self):
        assert detect_exit_intent("I'm done") is True

    def test_english_cancel(self):
        assert detect_exit_intent("cancel") is True

    def test_spanish_parar(self):
        assert detect_exit_intent("quiero parar") is True

    def test_spanish_salir(self):
        assert detect_exit_intent("quiero salir") is True

    def test_spanish_adios(self):
        assert detect_exit_intent("adiós") is True

    def test_spanish_no_quiero(self):
        assert detect_exit_intent("no quiero continuar") is True

    def test_frontend_stop_signal(self):
        assert detect_exit_intent(FRONTEND_STOP_SIGNAL) is True

    def test_normal_name(self):
        assert detect_exit_intent("My name is Maria") is False

    def test_normal_city(self):
        assert detect_exit_intent("I live in Madrid") is False

    def test_normal_answer(self):
        assert detect_exit_intent("full-time") is False

    def test_stop_with_please(self):
        """'stop please' should match since it starts with 'stop'."""
        assert detect_exit_intent("stop please") is True

    def test_case_insensitive(self):
        assert detect_exit_intent("QUIT") is True
        assert detect_exit_intent("Bye") is True


class TestDetectOffensiveContent:
    def test_english_profanity(self):
        assert detect_offensive_content("this is fucking stupid") is True

    def test_english_slur(self):
        assert detect_offensive_content("you're a stupid bot") is True

    def test_spanish_profanity(self):
        assert detect_offensive_content("esto es una mierda") is True

    def test_spanish_insult(self):
        assert detect_offensive_content("eres un pendejo") is True

    def test_clean_english(self):
        assert detect_offensive_content("My name is John") is False

    def test_clean_spanish(self):
        assert detect_offensive_content("Me llamo Juan y vivo en Madrid") is False

    def test_partial_word_no_match(self):
        """Should not flag 'class' which contains 'ass'."""
        assert detect_offensive_content("I took a class yesterday") is False


class TestDetectConfirmation:
    def test_yes(self):
        assert detect_confirmation("yes") is True

    def test_si_accent(self):
        assert detect_confirmation("sí") is True

    def test_si_no_accent(self):
        assert detect_confirmation("si") is True

    def test_claro(self):
        assert detect_confirmation("claro") is True

    def test_ok(self):
        assert detect_confirmation("ok") is True

    def test_random_text(self):
        assert detect_confirmation("Madrid") is False


class TestDetectDenial:
    def test_no(self):
        assert detect_denial("no") is True

    def test_nope(self):
        assert detect_denial("nope") is True

    def test_random_text(self):
        assert detect_denial("yes please") is False


# ======================================================================
# Integration tests — full pipeline with mocked LLM
# ======================================================================

@pytest.mark.asyncio
async def test_exit_flow_with_confirmation():
    """User says 'quit' -> keyword guardrail detects exit intent -> confirmation -> confirms -> WITHDRAWN."""
    conv_id = _create_conversation()

    # No LLM responses needed: "quit" is caught by keyword guardrail,
    # and "si" is handled by the confirmation detection (no LLM call).
    llm_responses = []
    p1, p2, mock_llm, mock_finalize = _patch_agent(llm_responses)
    with p1, p2:
        # Step 1: User expresses exit intent — keyword guardrail catches it
        result = await process_message(conv_id, "quit")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")
        assert "sure" in result["response"].lower() or "seguro" in result["response"].lower()

        # Step 2: User confirms
        result = await process_message(conv_id, "si")
        assert result["status"] in (ConversationStatus.WITHDRAWN, "withdrawn")
        mock_finalize.assert_called_once()


@pytest.mark.asyncio
async def test_exit_flow_with_denial():
    """User says 'quit' -> keyword guardrail detects exit intent -> confirmation -> denial -> continues."""
    conv_id = _create_conversation()

    llm_responses = [
        # Only one LLM response needed: "quit" is caught by keyword guardrail
        # (no LLM call), and the denial message goes through normal LLM flow.
        _mock_llm_response("Pedro Ruiz", "¡Hola Pedro! ¿Tienes carnet de conducir?"),
    ]
    p1, p2, mock_llm, mock_finalize = _patch_agent(llm_responses)
    with p1, p2:
        # Step 1: User expresses exit intent — keyword guardrail catches it
        result = await process_message(conv_id, "quit")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")

        # Step 2: User changes mind — denial clears _pending_withdrawal, normal flow resumes
        result = await process_message(conv_id, "no, me llamo Pedro Ruiz")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")

    conv = storage.get_conversation(conv_id)
    assert conv["status"] == ConversationStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_frontend_stop_signal_immediate_withdrawal():
    """Frontend [STOP] signal -> immediate withdrawal, no confirmation."""
    conv_id = _create_conversation()

    p1, p2, mock_llm, mock_finalize = _patch_agent([])
    with p1, p2:
        result = await process_message(conv_id, FRONTEND_STOP_SIGNAL)
        assert result["status"] in (ConversationStatus.WITHDRAWN, "withdrawn")
        mock_finalize.assert_called_once()

    # LLM should NOT have been called
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_offensive_first_warning():
    """First offensive message -> keyword guardrail catches it -> warning, conversation continues."""
    conv_id = _create_conversation()

    # No LLM responses needed: keyword guardrail catches offensive content
    # before the LLM is called.
    llm_responses = []
    p1, p2, mock_llm, mock_finalize = _patch_agent(llm_responses)
    with p1, p2:
        result = await process_message(conv_id, "this is fucking stupid")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")
        # Should contain warning about keeping things respectful
        assert "respect" in result["response"].lower() or "respetuoso" in result["response"].lower()

    # Strike counter should be stored
    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("_offensive_strikes") == 1


@pytest.mark.asyncio
async def test_offensive_repeated_termination():
    """Repeated offensive messages -> keyword guardrail catches them -> DISQUALIFIED termination."""
    conv_id = _create_conversation()

    # No LLM responses needed: keyword guardrail catches offensive content
    # before the LLM is called on both messages.
    llm_responses = []
    p1, p2, mock_llm, mock_finalize = _patch_agent(llm_responses)
    with p1, p2:
        # Strike 1
        result = await process_message(conv_id, "you're a fucking idiot")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")

        # Strike 2 (hits MAX_OFFENSIVE_STRIKES=2)
        result = await process_message(conv_id, "go to hell you piece of shit")
        assert result["status"] in (ConversationStatus.DISQUALIFIED, "disqualified")
        mock_finalize.assert_called_once()

    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("terminated_offensive") is True


@pytest.mark.asyncio
async def test_offensive_then_normal_continues():
    """Offensive warning followed by normal message -> screening continues."""
    conv_id = _create_conversation()

    llm_responses = [
        # Only one LLM response needed: the first message ("this is bullshit")
        # is caught by the pre-LLM keyword guardrail and never reaches the LLM.
        _mock_llm_response("Maria Garcia", "¡Hola Maria! ¿Tienes carnet de conducir?"),
    ]
    p1, p2, mock_llm, mock_finalize = _patch_agent(llm_responses)
    with p1, p2:
        # Strike 1 — keyword guardrail catches offensive content, warning issued
        result = await process_message(conv_id, "this is bullshit")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")

        # Normal message — screening continues via LLM
        result = await process_message(conv_id, "Sorry, my name is Maria Garcia")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")


@pytest.mark.asyncio
async def test_exit_during_mid_screening():
    """User goes through some stages then decides to stop via keyword-detected exit intent."""
    conv_id = _create_conversation()

    llm_responses = [
        _mock_llm_response("Ana López", "¡Hola Ana! ¿Tienes carnet de conducir?"),
        _mock_llm_response(True, "¡Perfecto! ¿En qué ciudad vives?"),
        # No third LLM response needed: "no quiero continuar" is caught by the
        # pre-LLM keyword guardrail and never reaches the LLM.
    ]
    p1, p2, mock_llm, mock_finalize = _patch_agent(llm_responses)
    with p1, p2:
        # Answer a few questions
        await process_message(conv_id, "Soy Ana López")
        await process_message(conv_id, "Sí, tengo carnet")

        # Now decide to stop — keyword guardrail catches exit intent
        result = await process_message(conv_id, "no quiero continuar")
        assert result["status"] in (ConversationStatus.IN_PROGRESS, "in_progress")

        # Confirm
        result = await process_message(conv_id, "sí")
        assert result["status"] in (ConversationStatus.WITHDRAWN, "withdrawn")
        mock_finalize.assert_called_once()

    # The partial data should be preserved
    conv = storage.get_conversation(conv_id)
    assert conv["extracted_data"].get("full_name") == "Ana López"
    assert conv["extracted_data"].get("driver_license") is True


@pytest.mark.asyncio
async def test_terminal_conversation_rejects_messages():
    """A WITHDRAWN conversation should reject new messages."""
    conv_id = _create_conversation()

    p1, p2, mock_llm, mock_finalize = _patch_agent([])
    with p1, p2:
        # Withdraw immediately
        result = await process_message(conv_id, FRONTEND_STOP_SIGNAL)
        assert result["status"] in (ConversationStatus.WITHDRAWN, "withdrawn")

        # Try to send another message
        result = await process_message(conv_id, "hello")
        assert "already ended" in result["response"].lower() or "ended" in result["response"].lower()
        assert result["status"] in (ConversationStatus.WITHDRAWN, "withdrawn")
