"""Tests for the re-engagement module (scheduled + manual triggers)."""
import sys
import os
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
from backend import storage
from backend.config import (
    generate_conversation_id,
    REENGAGEMENT_MAX_ATTEMPTS,
    CONVERSATION_ABANDON_HOURS,
)
from backend.reengagement import (
    get_reengagement_message,
    check_and_reengage,
    reengage_conversation,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("backend.storage.DB_PATH", db_path)
    monkeypatch.setattr("backend.config.DB_PATH", db_path)
    storage.init_db()
    yield


def _make_conv_with_message(minutes_ago=60, language="es", **overrides):
    """Create a conversation with a last_message_at in the past."""
    cid = generate_conversation_id()
    storage.create_conversation(cid)
    past = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    storage.update_conversation(cid, last_message_at=past, language=language, **overrides)
    return cid


# ===========================================================================
# get_reengagement_message
# ===========================================================================

class TestReengagementMessage:
    def test_spanish(self):
        msg = get_reengagement_message("es")
        assert "Hola" in msg or "hola" in msg
        assert "continuar" in msg or "mitad" in msg

    def test_english(self):
        msg = get_reengagement_message("en")
        assert "Hi" in msg
        assert "pick up" in msg

    def test_defaults_to_spanish(self):
        msg = get_reengagement_message("fr")  # unknown lang
        assert "Hola" in msg or "hola" in msg


# ===========================================================================
# check_and_reengage (scheduled job)
# ===========================================================================

class TestCheckAndReengage:
    @pytest.mark.asyncio
    async def test_sends_reengagement_to_inactive(self):
        cid = _make_conv_with_message(minutes_ago=60)
        msgs_before = len(storage.get_messages(cid))

        await check_and_reengage(timeout_minutes=30)

        msgs_after = len(storage.get_messages(cid))
        assert msgs_after == msgs_before + 1
        conv = storage.get_conversation(cid)
        assert conv["reengagement_count"] == 1

    @pytest.mark.asyncio
    async def test_skips_recent_conversations(self):
        cid = _make_conv_with_message(minutes_ago=5)  # very recent
        await check_and_reengage(timeout_minutes=30)

        conv = storage.get_conversation(cid)
        assert conv["reengagement_count"] == 0  # no re-engagement sent

    @pytest.mark.asyncio
    async def test_skips_terminated_conversations(self):
        cid = _make_conv_with_message(minutes_ago=60, status="qualified")
        await check_and_reengage(timeout_minutes=30)

        conv = storage.get_conversation(cid)
        assert conv["reengagement_count"] == 0

    @pytest.mark.asyncio
    async def test_abandons_after_threshold(self):
        hours = CONVERSATION_ABANDON_HOURS + 1
        cid = _make_conv_with_message(minutes_ago=hours * 60)

        await check_and_reengage(timeout_minutes=30)

        conv = storage.get_conversation(cid)
        assert conv["status"] == "abandoned"

    @pytest.mark.asyncio
    async def test_skips_max_reengagement_attempts(self):
        cid = _make_conv_with_message(minutes_ago=60)
        # Simulate max attempts already reached
        for _ in range(REENGAGEMENT_MAX_ATTEMPTS):
            storage.increment_reengagement_count(cid)

        msgs_before = len(storage.get_messages(cid))
        await check_and_reengage(timeout_minutes=30)

        msgs_after = len(storage.get_messages(cid))
        assert msgs_after == msgs_before  # no new message

    @pytest.mark.asyncio
    async def test_handles_null_last_message_at(self):
        """Conversations with no last_message_at should be skipped gracefully."""
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        # Don't set last_message_at, but hack it into incomplete results
        # Actually, get_incomplete_conversations requires last_message_at IS NOT NULL
        # So this test just ensures no crash on empty results
        await check_and_reengage(timeout_minutes=30)

    @pytest.mark.asyncio
    async def test_multiple_conversations(self):
        cid1 = _make_conv_with_message(minutes_ago=60, language="es")
        cid2 = _make_conv_with_message(minutes_ago=90, language="en")

        await check_and_reengage(timeout_minutes=30)

        conv1 = storage.get_conversation(cid1)
        conv2 = storage.get_conversation(cid2)
        assert conv1["reengagement_count"] == 1
        assert conv2["reengagement_count"] == 1

        # Check language-appropriate messages
        msgs1 = storage.get_messages(cid1)
        msgs2 = storage.get_messages(cid2)
        assert any("continuar" in m["content"] or "mitad" in m["content"] for m in msgs1)
        assert any("pick up" in m["content"] for m in msgs2)


# ===========================================================================
# reengage_conversation (manual trigger)
# ===========================================================================

class TestManualReengage:
    @pytest.mark.asyncio
    async def test_manual_reengage_success(self):
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        msg = await reengage_conversation(cid)
        assert msg is not None
        assert len(msg) > 0

    @pytest.mark.asyncio
    async def test_manual_reengage_nonexistent(self):
        msg = await reengage_conversation("nonexistent-id")
        assert msg is None

    @pytest.mark.asyncio
    async def test_manual_reengage_terminal_fails(self):
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        storage.update_conversation(cid, status="qualified")
        msg = await reengage_conversation(cid)
        assert msg is None

    @pytest.mark.asyncio
    async def test_manual_reengage_increments_count(self):
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        await reengage_conversation(cid)
        await reengage_conversation(cid)
        conv = storage.get_conversation(cid)
        assert conv["reengagement_count"] == 2

    @pytest.mark.asyncio
    async def test_manual_reengage_uses_conversation_language(self):
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        storage.update_conversation(cid, language="en")
        msg = await reengage_conversation(cid)
        assert "pick up" in msg  # English message

    @pytest.mark.asyncio
    async def test_manual_reengage_saves_message(self):
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        await reengage_conversation(cid)
        msgs = storage.get_messages(cid)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_manual_reengage_sets_timestamp(self):
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        await reengage_conversation(cid)
        conv = storage.get_conversation(cid)
        assert conv["last_reengagement_at"] is not None
