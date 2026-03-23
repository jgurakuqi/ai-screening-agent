"""Tests for privacy module: input sanitization, event logging, data retention."""
import sys
import os
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.privacy import sanitize_input, log_event, run_data_retention
from backend import storage


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("backend.storage.DB_PATH", db_path)
    monkeypatch.setattr("backend.config.DB_PATH", db_path)
    storage.init_db()
    yield


# ===========================================================================
# sanitize_input
# ===========================================================================

class TestSanitizeInput:
    def test_normal_text_unchanged(self):
        assert sanitize_input("Hello, my name is Maria") == "Hello, my name is Maria"

    def test_strips_control_characters(self):
        # \x00 (null), \x07 (bell), \x1f (unit separator)
        result = sanitize_input("Hello\x00World\x07!\x1f")
        assert result == "HelloWorld!"

    def test_preserves_newlines_and_tabs(self):
        # \n (0x0a) and \t (0x09) and \r (0x0d) are NOT in the control char range stripped
        result = sanitize_input("Line1\nLine2\tTabbed")
        assert "\n" in result
        assert "\t" in result

    def test_truncates_to_max_length(self):
        long_text = "a" * 600
        result = sanitize_input(long_text)
        assert len(result) == 450

    def test_exactly_max_length_not_truncated(self):
        text = "b" * 450
        result = sanitize_input(text)
        assert len(result) == 450

    def test_strips_whitespace(self):
        result = sanitize_input("  hello  ")
        assert result == "hello"

    def test_empty_string(self):
        result = sanitize_input("")
        assert result == ""

    def test_only_control_chars(self):
        result = sanitize_input("\x00\x01\x02\x03")
        assert result == ""

    def test_unicode_preserved(self):
        result = sanitize_input("Hola, soy María García")
        assert result == "Hola, soy María García"

    def test_emoji_preserved(self):
        result = sanitize_input("Hello 👋🏡")
        assert "👋" in result


# ===========================================================================
# log_event (smoke test — should not raise)
# ===========================================================================

class TestLogEvent:
    def test_log_event_does_not_raise(self):
        log_event("abc12345-xxxx", "name", "message_received")

    def test_log_event_truncates_id(self):
        """log_event uses first 8 chars of conv_id — should not crash on short ids."""
        log_event("short", "name", "test")


# ===========================================================================
# run_data_retention
# ===========================================================================

class TestDataRetention:
    def test_purges_old_conversations(self, monkeypatch):
        monkeypatch.setattr("backend.privacy.DATA_RETENTION_DAYS", 7)

        # Create an old conversation
        cid = "test-old-conv"
        storage.create_conversation(cid)
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        conn = storage.get_connection()
        conn.execute("UPDATE conversations SET created_at = ? WHERE id = ?", (old_time, cid))
        conn.commit()
        conn.close()

        count = run_data_retention()
        assert count == 1
        assert storage.get_conversation(cid) is None

    def test_keeps_recent(self, monkeypatch):
        monkeypatch.setattr("backend.privacy.DATA_RETENTION_DAYS", 30)
        cid = "test-recent-conv"
        storage.create_conversation(cid)

        count = run_data_retention()
        assert count == 0
        assert storage.get_conversation(cid) is not None

    def test_returns_zero_on_empty_db(self, monkeypatch):
        monkeypatch.setattr("backend.privacy.DATA_RETENTION_DAYS", 30)
        assert run_data_retention() == 0
