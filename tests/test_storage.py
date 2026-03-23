"""Comprehensive tests for the storage layer (SQLite CRUD, analytics, reset, purge)."""
import sys
import os
import json
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import storage
from backend.config import ConversationStatus, ConversationStage, generate_conversation_id


# ---------------------------------------------------------------------------
# Fixture: fresh temp DB per test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("backend.storage.DB_PATH", db_path)
    storage.init_db()
    yield


def _make_conv(conv_id=None, **overrides):
    """Helper: create a conversation and return its id."""
    cid = conv_id or generate_conversation_id()
    storage.create_conversation(cid)
    if overrides:
        storage.update_conversation(cid, **overrides)
    return cid


# ===========================================================================
# init_db
# ===========================================================================

class TestInitDb:
    def test_creates_tables(self, tmp_path, monkeypatch):
        """init_db should create all three tables without error."""
        db_path = str(tmp_path / "fresh.db")
        monkeypatch.setattr("backend.storage.DB_PATH", db_path)
        storage.init_db()  # should not raise
        conn = storage.get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "conversations" in table_names
        assert "messages" in table_names
        assert "field_attempts" in table_names
        conn.close()

    def test_idempotent(self):
        """Calling init_db twice should not raise."""
        storage.init_db()
        storage.init_db()


# ===========================================================================
# create_conversation / get_conversation
# ===========================================================================

class TestCreateAndGet:
    def test_create_and_retrieve(self):
        cid = _make_conv()
        conv = storage.get_conversation(cid)
        assert conv is not None
        assert conv["id"] == cid
        assert conv["status"] == "in_progress"
        assert conv["stage"] == "greeting"
        assert conv["language"] == "es"
        assert conv["extracted_data"] == {}

    def test_get_nonexistent_returns_none(self):
        assert storage.get_conversation("nonexistent-id") is None

    def test_created_at_is_set(self):
        cid = _make_conv()
        conv = storage.get_conversation(cid)
        assert conv["created_at"] is not None
        # Should be parseable as ISO datetime
        datetime.fromisoformat(conv["created_at"])

    def test_duplicate_id_raises(self):
        cid = generate_conversation_id()
        storage.create_conversation(cid)
        with pytest.raises(Exception):  # IntegrityError
            storage.create_conversation(cid)


# ===========================================================================
# update_conversation
# ===========================================================================

class TestUpdateConversation:
    def test_update_status(self):
        cid = _make_conv()
        storage.update_conversation(cid, status=ConversationStatus.QUALIFIED.value)
        conv = storage.get_conversation(cid)
        assert conv["status"] == "qualified"

    def test_update_stage(self):
        cid = _make_conv()
        storage.update_conversation(cid, stage=ConversationStage.LICENSE.value)
        conv = storage.get_conversation(cid)
        assert conv["stage"] == "license"

    def test_update_multiple_fields(self):
        cid = _make_conv()
        storage.update_conversation(
            cid,
            status="disqualified",
            stage="city",
            language="en",
        )
        conv = storage.get_conversation(cid)
        assert conv["status"] == "disqualified"
        assert conv["stage"] == "city"
        assert conv["language"] == "en"

    def test_update_extracted_data_as_dict(self):
        cid = _make_conv()
        storage.update_conversation(cid, extracted_data={"full_name": "Test"})
        conv = storage.get_conversation(cid)
        assert conv["extracted_data"]["full_name"] == "Test"


# ===========================================================================
# update_extracted_field
# ===========================================================================

class TestUpdateExtractedField:
    def test_set_single_field(self):
        cid = _make_conv()
        storage.update_extracted_field(cid, "full_name", "Maria")
        conv = storage.get_conversation(cid)
        assert conv["extracted_data"]["full_name"] == "Maria"

    def test_set_multiple_fields_incrementally(self):
        cid = _make_conv()
        storage.update_extracted_field(cid, "full_name", "Maria")
        storage.update_extracted_field(cid, "driver_license", True)
        storage.update_extracted_field(cid, "experience_platforms", ["Glovo"])
        conv = storage.get_conversation(cid)
        assert conv["extracted_data"]["full_name"] == "Maria"
        assert conv["extracted_data"]["driver_license"] is True
        assert conv["extracted_data"]["experience_platforms"] == ["Glovo"]

    def test_overwrite_field(self):
        cid = _make_conv()
        storage.update_extracted_field(cid, "full_name", "Maria")
        storage.update_extracted_field(cid, "full_name", "Maria Garcia")
        conv = storage.get_conversation(cid)
        assert conv["extracted_data"]["full_name"] == "Maria Garcia"

    def test_nonexistent_conversation_no_error(self):
        """update_extracted_field on missing conv should not raise."""
        storage.update_extracted_field("nonexistent-id", "full_name", "X")


# ===========================================================================
# Messages (save / get)
# ===========================================================================

class TestMessages:
    def test_save_and_get(self):
        cid = _make_conv()
        storage.save_message(cid, "user", "Hello", "es")
        storage.save_message(cid, "assistant", "Hola!", "es")
        msgs = storage.get_messages(cid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"

    def test_messages_ordered_by_id(self):
        cid = _make_conv()
        for i in range(5):
            storage.save_message(cid, "user", f"msg-{i}", "es")
        msgs = storage.get_messages(cid)
        contents = [m["content"] for m in msgs]
        assert contents == [f"msg-{i}" for i in range(5)]

    def test_get_messages_empty(self):
        cid = _make_conv()
        msgs = storage.get_messages(cid)
        assert msgs == []

    def test_save_message_updates_last_message_at(self):
        cid = _make_conv()
        conv_before = storage.get_conversation(cid)
        assert conv_before["last_message_at"] is None

        storage.save_message(cid, "user", "Hi", "es")
        conv_after = storage.get_conversation(cid)
        assert conv_after["last_message_at"] is not None

    def test_messages_isolated_between_conversations(self):
        cid1 = _make_conv()
        cid2 = _make_conv()
        storage.save_message(cid1, "user", "conv1-msg", "es")
        storage.save_message(cid2, "user", "conv2-msg", "en")
        assert len(storage.get_messages(cid1)) == 1
        assert len(storage.get_messages(cid2)) == 1
        assert storage.get_messages(cid1)[0]["content"] == "conv1-msg"


# ===========================================================================
# Field attempts
# ===========================================================================

class TestFieldAttempts:
    def test_initial_count_is_zero(self):
        cid = _make_conv()
        assert storage.get_field_attempts(cid, "full_name") == 0

    def test_increment(self):
        cid = _make_conv()
        storage.increment_field_attempts(cid, "full_name")
        assert storage.get_field_attempts(cid, "full_name") == 1
        storage.increment_field_attempts(cid, "full_name")
        assert storage.get_field_attempts(cid, "full_name") == 2

    def test_independent_fields(self):
        cid = _make_conv()
        storage.increment_field_attempts(cid, "full_name")
        storage.increment_field_attempts(cid, "city_zone")
        storage.increment_field_attempts(cid, "city_zone")
        assert storage.get_field_attempts(cid, "full_name") == 1
        assert storage.get_field_attempts(cid, "city_zone") == 2

    def test_independent_conversations(self):
        cid1 = _make_conv()
        cid2 = _make_conv()
        storage.increment_field_attempts(cid1, "full_name")
        assert storage.get_field_attempts(cid1, "full_name") == 1
        assert storage.get_field_attempts(cid2, "full_name") == 0


# ===========================================================================
# Re-engagement helpers
# ===========================================================================

class TestReengagementStorage:
    def test_increment_reengagement_count(self):
        cid = _make_conv()
        storage.increment_reengagement_count(cid)
        conv = storage.get_conversation(cid)
        assert conv["reengagement_count"] == 1
        storage.increment_reengagement_count(cid)
        conv = storage.get_conversation(cid)
        assert conv["reengagement_count"] == 2

    def test_set_last_reengagement_at(self):
        cid = _make_conv()
        now = datetime.now(timezone.utc).isoformat()
        storage.set_last_reengagement_at(cid, now)
        conv = storage.get_conversation(cid)
        assert conv["last_reengagement_at"] == now

    def test_get_incomplete_conversations(self):
        # Create an old conversation with a last_message_at in the past
        cid = _make_conv()
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        storage.update_conversation(cid, last_message_at=old_time)

        result = storage.get_incomplete_conversations(timeout_minutes=30)
        assert len(result) == 1
        assert result[0]["id"] == cid

    def test_get_incomplete_skips_recent(self):
        cid = _make_conv()
        storage.save_message(cid, "user", "hi", "es")  # sets last_message_at to now
        result = storage.get_incomplete_conversations(timeout_minutes=30)
        assert len(result) == 0  # too recent

    def test_get_incomplete_skips_terminated(self):
        cid = _make_conv(status="qualified")
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        storage.update_conversation(cid, last_message_at=old_time)
        result = storage.get_incomplete_conversations(timeout_minutes=30)
        assert len(result) == 0  # not in_progress


# ===========================================================================
# delete_all (the new reset feature)
# ===========================================================================

class TestDeleteAll:
    def test_deletes_everything(self):
        cid1 = _make_conv()
        cid2 = _make_conv()
        storage.save_message(cid1, "user", "hello", "es")
        storage.save_message(cid2, "user", "hola", "es")
        storage.increment_field_attempts(cid1, "full_name")

        count = storage.delete_all()
        assert count == 2

        assert storage.get_conversation(cid1) is None
        assert storage.get_conversation(cid2) is None
        assert storage.get_messages(cid1) == []
        assert storage.get_field_attempts(cid1, "full_name") == 0

    def test_delete_all_empty_db(self):
        count = storage.delete_all()
        assert count == 0

    def test_delete_all_then_create_works(self):
        """DB should be usable after deletion."""
        _make_conv()
        storage.delete_all()
        cid = _make_conv()
        assert storage.get_conversation(cid) is not None


# ===========================================================================
# purge_old_conversations
# ===========================================================================

class TestPurgeOldConversations:
    def test_purge_old(self):
        cid = _make_conv()
        # Backdate created_at to 60 days ago
        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        conn = storage.get_connection()
        conn.execute("UPDATE conversations SET created_at = ? WHERE id = ?", (old_time, cid))
        conn.commit()
        conn.close()

        storage.save_message(cid, "user", "hi", "es")
        storage.increment_field_attempts(cid, "name")

        purged = storage.purge_old_conversations(retention_days=30)
        assert purged == 1
        assert storage.get_conversation(cid) is None
        assert storage.get_messages(cid) == []

    def test_purge_keeps_recent(self):
        cid = _make_conv()  # created just now
        purged = storage.purge_old_conversations(retention_days=30)
        assert purged == 0
        assert storage.get_conversation(cid) is not None

    def test_purge_selective(self):
        """Purges old, keeps recent."""
        old_cid = _make_conv()
        new_cid = _make_conv()

        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        conn = storage.get_connection()
        conn.execute("UPDATE conversations SET created_at = ? WHERE id = ?", (old_time, old_cid))
        conn.commit()
        conn.close()

        purged = storage.purge_old_conversations(retention_days=30)
        assert purged == 1
        assert storage.get_conversation(old_cid) is None
        assert storage.get_conversation(new_cid) is not None


# ===========================================================================
# build_extracted_data_from_db
# ===========================================================================

class TestBuildExtractedData:
    def test_returns_stored_data(self):
        cid = _make_conv()
        storage.update_extracted_field(cid, "full_name", "Maria")
        storage.update_extracted_field(cid, "driver_license", True)
        data = storage.build_extracted_data_from_db(cid)
        assert data["full_name"] == "Maria"
        assert data["driver_license"] is True

    def test_nonexistent_returns_empty(self):
        data = storage.build_extracted_data_from_db("nonexistent")
        assert data == {}


# ===========================================================================
# Analytics
# ===========================================================================

class TestAnalytics:
    def test_empty_db(self):
        data = storage.get_analytics()
        assert data["total_conversations"] == 0
        assert data["completion_rate"] == 0.0
        assert data["qualification_rate"] == 0.0
        assert data["by_status"] == {}
        assert data["avg_turns_qualified"] == 0.0

    def test_single_qualified(self):
        cid = _make_conv(status="qualified")
        storage.save_message(cid, "user", "hi", "es")
        storage.save_message(cid, "assistant", "hello", "es")

        data = storage.get_analytics()
        assert data["total_conversations"] == 1
        assert data["by_status"]["qualified"] == 1
        assert data["completion_rate"] == 1.0
        assert data["qualification_rate"] == 1.0
        assert data["avg_turns_qualified"] == 2.0

    def test_mixed_statuses(self):
        _make_conv(status="qualified")
        _make_conv(status="qualified")
        _make_conv(status="disqualified")
        _make_conv(status="in_progress")

        data = storage.get_analytics()
        assert data["total_conversations"] == 4
        assert data["by_status"]["qualified"] == 2
        assert data["by_status"]["disqualified"] == 1
        assert data["by_status"]["in_progress"] == 1
        assert data["completion_rate"] == 0.75  # 3/4
        assert data["qualification_rate"] == round(2 / 3, 4)

    def test_disqualification_reasons(self):
        cid1 = _make_conv(status="disqualified")
        storage.update_extracted_field(cid1, "disqualification_reason", "no_license")
        cid2 = _make_conv(status="disqualified")
        storage.update_extracted_field(cid2, "disqualification_reason", "outside_area")
        cid3 = _make_conv(status="disqualified")
        storage.update_extracted_field(cid3, "disqualification_reason", "no_license")

        data = storage.get_analytics()
        assert data["disqualification_reasons"]["no_license"] == 2
        assert data["disqualification_reasons"]["outside_area"] == 1

    def test_dropoff_by_stage(self):
        _make_conv(status="abandoned", stage="city")
        _make_conv(status="abandoned", stage="city")
        _make_conv(status="abandoned", stage="name")

        data = storage.get_analytics()
        assert data["dropoff_by_stage"]["city"] == 2
        assert data["dropoff_by_stage"]["name"] == 1

    def test_analytics_after_reset(self):
        _make_conv(status="qualified")
        _make_conv(status="disqualified")
        storage.delete_all()

        data = storage.get_analytics()
        assert data["total_conversations"] == 0
        assert data["by_status"] == {}
