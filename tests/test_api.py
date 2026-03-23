"""Integration tests for all FastAPI endpoints (positive + negative cases)."""

import sys
import os
import json

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import AsyncMock, patch
from backend import storage
from backend.config import (
    ConversationStatus,
    ConversationStage,
    generate_conversation_id,
)


# ---------------------------------------------------------------------------
# Fixture: fresh DB + test client per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("backend.storage.DB_PATH", db_path)
    monkeypatch.setattr("backend.config.DB_PATH", db_path)
    storage.init_db()
    yield


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app, raise_server_exceptions=False)


def _mock_llm_response(field_value, response_text):
    return {
        "field_value": field_value,
        "response": response_text,
        "detected_language": "es",
        "exit_intent": False,
        "is_offensive": False,
        "sentiment": "neutral",
    }


# ===========================================================================
# POST /conversations
# ===========================================================================


class TestCreateConversation:
    def test_create_success(self, client):
        resp = client.post("/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "greeting_message" in data
        assert len(data["conversation_id"]) > 0
        assert "Grupo Sazón" in data["greeting_message"]

    def test_create_multiple(self, client):
        r1 = client.post("/conversations").json()
        r2 = client.post("/conversations").json()
        assert r1["conversation_id"] != r2["conversation_id"]

    def test_conversation_persisted(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        conv = storage.get_conversation(cid)
        assert conv is not None
        assert conv["stage"] == "name"  # advanced past greeting
        assert conv["status"] == "in_progress"

    def test_greeting_saved_as_message(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        msgs = storage.get_messages(cid)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"


# ===========================================================================
# POST /conversations/{id}/messages
# ===========================================================================


class TestSendMessage:
    def test_send_message_success(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        llm_resp = _mock_llm_response("Maria Garcia", "Hola Maria!")

        with patch(
            "backend.agent.call_llm", new_callable=AsyncMock, return_value=llm_resp
        ):
            resp = client.post(
                f"/conversations/{cid}/messages",
                json={"message": "Me llamo Maria Garcia"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "status" in data
        assert "stage" in data

    def test_send_to_nonexistent_conversation(self, client):
        resp = client.post(
            "/conversations/nonexistent/messages",
            json={"message": "hello"},
        )
        assert resp.status_code == 404

    def test_send_empty_message(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        resp = client.post(
            f"/conversations/{cid}/messages",
            json={"message": ""},
        )
        assert resp.status_code == 422  # Pydantic validation (min_length=1)

    def test_send_too_long_message(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        resp = client.post(
            f"/conversations/{cid}/messages",
            json={"message": "x" * 451},
        )
        assert resp.status_code == 422  # max_length=450

    def test_missing_message_field(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        resp = client.post(f"/conversations/{cid}/messages", json={})
        assert resp.status_code == 422

    def test_response_advances_stage(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        llm_resp = _mock_llm_response("Pedro Sanchez", "Do you have a license?")

        with patch(
            "backend.agent.call_llm", new_callable=AsyncMock, return_value=llm_resp
        ):
            resp = client.post(
                f"/conversations/{cid}/messages",
                json={"message": "Me llamo Pedro Sanchez"},
            )
        data = resp.json()
        assert data["stage"] == "license"  # advanced from name

    def test_terminal_conversation_rejects(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        storage.update_conversation(cid, status=ConversationStatus.QUALIFIED.value)

        resp = client.post(
            f"/conversations/{cid}/messages",
            json={"message": "hello again"},
        )
        assert resp.status_code == 200
        assert "already ended" in resp.json()["response"]


# ===========================================================================
# GET /conversations/{id}
# ===========================================================================


class TestGetConversation:
    def test_get_existing(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        resp = client.get(f"/conversations/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == cid
        assert data["status"] == "in_progress"
        assert "extracted_data" in data

    def test_get_nonexistent(self, client):
        resp = client.get("/conversations/nonexistent-id")
        assert resp.status_code == 404

    def test_get_with_extracted_data(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        storage.update_extracted_field(cid, "full_name", "Test User")
        resp = client.get(f"/conversations/{cid}")
        assert resp.json()["extracted_data"]["full_name"] == "Test User"


# ===========================================================================
# GET /conversations/{id}/messages
# ===========================================================================


class TestGetMessages:
    def test_get_messages(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        resp = client.get(f"/conversations/{cid}/messages")
        assert resp.status_code == 200
        msgs = resp.json()
        assert len(msgs) >= 1  # at least the greeting
        assert msgs[0]["role"] == "assistant"

    def test_get_messages_nonexistent(self, client):
        resp = client.get("/conversations/nonexistent-id/messages")
        assert resp.status_code == 404

    def test_messages_grow_after_exchange(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        llm_resp = _mock_llm_response("Ana Lopez", "Do you have a license?")

        with patch(
            "backend.agent.call_llm", new_callable=AsyncMock, return_value=llm_resp
        ):
            client.post(
                f"/conversations/{cid}/messages", json={"message": "Soy Ana Lopez"}
            )

        msgs = client.get(f"/conversations/{cid}/messages").json()
        assert len(msgs) == 3  # greeting + user msg + assistant reply


# ===========================================================================
# POST /conversations/{id}/reengage
# ===========================================================================


class TestReengage:
    def test_reengage_in_progress(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        resp = client.post(f"/conversations/{cid}/reengage")
        assert resp.status_code == 200
        assert "message" in resp.json()

    def test_reengage_terminal_fails(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        storage.update_conversation(cid, status="qualified")
        resp = client.post(f"/conversations/{cid}/reengage")
        assert resp.status_code == 400

    def test_reengage_nonexistent(self, client):
        resp = client.post("/conversations/nonexistent-id/reengage")
        assert resp.status_code == 400

    def test_reengage_increments_count(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        client.post(f"/conversations/{cid}/reengage")
        conv = storage.get_conversation(cid)
        assert conv["reengagement_count"] == 1

    def test_reengage_adds_message(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        msgs_before = len(storage.get_messages(cid))
        client.post(f"/conversations/{cid}/reengage")
        msgs_after = len(storage.get_messages(cid))
        assert msgs_after == msgs_before + 1


# ===========================================================================
# GET /analytics
# ===========================================================================


class TestAnalyticsEndpoint:
    def test_analytics_empty(self, client):
        resp = client.get("/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_conversations"] == 0

    def test_analytics_after_conversations(self, client):
        client.post("/conversations")
        client.post("/conversations")
        resp = client.get("/analytics")
        data = resp.json()
        assert data["total_conversations"] == 2
        assert data["by_status"]["in_progress"] == 2

    def test_analytics_schema(self, client):
        resp = client.get("/analytics")
        data = resp.json()
        required_keys = {
            "total_conversations",
            "total_today",
            "by_status",
            "completion_rate",
            "qualification_rate",
            "disqualification_reasons",
            "dropoff_by_stage",
            "avg_turns_qualified",
            "avg_turns_disqualified",
        }
        assert required_keys.issubset(set(data.keys()))


# ===========================================================================
# DELETE /reset
# ===========================================================================


class TestResetEndpoint:
    def test_reset_empty(self, client):
        resp = client.delete("/reset")
        assert resp.status_code == 200
        assert resp.json()["deleted_conversations"] == 0

    def test_reset_clears_data(self, client):
        cid1 = client.post("/conversations").json()["conversation_id"]
        cid2 = client.post("/conversations").json()["conversation_id"]

        resp = client.delete("/reset")
        assert resp.status_code == 200
        assert resp.json()["deleted_conversations"] == 2

        # Verify everything is gone
        assert storage.get_conversation(cid1) is None
        assert storage.get_conversation(cid2) is None

    def test_reset_clears_analytics(self, client):
        client.post("/conversations")
        client.post("/conversations")
        client.delete("/reset")

        analytics = client.get("/analytics").json()
        assert analytics["total_conversations"] == 0

    def test_reset_then_create_works(self, client):
        """App should work normally after a reset."""
        client.post("/conversations")
        client.delete("/reset")
        resp = client.post("/conversations")
        assert resp.status_code == 200
        assert "conversation_id" in resp.json()

    def test_reset_clears_messages(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        storage.save_message(cid, "user", "hi", "es")

        client.delete("/reset")
        assert storage.get_messages(cid) == []


# ===========================================================================
# GET / (frontend)
# ===========================================================================


class TestFrontend:
    def test_serve_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_static_js(self, client):
        resp = client.get("/static/app.js")
        assert resp.status_code == 200


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_invalid_json_body(self, client):
        cid = client.post("/conversations").json()["conversation_id"]
        resp = client.post(
            f"/conversations/{cid}/messages",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_wrong_http_method(self, client):
        # resp = client.get("/conversations")  # should be POST
        # assert resp.status_code in (404, 405)
        resp = client.put("/conversations")  # PUT is not allowed
        assert resp.status_code == 405

    def test_concurrent_conversations_isolated(self, client):
        """Two conversations should not interfere with each other."""
        cid1 = client.post("/conversations").json()["conversation_id"]
        cid2 = client.post("/conversations").json()["conversation_id"]

        llm_resp1 = _mock_llm_response("User One", "Hello One!")
        llm_resp2 = _mock_llm_response("User Two", "Hello Two!")

        with patch(
            "backend.agent.call_llm", new_callable=AsyncMock, return_value=llm_resp1
        ):
            client.post(
                f"/conversations/{cid1}/messages", json={"message": "I am User One"}
            )

        with patch(
            "backend.agent.call_llm", new_callable=AsyncMock, return_value=llm_resp2
        ):
            client.post(
                f"/conversations/{cid2}/messages", json={"message": "I am User Two"}
            )

        conv1 = storage.get_conversation(cid1)
        conv2 = storage.get_conversation(cid2)
        assert conv1["extracted_data"]["full_name"] == "User One"
        assert conv2["extracted_data"]["full_name"] == "User Two"
