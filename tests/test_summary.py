"""Tests for the two-pass extraction + summary module."""
import sys
import os
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import AsyncMock, patch
from backend import storage
from backend.config import generate_conversation_id
from backend.summary import (
    format_transcript,
    validate_extraction_schema,
    extract_and_summarize,
)


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("backend.storage.DB_PATH", db_path)
    monkeypatch.setattr("backend.config.DB_PATH", db_path)
    storage.init_db()
    yield


def _make_conv_with_messages():
    """Create a conversation with a realistic transcript."""
    cid = generate_conversation_id()
    storage.create_conversation(cid)
    storage.save_message(cid, "assistant", "Hola! Bienvenido. Como te llamas?", "es")
    storage.save_message(cid, "user", "Me llamo Maria Garcia", "es")
    storage.save_message(cid, "assistant", "Hola Maria! Tienes carnet de conducir?", "es")
    storage.save_message(cid, "user", "Si, tengo carnet", "es")
    return cid


# ===========================================================================
# format_transcript
# ===========================================================================

class TestFormatTranscript:
    def test_formats_messages(self):
        messages = [
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "How are you?"},
        ]
        result = format_transcript(messages)
        assert "Assistant: Hello!" in result
        assert "Candidate: Hi there" in result
        assert "Assistant: How are you?" in result

    def test_empty_messages(self):
        result = format_transcript([])
        assert result == ""

    def test_single_message(self):
        result = format_transcript([{"role": "user", "content": "Hello"}])
        assert result == "Candidate: Hello"

    def test_preserves_order(self):
        messages = [
            {"role": "assistant", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Third"},
        ]
        result = format_transcript(messages)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("Assistant:")
        assert lines[1].startswith("Candidate:")


# ===========================================================================
# validate_extraction_schema
# ===========================================================================

class TestValidateExtractionSchema:
    def test_valid_complete(self):
        data = {
            "full_name": "Maria Garcia",
            "driver_license": True,
            "city_zone": "Madrid",
            "availability": "full-time",
            "preferred_schedule": "morning",
            "experience_years": 2,
            "experience_platforms": ["Glovo"],
            "start_date": "2026-05-01",
            "disqualification_reason": None,
        }
        validate_extraction_schema(data)  # should not raise

    def test_valid_with_nulls(self):
        data = {
            "full_name": "Maria",
            "driver_license": None,
            "city_zone": None,
        }
        validate_extraction_schema(data)  # nulls are fine

    def test_valid_empty_dict(self):
        validate_extraction_schema({})  # missing fields OK

    def test_invalid_driver_license_type(self):
        data = {"driver_license": "yes"}  # should be bool
        with pytest.raises(ValueError, match="driver_license must be bool"):
            validate_extraction_schema(data)

    def test_invalid_experience_platforms_type(self):
        data = {"experience_platforms": "Glovo"}  # should be list
        with pytest.raises(ValueError, match="experience_platforms must be list"):
            validate_extraction_schema(data)

    def test_invalid_experience_years_type(self):
        data = {"experience_years": "two"}  # should be int
        with pytest.raises(ValueError, match="experience_years must be int"):
            validate_extraction_schema(data)

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="not a dict"):
            validate_extraction_schema("not a dict")

    def test_not_a_dict_list(self):
        with pytest.raises(ValueError, match="not a dict"):
            validate_extraction_schema([1, 2, 3])


# ===========================================================================
# extract_and_summarize
# ===========================================================================

class TestExtractAndSummarize:
    @pytest.mark.asyncio
    async def test_successful_extraction_and_summary(self):
        cid = _make_conv_with_messages()

        extraction_json = json.dumps({
            "full_name": "Maria Garcia",
            "driver_license": True,
            "city_zone": "Madrid",
            "availability": "full-time",
            "preferred_schedule": "morning",
            "experience_years": 2,
            "experience_platforms": ["Glovo"],
            "start_date": "2026-05-01",
            "disqualification_reason": None,
        })
        summary_text = "Maria Garcia is a qualified candidate from Madrid with 2 years experience."

        mock_llm = AsyncMock(side_effect=[extraction_json, summary_text])
        with patch("backend.agent.call_llm", mock_llm):
            result = await extract_and_summarize(cid)

        assert result["extracted_data"]["full_name"] == "Maria Garcia"
        assert result["extracted_data"]["driver_license"] is True
        assert result["summary"] == summary_text
        assert mock_llm.await_count == 2

    @pytest.mark.asyncio
    async def test_extraction_failure_falls_back_to_db(self):
        cid = _make_conv_with_messages()
        storage.update_extracted_field(cid, "full_name", "Maria Garcia")
        storage.update_extracted_field(cid, "driver_license", True)

        summary_text = "Summary from DB fallback."
        mock_llm = AsyncMock(side_effect=[
            "not valid json!!!",  # extraction fails
            summary_text,         # summary succeeds
        ])

        with patch("backend.agent.call_llm", mock_llm):
            result = await extract_and_summarize(cid)

        # Should fall back to DB data
        assert result["extracted_data"]["full_name"] == "Maria Garcia"
        assert result["summary"] == summary_text

    @pytest.mark.asyncio
    async def test_summary_failure_returns_fallback_text(self):
        cid = _make_conv_with_messages()
        storage.update_conversation(cid, status="qualified")

        extraction_json = json.dumps({"full_name": "Maria", "driver_license": True})
        mock_llm = AsyncMock(side_effect=[
            extraction_json,
            Exception("LLM summary error"),
        ])

        with patch("backend.agent.call_llm", mock_llm):
            result = await extract_and_summarize(cid)

        assert "Summary unavailable" in result["summary"]
        assert result["extracted_data"]["full_name"] == "Maria"

    @pytest.mark.asyncio
    async def test_both_passes_fail(self):
        cid = _make_conv_with_messages()
        storage.update_conversation(cid, status="in_progress")
        storage.update_extracted_field(cid, "full_name", "DB Fallback")

        mock_llm = AsyncMock(side_effect=[
            Exception("extraction error"),
            Exception("summary error"),
        ])

        with patch("backend.agent.call_llm", mock_llm):
            result = await extract_and_summarize(cid)

        assert result["extracted_data"]["full_name"] == "DB Fallback"
        assert "Summary unavailable" in result["summary"]

    @pytest.mark.asyncio
    async def test_extraction_with_clean_json(self):
        cid = _make_conv_with_messages()

        extraction_json = '{"full_name": "Maria", "driver_license": true}'
        summary_text = "Maria is a candidate."

        mock_llm = AsyncMock(side_effect=[extraction_json, summary_text])
        with patch("backend.agent.call_llm", mock_llm):
            result = await extract_and_summarize(cid)

        assert result["extracted_data"]["full_name"] == "Maria"

    @pytest.mark.asyncio
    async def test_extraction_schema_violation_falls_back(self):
        cid = _make_conv_with_messages()
        storage.update_extracted_field(cid, "full_name", "From DB")

        # driver_license as string instead of bool
        bad_json = json.dumps({"driver_license": "yes", "full_name": "Maria"})
        summary_text = "Summary text."

        mock_llm = AsyncMock(side_effect=[bad_json, summary_text])
        with patch("backend.agent.call_llm", mock_llm):
            result = await extract_and_summarize(cid)

        # Schema validation fails -> falls back to DB
        assert result["extracted_data"]["full_name"] == "From DB"

    @pytest.mark.asyncio
    async def test_malformed_summary_uses_deterministic_fallback(self):
        cid = _make_conv_with_messages()
        storage.update_conversation(cid, status="qualified")

        extraction_json = json.dumps({
            "full_name": "Luca",
            "driver_license": True,
            "city_zone": "Madrid",
            "availability": "full-time",
            "preferred_schedule": "morning",
            "experience_years": 3,
            "experience_platforms": ["Glovo", "Uber Eats"],
            "start_date": "ASAP",
            "disqualification_reason": None,
        })
        malformed_summary = (
            "Lucaconfirmed he holds a valid driver’s license, resides in Madrid "
            "(within the service area), and is available for fulltime work with a morning preference. "
            "He reported three years of deliverydriver experience, specifically with Glovo and Uber Eats, "
            "and indicated he could start as soon as possible. No disqualifying issues were raised during "
            "the conversation. Based on these signals, Luca is qualified for the repartidor position."
        )

        mock_llm = AsyncMock(side_effect=[extraction_json, malformed_summary])
        with patch("backend.agent.call_llm", mock_llm):
            result = await extract_and_summarize(cid)

        assert "Lucaconfirmed" not in result["summary"]
        assert "deliverydriver" not in result["summary"]
        assert "fulltime" not in result["summary"]
        assert result["summary"].startswith("Luca confirmed a valid driver's license")
