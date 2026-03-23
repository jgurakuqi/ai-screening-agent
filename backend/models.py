"""Pydantic request/response models for the FastAPI endpoints and LLM output parsing."""

from pydantic import BaseModel, Field

from backend.config import ConversationStatus, ConversationStage


# --- Request models ---


class MessageRequest(BaseModel):
    """Incoming user message sent to the screening agent."""

    message: str = Field(..., min_length=1, max_length=450, description="User message text")
    whisper_language: str | None = Field(default=None, description="Language detected by Whisper STT")


class TTSRequest(BaseModel):
    """Text-to-speech synthesis request."""

    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    language: str = Field(default="es", pattern=r"^(es|en)$", description="Language code")


# --- Response models ---


class CreateConversationResponse(BaseModel):
    """Returned when a new screening conversation is created."""

    conversation_id: str
    greeting_message: str


class MessageResponse(BaseModel):
    """Agent reply to a user message, including updated conversation state."""

    response: str
    status: ConversationStatus
    stage: ConversationStage
    extracted_data: dict
    language: str = "es"


class ConversationResponse(BaseModel):
    """Full conversation record including extracted data and summary."""

    id: str
    status: ConversationStatus
    stage: ConversationStage
    language: str
    created_at: str
    last_message_at: str | None
    extracted_data: dict
    summary: str | None


class ConversationListItem(BaseModel):
    """Lightweight conversation entry for the sidebar listing."""

    id: str
    status: ConversationStatus
    display_name: str
    last_activity: str


class MessageRecord(BaseModel):
    """Single message within a conversation history."""

    role: str
    content: str
    language: str | None
    created_at: str


class AnalyticsResponse(BaseModel):
    """Aggregate analytics across all screening conversations."""

    total_conversations: int
    total_today: int
    by_status: dict[str, int]
    completion_rate: float
    qualification_rate: float
    disqualification_reasons: dict[str, int]
    dropoff_by_stage: dict[str, int]
    avg_turns_qualified: float
    avg_turns_disqualified: float
    avg_turns_needs_review: float
    avg_sentiment: float | None = None


# --- Internal model for LLM JSON output ---


class LLMScreeningResponse(BaseModel):
    """Parses the structured output from the conversational LLM (via function calling)."""

    field_value: str | bool | int | float | list | None = None  # backward compat
    extracted_fields: dict = Field(default_factory=dict)
    response: str
    detected_language: str | None = None
    response_language: str | None = None
    exit_intent: bool = False
    is_offensive: bool = False
    sentiment: str | None = None
