"""Application configuration, enums, constants, and stage-flow helpers.

Loads settings from environment variables (with sensible defaults) and defines
the conversation stage machine used by the screening agent.
"""

import os
import uuid
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


# --- Environment variables ---

# --- LLM provider selection ---
# Set LLM_PROVIDER to "azure" to use Azure AI Foundry, or "openrouter" for OpenRouter.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")

# --- Azure OpenAI configuration ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_PRIMARY_MODEL = os.getenv("AZURE_PRIMARY_MODEL", "gpt-5.4-mini")
AZURE_FALLBACK_MODEL = os.getenv("AZURE_FALLBACK_MODEL", "")

# --- OpenRouter configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "google/gemma-2-9b-it:free")
FALLBACK_MODEL_2 = os.getenv("FALLBACK_MODEL_2", "mistralai/mistral-7b-instruct:free")
# Additional free-tier fallback models (comma-separated env override)
EXTRA_FREE_MODELS: list[str] = [
    m.strip()
    for m in os.getenv(
        "EXTRA_FREE_MODELS",
        ",".join(
            [
                "mistralai/mistral-small-3.1-24b-instruct:free",
                "google/gemma-3-27b-it:free",
                "google/gemma-3-12b-it:free",
                "google/gemma-3-4b-it:free",
                "qwen/qwen3-coder:free",
                "openai/gpt-oss-120b:free",
                "openai/gpt-oss-20b:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "qwen/qwen3-next-80b-a3b-instruct:free",
                "minimax/minimax-m2.5:free",
                "nvidia/nemotron-nano-12b-v2-vl:free",
                "nvidia/nemotron-3-nano-30b-a3b:free",
                "nvidia/nemotron-3-super-120b-a12b:free",
                "nvidia/nemotron-nano-9b-v2:free",
                "stepfun/step-3.5-flash:free",
                "z-ai/glm-4.5-air:free",
                "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
                "liquid/lfm-2.5-1.2b-thinking:free",
                "arcee-ai/trinity-large-preview:free",
            ]
        ),
    ).split(",")
    if m.strip()
]
DB_PATH = os.getenv("DB_PATH", "data/conversations.db")
REENGAGEMENT_TIMEOUT_MINUTES = float(os.getenv("REENGAGEMENT_TIMEOUT_MINUTES", "30"))
REENGAGEMENT_MAX_ATTEMPTS = int(os.getenv("REENGAGEMENT_MAX_ATTEMPTS", "3"))
CONVERSATION_ABANDON_HOURS = float(os.getenv("CONVERSATION_ABANDON_HOURS", "24"))
MAX_CLARIFICATION_ATTEMPTS = int(os.getenv("MAX_CLARIFICATION_ATTEMPTS", "3"))
ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "true").lower() == "true"
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "30"))
LLM_MAX_RETRIES_PER_MODEL = int(os.getenv("LLM_MAX_RETRIES_PER_MODEL", "3"))
LLM_RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "2.0"))
LLM_RATE_LIMIT_BASE_DELAY_SECONDS = float(
    os.getenv("LLM_RATE_LIMIT_BASE_DELAY_SECONDS", "12.0")
)
LLM_RATE_LIMIT_MAX_WAIT_SECONDS = float(
    os.getenv("LLM_RATE_LIMIT_MAX_WAIT_SECONDS", "60.0")
)
LLM_PASS_BASE_DELAY_SECONDS = float(os.getenv("LLM_PASS_BASE_DELAY_SECONDS", "15.0"))
LLM_PASS_MAX_WAIT_SECONDS = float(os.getenv("LLM_PASS_MAX_WAIT_SECONDS", "300.0"))
LLM_MAX_FULL_PASSES = int(os.getenv("LLM_MAX_FULL_PASSES", "0"))
LLM_JITTER_FRACTION = float(os.getenv("LLM_JITTER_FRACTION", "0.3"))
LLM_EMPTY_CONTENT_COOLDOWN_SECONDS = int(
    os.getenv("LLM_EMPTY_CONTENT_COOLDOWN_SECONDS", "600")
)
MAX_OFFENSIVE_STRIKES = int(os.getenv("MAX_OFFENSIVE_STRIKES", "2"))

# --- TTS configuration ---
TTS_PROVIDER = os.getenv(
    "TTS_PROVIDER", "edge-tts"
)  # "edge-tts" | "browser"
TTS_EDGE_VOICE_ES = os.getenv("TTS_EDGE_VOICE_ES", "es-MX-DaliaNeural")
TTS_EDGE_VOICE_EN = os.getenv("TTS_EDGE_VOICE_EN", "en-US-JennyNeural")

# --- STT configuration ---
STT_MODEL_SIZE = os.getenv(
    "STT_MODEL_SIZE", "small"
)  # "tiny","base","small","medium","large-v3"
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")  # "cpu" or "cuda"
STT_COMPUTE_TYPE = os.getenv(
    "STT_COMPUTE_TYPE", "int8"
)  # "int8" fastest on CPU, "float16" for GPU
STT_AUTO_LANGUAGE_THRESHOLD = float(os.getenv("STT_AUTO_LANGUAGE_THRESHOLD", "0.65"))
STT_VAD_MIN_SILENCE_MS = int(os.getenv("STT_VAD_MIN_SILENCE_MS", "500"))


# --- Enums ---


class ConversationStatus(str, Enum):
    """Terminal and non-terminal states a screening conversation can be in."""

    IN_PROGRESS = "in_progress"
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    NEEDS_REVIEW = "needs_review"
    ABANDONED = "abandoned"
    WITHDRAWN = "withdrawn"


class ConversationStage(str, Enum):
    """Ordered stages the screening conversation progresses through."""

    GREETING = "greeting"
    NAME = "name"
    LICENSE = "license"
    CITY = "city"
    AVAILABILITY = "availability"
    SCHEDULE = "schedule"
    EXPERIENCE_YEARS = "experience_years"
    EXPERIENCE_PLATFORM = "experience_platform"
    START_DATE = "start_date"
    CLOSING = "closing"


class Language(str, Enum):
    """Supported conversation languages."""

    ES = "es"
    EN = "en"


# --- Constants ---

STAGE_ORDER = [
    ConversationStage.GREETING,
    ConversationStage.NAME,
    ConversationStage.LICENSE,
    ConversationStage.CITY,
    ConversationStage.AVAILABILITY,
    ConversationStage.SCHEDULE,
    ConversationStage.EXPERIENCE_YEARS,
    ConversationStage.EXPERIENCE_PLATFORM,  # conditionally skipped if years == 0
    ConversationStage.START_DATE,
    ConversationStage.CLOSING,
]

HARD_DISQUALIFIERS = {ConversationStage.LICENSE, ConversationStage.CITY}

REQUIRED_FIELDS = {
    "full_name",
    "driver_license",
    "city_zone",
    "availability",
    "preferred_schedule",
    "experience_years",
    "experience_platforms",
    "start_date",
}

KNOWN_PLATFORMS = [
    "Glovo",
    "Uber Eats",
    "Deliveroo",
    "Just Eat",
    "Rappi",
    "Stuart",
    "Amazon Flex",
    "DoorDash",
    "Didi Food",
]

# Map stages to their extracted_data field names
STAGE_TO_FIELD = {
    ConversationStage.NAME: "full_name",
    ConversationStage.LICENSE: "driver_license",
    ConversationStage.CITY: "city_zone",
    ConversationStage.AVAILABILITY: "availability",
    ConversationStage.SCHEDULE: "preferred_schedule",
    ConversationStage.EXPERIENCE_YEARS: "experience_years",
    ConversationStage.EXPERIENCE_PLATFORM: "experience_platforms",
    ConversationStage.START_DATE: "start_date",
}

# Reverse mapping: field name → stage (for multi-field validation lookups)
FIELD_TO_STAGE = {v: k for k, v in STAGE_TO_FIELD.items()}


# --- Helpers ---


def generate_conversation_id() -> str:
    """Generate a UUID4 conversation ID. Called server-side in POST /conversations.

    Returns:
        A unique conversation identifier string.
    """
    return str(uuid.uuid4())


def get_next_stage(
    current_stage: ConversationStage, extracted_data: dict
) -> ConversationStage | None:
    """Return the next stage in STAGE_ORDER, applying conditional skip logic.

    After EXPERIENCE_YEARS, if years == 0 the EXPERIENCE_PLATFORM stage is
    skipped and the flow jumps directly to START_DATE.

    Args:
        current_stage: The stage the conversation is currently on.
        extracted_data: Fields collected so far (used for skip logic).

    Returns:
        The next ``ConversationStage``, or ``None`` if the conversation is
        at CLOSING or beyond (i.e. complete).
    """
    current_index = STAGE_ORDER.index(current_stage)
    if current_index >= len(STAGE_ORDER) - 1:
        return None  # at CLOSING or beyond

    next_stage = STAGE_ORDER[current_index + 1]

    # Skip platform sub-stage if experience years == 0
    if next_stage == ConversationStage.EXPERIENCE_PLATFORM:
        experience_years = extracted_data.get("experience_years", 0)
        if experience_years == 0 or experience_years is None:
            return STAGE_ORDER[current_index + 2]  # skip to START_DATE

    return next_stage
