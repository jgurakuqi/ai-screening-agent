"""Privacy and data-retention utilities.

Provides input sanitization (control-character stripping, length enforcement),
privacy-safe event logging, and scheduled purging of expired conversations.
"""

import re
from datetime import datetime, timezone

from backend.logging_config import logger
from backend import storage
from backend.config import DATA_RETENTION_DAYS

MAX_INPUT_LENGTH = 450

# Control characters regex (excluding common whitespace)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize_input(text: str) -> str:
    """Strip control characters, enforce max length, and trim whitespace.

    Args:
        text: Raw user input string.

    Returns:
        Sanitised and length-capped text.
    """
    original_len = len(text)
    # Strip control characters
    text = _CONTROL_CHARS.sub("", text)
    if len(text) != original_len:
        logger.debug("Stripped {} control characters from input", original_len - len(text))

    # Enforce max length
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]
        logger.info("Input truncated from {} to {} characters", original_len, MAX_INPUT_LENGTH)

    return text.strip()


def log_event(conversation_id: str, stage: str, event: str) -> None:
    """Log event metadata only — never raw message content.

    Args:
        conversation_id: Full conversation UUID (truncated in log output).
        stage: Current conversation stage name.
        event: Event label (e.g. ``"user_message"``, ``"field_collected"``).
    """
    logger.info(
        "EVENT conv={} stage={} event={}",
        conversation_id[:8], stage, event,
    )


def run_data_retention() -> int:
    """Purge conversations older than ``DATA_RETENTION_DAYS``.

    Called on FastAPI startup via the lifespan event.

    Returns:
        Number of conversations purged.
    """
    logger.debug("Running data retention check (retention_days={})", DATA_RETENTION_DAYS)
    count = storage.purge_old_conversations(DATA_RETENTION_DAYS)
    if count > 0:
        logger.info("Purged {} old conversations (retention={} days)", count, DATA_RETENTION_DAYS)
    else:
        logger.debug("No conversations to purge")
    return count
