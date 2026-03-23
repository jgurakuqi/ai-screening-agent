"""Re-engagement logic for idle screening conversations.

Provides scheduled and manual re-engagement: sends a nudge message to
candidates who stopped responding, and marks conversations as abandoned
after the configured timeout window.
"""

from datetime import datetime, timezone

from backend.logging_config import logger
from backend import storage
from backend.config import CONVERSATION_ABANDON_HOURS, REENGAGEMENT_MAX_ATTEMPTS


def get_reengagement_message(language: str) -> str:
    """Return a bilingual nudge message for an idle candidate.

    Args:
        language: ``"es"`` or ``"en"``.

    Returns:
        Localised re-engagement prompt.
    """
    if language == "en":
        return (
            "Hi! We noticed you stopped midway through your application. "
            "Would you like to pick up where you left off?"
        )
    return (
        "¡Hola! Notamos que te quedaste a mitad. "
        "¿Quieres continuar donde lo dejaste?"
    )


async def check_and_reengage(timeout_minutes: int) -> None:
    """Scheduled job: send nudge messages to idle conversations or mark them abandoned.

    For each in-progress conversation that has been inactive longer than
    *timeout_minutes*, either sends a re-engagement message (up to
    ``REENGAGEMENT_MAX_ATTEMPTS``) or marks the conversation as abandoned
    (after ``CONVERSATION_ABANDON_HOURS``).

    Args:
        timeout_minutes: Inactivity threshold passed from the scheduler
            (matches ``REENGAGEMENT_TIMEOUT_MINUTES``).
    """
    logger.debug("Running scheduled re-engagement check (timeout={}min)", timeout_minutes)
    conversations = storage.get_incomplete_conversations(timeout_minutes=timeout_minutes)
    logger.debug("Found {} inactive conversations", len(conversations))

    for conv in conversations:
        cid = conv["id"][:8]
        now = datetime.now(timezone.utc)

        # Parse last_message_at
        last_msg_str = conv.get("last_message_at")
        if not last_msg_str:
            continue

        try:
            last_msg_at = datetime.fromisoformat(last_msg_str)
            if last_msg_at.tzinfo is None:
                last_msg_at = last_msg_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logger.warning("[{}] Could not parse last_message_at: {}", cid, last_msg_str)
            continue

        hours_since = (now - last_msg_at).total_seconds() / 3600

        if hours_since > CONVERSATION_ABANDON_HOURS:
            storage.update_conversation(conv["id"], status="abandoned")
            # Clean up per-conversation lock to prevent unbounded memory growth
            from backend.agent import cleanup_conversation_lock
            cleanup_conversation_lock(conv["id"])
            logger.info("[{}] Marked as abandoned (inactive {:.1f}h > {}h threshold)", cid, hours_since, CONVERSATION_ABANDON_HOURS)
            continue

        if conv.get("reengagement_count", 0) >= REENGAGEMENT_MAX_ATTEMPTS:
            logger.debug("[{}] Max re-engagement attempts reached ({}), skipping", cid, REENGAGEMENT_MAX_ATTEMPTS)
            continue

        language = conv.get("language", "es")
        message = get_reengagement_message(language)
        storage.save_message(conv["id"], "assistant", message, language)
        storage.increment_reengagement_count(conv["id"])
        storage.set_last_reengagement_at(conv["id"], now.isoformat())
        attempt_num = conv.get("reengagement_count", 0) + 1
        logger.info("[{}] Re-engagement sent (attempt {}/{}, inactive {:.1f}h)", cid, attempt_num, REENGAGEMENT_MAX_ATTEMPTS, hours_since)


async def reengage_conversation(conversation_id: str) -> str | None:
    """Manually trigger re-engagement for a specific conversation.

    Only works for in-progress conversations; otherwise returns ``None``.

    Args:
        conversation_id: UUID of the conversation.

    Returns:
        The re-engagement message text, or ``None`` if the conversation
        was not found or is not in progress.
    """
    cid = conversation_id[:8]
    conv = storage.get_conversation(conversation_id)
    if conv is None:
        logger.debug("[{}] Manual re-engagement: conversation not found", cid)
        return None
    if conv["status"] != "in_progress":
        logger.debug("[{}] Manual re-engagement: conversation not in_progress (status={})", cid, conv["status"])
        return None

    language = conv.get("language", "es")
    message = get_reengagement_message(language)
    storage.save_message(conversation_id, "assistant", message, language)
    storage.increment_reengagement_count(conversation_id)
    storage.set_last_reengagement_at(
        conversation_id, datetime.now(timezone.utc).isoformat()
    )
    logger.info("[{}] Manual re-engagement sent (language={})", cid, language)
    return message
