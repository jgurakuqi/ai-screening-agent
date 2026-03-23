"""SQLite persistence layer for conversations, messages, and field attempts.

Every public function opens and closes its own connection (no shared state),
making the module safe for concurrent use from async handlers.
"""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any

from backend.logging_config import logger
from backend.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with foreign keys enabled and Row factory.

    Returns:
        A ``sqlite3.Connection`` with ``row_factory = sqlite3.Row``.
    """
    logger.trace("Opening SQLite connection to {}", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the SQLite tables required by the screening workflow.

    The schema is created idempotently so the function is safe to call on every
    application startup.
    """
    logger.debug("Initializing database schema...")
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'in_progress',
                stage TEXT DEFAULT 'greeting',
                language TEXT DEFAULT 'es',
                created_at TEXT,
                last_message_at TEXT,
                reengagement_count INTEGER DEFAULT 0,
                last_reengagement_at TEXT,
                extracted_data TEXT DEFAULT '{}',
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                language TEXT,
                created_at TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS field_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                field_name TEXT,
                attempts INTEGER DEFAULT 0,
                UNIQUE(conversation_id, field_name),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
        """)
        conn.commit()
    finally:
        conn.close()


def create_conversation(conversation_id: str) -> None:
    """Insert a new conversation row with default values.

    Args:
        conversation_id: UUID for the new conversation.
    """
    logger.debug("Creating conversation row: {}", conversation_id[:8])
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO conversations (id, created_at, extracted_data) VALUES (?, ?, ?)",
            (conversation_id, now, "{}"),
        )
        conn.commit()
    finally:
        conn.close()


def get_conversation(conversation_id: str) -> dict | None:
    """Fetch a conversation record by ID.

    Args:
        conversation_id: UUID of the conversation.

    Returns:
        A dict with all conversation columns (``extracted_data`` already
        parsed from JSON), or ``None`` if not found.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["extracted_data"] = json.loads(d["extracted_data"]) if d["extracted_data"] else {}
        return d
    finally:
        conn.close()


def update_conversation(conversation_id: str, **kwargs: Any) -> None:
    """Update one or more columns on a conversation row.

    Dict-typed values for ``extracted_data`` are automatically serialised
    to JSON before writing.

    Args:
        conversation_id: UUID of the conversation to update.
        **kwargs: Column-name / value pairs (e.g. ``status="qualified"``).
    """
    logger.trace("Updating conversation {}: {}", conversation_id[:8], list(kwargs.keys()))
    conn = get_connection()
    try:
        sets = []
        values = []
        for key, value in kwargs.items():
            sets.append(f"{key} = ?")
            if key == "extracted_data" and isinstance(value, dict):
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                values.append(value)
        values.append(conversation_id)
        conn.execute(
            f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def update_extracted_field(conversation_id: str, field_name: str, value: object) -> None:
    """Set a single key inside the conversation's ``extracted_data`` JSON.

    Loads the current JSON, updates the specified field, and persists it
    back in a single transaction.

    Args:
        conversation_id: UUID of the conversation.
        field_name: Key to set inside ``extracted_data``.
        value: The value to store (must be JSON-serialisable).
    """
    logger.trace("Setting extracted field {}.{}={}", conversation_id[:8], field_name, repr(value))
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT extracted_data FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            logger.warning("Cannot update field — conversation {} not found", conversation_id[:8])
            return
        data = json.loads(row["extracted_data"]) if row["extracted_data"] else {}
        data[field_name] = value
        conn.execute(
            "UPDATE conversations SET extracted_data = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_message(conversation_id: str, role: str, content: str, language: str = "es") -> None:
    """Persist a message and update the conversation's ``last_message_at``.

    Args:
        conversation_id: UUID of the conversation.
        role: ``"user"`` or ``"assistant"``.
        content: Message text.
        language: Language code (``"es"`` or ``"en"``).
    """
    logger.trace("Saving {} message for conv={} (len={})", role, conversation_id[:8], len(content))
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, language, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, language, now),
        )
        conn.execute(
            "UPDATE conversations SET last_message_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_messages(conversation_id: str) -> list[dict]:
    """Return the full message history for a conversation.

    Args:
        conversation_id: UUID of the conversation.

    Returns:
        List of dicts with keys ``role``, ``content``, ``language``,
        and ``created_at``, ordered chronologically.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, language, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_incomplete_conversations(timeout_minutes: int) -> list[dict]:
    """Fetch in-progress conversations idle for longer than the given timeout.

    Args:
        timeout_minutes: Inactivity threshold in minutes.

    Returns:
        List of conversation dicts (``extracted_data`` already parsed)
        that have been idle longer than the threshold.
    """
    conn = get_connection()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)).isoformat()
        rows = conn.execute(
            """SELECT * FROM conversations
               WHERE status = 'in_progress'
               AND last_message_at IS NOT NULL
               AND last_message_at < ?""",
            (cutoff,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["extracted_data"] = json.loads(d["extracted_data"]) if d["extracted_data"] else {}
            result.append(d)
        return result
    finally:
        conn.close()


def increment_reengagement_count(conversation_id: str) -> None:
    """Atomically increment the re-engagement attempt counter.

    Args:
        conversation_id: UUID of the conversation.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE conversations SET reengagement_count = reengagement_count + 1 WHERE id = ?",
            (conversation_id,),
        )
        conn.commit()
    finally:
        conn.close()


def set_last_reengagement_at(conversation_id: str, iso_timestamp: str) -> None:
    """Record the timestamp of the most recent re-engagement attempt.

    Args:
        conversation_id: UUID of the conversation.
        iso_timestamp: ISO 8601 timestamp string.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE conversations SET last_reengagement_at = ? WHERE id = ?",
            (iso_timestamp, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_field_attempts(conversation_id: str, field_name: str) -> int:
    """Return the number of clarification attempts for a specific field.

    Args:
        conversation_id: UUID of the conversation.
        field_name: Screening field name (e.g. ``"full_name"``).

    Returns:
        Number of attempts so far, or ``0`` if no record exists.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT attempts FROM field_attempts WHERE conversation_id = ? AND field_name = ?",
            (conversation_id, field_name),
        ).fetchone()
        return row["attempts"] if row else 0
    finally:
        conn.close()


def increment_field_attempts(conversation_id: str, field_name: str) -> None:
    """Increment (or initialise) the clarification attempt counter for a field.

    Uses ``INSERT … ON CONFLICT … DO UPDATE`` so the first call creates
    the row with ``attempts = 1``.

    Args:
        conversation_id: UUID of the conversation.
        field_name: Screening field name (e.g. ``"city_zone"``).
    """
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO field_attempts (conversation_id, field_name, attempts)
               VALUES (?, ?, 1)
               ON CONFLICT(conversation_id, field_name)
               DO UPDATE SET attempts = attempts + 1""",
            (conversation_id, field_name),
        )
        conn.commit()
    finally:
        conn.close()


def build_extracted_data_from_db(conversation_id: str) -> dict:
    """Read the current ``extracted_data`` JSON from the conversations table.

    Args:
        conversation_id: UUID of the conversation.

    Returns:
        Parsed dict of extracted fields, or an empty dict if not found.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT extracted_data FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return {}
        return json.loads(row["extracted_data"]) if row["extracted_data"] else {}
    finally:
        conn.close()


def purge_old_conversations(retention_days: int) -> int:
    """Delete conversations, messages, and field attempts older than the retention window.

    Args:
        retention_days: Age threshold in days — conversations created before
            ``now - retention_days`` are purged.

    Returns:
        Number of conversations deleted.
    """
    logger.debug("Running data retention purge (retention_days={})", retention_days)
    conn = get_connection()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        # Get IDs to purge
        rows = conn.execute(
            "SELECT id FROM conversations WHERE created_at < ?", (cutoff,)
        ).fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            logger.debug("No conversations to purge")
            return 0
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM field_attempts WHERE conversation_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", ids)
        conn.commit()
        logger.info("Purged {} conversations older than {} days", len(ids), retention_days)
        return len(ids)
    finally:
        conn.close()


def delete_all() -> int:
    """Delete all conversations, messages, and field attempts from the database.

    Returns:
        Number of conversations that were deleted.
    """
    logger.warning("Deleting ALL data from database")
    conn = get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM field_attempts")
        conn.execute("DELETE FROM conversations")
        conn.commit()
        logger.info("Deleted {} conversations and all related records", count)
        return count
    finally:
        conn.close()


def list_conversations() -> list[dict]:
    """Return all conversations with summary fields, sorted by most recent activity.

    Returns:
        List of conversation dicts (``extracted_data`` already parsed),
        ordered by ``last_message_at`` descending.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT id, status, stage, created_at, last_message_at, extracted_data
               FROM conversations
               ORDER BY COALESCE(last_message_at, created_at) DESC"""
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["extracted_data"] = json.loads(d["extracted_data"]) if d["extracted_data"] else {}
            result.append(d)
        return result
    finally:
        conn.close()


def get_analytics() -> dict:
    """Compute aggregate analytics across all conversations.

    Returns:
        Dict with keys ``total_conversations``, ``total_today``,
        ``by_status``, ``completion_rate``, ``qualification_rate``,
        ``disqualification_reasons``, ``dropoff_by_stage``,
        ``avg_turns_qualified``, and ``avg_turns_disqualified``.
    """
    conn = get_connection()
    try:
        # Total conversations
        total = conn.execute("SELECT COUNT(*) as c FROM conversations").fetchone()["c"]

        # Total today
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        total_today = conn.execute(
            "SELECT COUNT(*) as c FROM conversations WHERE created_at >= ?", (today_start,)
        ).fetchone()["c"]

        # By status
        status_rows = conn.execute(
            "SELECT status, COUNT(*) as c FROM conversations GROUP BY status"
        ).fetchall()
        by_status = {r["status"]: r["c"] for r in status_rows}

        qualified = by_status.get("qualified", 0)
        disqualified = by_status.get("disqualified", 0)
        needs_review = by_status.get("needs_review", 0)
        withdrawn = by_status.get("withdrawn", 0)
        completed = qualified + disqualified + needs_review + withdrawn

        completion_rate = completed / total if total > 0 else 0.0
        qualification_rate = qualified / (qualified + disqualified + needs_review) if (qualified + disqualified + needs_review) > 0 else 0.0

        # Disqualification reasons
        disq_rows = conn.execute(
            "SELECT extracted_data FROM conversations WHERE status = 'disqualified'"
        ).fetchall()
        disq_reasons = {"no_license": 0, "outside_area": 0}
        for row in disq_rows:
            data = json.loads(row["extracted_data"]) if row["extracted_data"] else {}
            reason = data.get("disqualification_reason")
            if reason in disq_reasons:
                disq_reasons[reason] += 1

        # Drop-off by stage (abandoned/withdrawn conversations only — in_progress
        # conversations are still active and should not count as drop-offs)
        dropoff_rows = conn.execute(
            "SELECT stage, COUNT(*) as c FROM conversations WHERE status IN ('abandoned', 'withdrawn') GROUP BY stage"
        ).fetchall()
        dropoff_by_stage = {r["stage"]: r["c"] for r in dropoff_rows}

        # Average turns
        def avg_turns_for_status(status: str) -> float:
            rows = conn.execute(
                """SELECT c.id, COUNT(m.id) as turns
                   FROM conversations c
                   LEFT JOIN messages m ON m.conversation_id = c.id
                   WHERE c.status = ?
                   GROUP BY c.id""",
                (status,),
            ).fetchall()
            if not rows:
                return 0.0
            return sum(r["turns"] for r in rows) / len(rows)

        # Average sentiment across all conversations with sentiment data
        sentiment_rows = conn.execute(
            "SELECT extracted_data FROM conversations WHERE extracted_data IS NOT NULL"
        ).fetchall()
        all_sentiments: list[float] = []
        sentiment_map = {"positive": 1.0, "neutral": 0.5, "frustrated": 0.2, "confused": 0.3}
        for row in sentiment_rows:
            data = json.loads(row["extracted_data"]) if row["extracted_data"] else {}
            for s in data.get("sentiment_history", []):
                if s in sentiment_map:
                    all_sentiments.append(sentiment_map[s])
        avg_sentiment = round(sum(all_sentiments) / len(all_sentiments), 2) if all_sentiments else None

        return {
            "total_conversations": total,
            "total_today": total_today,
            "by_status": by_status,
            "completion_rate": round(completion_rate, 4),
            "qualification_rate": round(qualification_rate, 4),
            "disqualification_reasons": disq_reasons,
            "dropoff_by_stage": dropoff_by_stage,
            "avg_turns_qualified": round(avg_turns_for_status("qualified"), 1),
            "avg_turns_disqualified": round(avg_turns_for_status("disqualified"), 1),
            "avg_turns_needs_review": round(avg_turns_for_status("needs_review"), 1),
            "avg_sentiment": avg_sentiment,
        }
    finally:
        conn.close()
