"""Two-pass extraction and summarization of completed screening conversations.

After a conversation reaches a terminal state, this module runs an extraction
pass (structured JSON) and a summary pass (recruiter-facing prose) via the LLM,
with deterministic fallbacks when the LLM output is malformed or unavailable.
"""

import json
import re
import time

from backend.logging_config import logger
from backend import storage
from backend.prompts import EXTRACTION_PROMPT, SUMMARY_PROMPT


def format_transcript(messages: list[dict]) -> str:
    """Convert stored message rows into a plain-text conversation transcript.

    Args:
        messages: Chronological message records containing ``role`` and
            ``content`` keys.

    Returns:
        Newline-delimited transcript suitable for prompt injection, with roles
        normalized to ``Assistant`` and ``Candidate``.
    """
    lines = []
    for msg in messages:
        role = "Assistant" if msg["role"] == "assistant" else "Candidate"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def validate_extraction_schema(data: dict) -> None:
    """Validate the basic shape of the extraction payload returned by the LLM.

    The function allows missing fields, but rejects unsupported top-level data
    types for the known extraction keys.

    Args:
        data: Parsed JSON object returned by the extraction pass.

    Raises:
        ValueError: If ``data`` is not a dict or contains known fields with an
            invalid value type.
    """
    expected_fields = {
        "full_name", "driver_license", "city_zone", "availability",
        "preferred_schedule", "experience_years", "experience_platforms",
        "start_date", "disqualification_reason",
    }
    if not isinstance(data, dict):
        raise ValueError("Extraction result is not a dict")
    # Allow missing fields (they'll be null) but reject unknown types
    for key in expected_fields:
        if key in data and data[key] is not None:
            # Basic type checks
            if key == "driver_license" and not isinstance(data[key], bool):
                raise ValueError(f"driver_license must be bool, got {type(data[key])}")
            if key == "experience_platforms" and not isinstance(data[key], list):
                raise ValueError(f"experience_platforms must be list, got {type(data[key])}")
            if key == "experience_years" and not isinstance(data[key], (int, float)):
                raise ValueError(f"experience_years must be int, got {type(data[key])}")


def _format_list(values: list[str]) -> str:
    """Format a short string list into recruiter-facing prose.

    Args:
        values: Raw platform or attribute labels that may include blanks.

    Returns:
        A cleaned human-readable list using English conjunction rules, or an
        empty string when no usable values remain.
    """
    cleaned = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _build_fallback_summary(extracted_data: dict, status: str) -> str:
    """Build a deterministic recruiter summary when the LLM summary is unusable.

    Args:
        extracted_data: Normalized candidate fields collected during screening.
        status: Final conversation status used to tailor the summary outcome.

    Returns:
        Recruiter-facing summary text assembled without another model call.
    """
    name = (extracted_data.get("full_name") or "").strip() or "The candidate"
    availability = extracted_data.get("availability")
    schedule = extracted_data.get("preferred_schedule")
    experience_years = extracted_data.get("experience_years")
    platforms = extracted_data.get("experience_platforms") or []
    start_date = extracted_data.get("start_date")
    city_zone = extracted_data.get("city_zone")
    disqualification_reason = extracted_data.get("disqualification_reason")

    sentences: list[str] = []

    if status == "disqualified":
        if disqualification_reason == "no_license":
            sentences.append(
                f"{name} was disqualified because a valid driver's license was not confirmed, "
                "which is a mandatory requirement for the role."
            )
        elif disqualification_reason == "outside_area":
            if city_zone:
                sentences.append(
                    f"{name} was disqualified because {city_zone} is outside the current service area."
                )
            else:
                sentences.append(
                    f"{name} was disqualified because the candidate is outside the current service area."
                )
        else:
            sentences.append(f"{name} was disqualified during the screening process.")

        details: list[str] = []
        if availability:
            details.append(f"availability for {availability} work")
        if schedule:
            details.append(f"a {schedule} schedule preference")
        if experience_years is not None:
            if experience_years == 0:
                details.append("no prior delivery experience")
            elif experience_years < 1:
                months = round(experience_years * 12)
                details.append(f"{months} month{'s' if months != 1 else ''} of delivery experience")
            else:
                details.append(f"{experience_years} years of delivery experience")

        if details:
            sentences.append(f"Before the conversation ended, the recruiter captured {', '.join(details)}.")

        sentences.append("The record should be available for recruiter review if additional context is needed.")
        return " ".join(sentences)

    if city_zone and extracted_data.get("driver_license") is True:
        sentences.append(
            f"{name} confirmed a valid driver's license and lives in {city_zone}, which is within the service area."
        )
    elif extracted_data.get("driver_license") is True:
        sentences.append(f"{name} confirmed a valid driver's license.")
    elif city_zone:
        sentences.append(f"{name} lives in {city_zone}.")

    work_details: list[str] = []
    if availability:
        work_details.append(f"available for {availability} work")
    if schedule:
        work_details.append(f"with a {schedule} preference")
    if work_details:
        sentences.append(f"{name} is {' '.join(work_details)}.")

    experience_sentence: list[str] = []
    if experience_years is not None:
        if experience_years == 0:
            experience_sentence.append("reported no prior delivery experience")
        elif experience_years < 1:
            months = round(experience_years * 12)
            experience_sentence.append(f"reported {months} month{'s' if months != 1 else ''} of delivery experience")
        elif experience_years == 1:
            experience_sentence.append("reported 1 year of delivery experience")
        else:
            experience_sentence.append(f"reported {experience_years} years of delivery experience")
    platform_list = _format_list(platforms)
    if platform_list:
        if experience_sentence:
            experience_sentence.append(f"including work with {platform_list}")
        else:
            experience_sentence.append(f"has worked with {platform_list}")
    if start_date:
        start_phrase = "as soon as possible" if start_date == "ASAP" else f"on {start_date}"
        experience_sentence.append(f"could start {start_phrase}")
    if experience_sentence:
        sentences.append(f"{name} {' and '.join(experience_sentence)}.")

    if status == "needs_review":
        sentences.append(
            "Some required fields remain missing or ambiguous, so the conversation should be reviewed by a recruiter."
        )
    else:
        sentences.append("No disqualifying issues were identified, and the candidate is qualified for the repartidor position.")

    return " ".join(sentences)


def _summary_needs_fallback(summary_text: str, extracted_data: dict) -> bool:
    """Decide whether an LLM-generated summary should be discarded.

    Args:
        summary_text: Recruiter summary returned by the LLM.
        extracted_data: Structured candidate data used for sanity checks.

    Returns:
        ``True`` when the summary is empty or matches known malformed-output
        patterns that should trigger the deterministic fallback.
    """
    text = (summary_text or "").strip()
    if not text:
        return True

    name = (extracted_data.get("full_name") or "").strip()
    if name and text.startswith(name) and len(text) > len(name):
        next_char = text[len(name)]
        if next_char.isalpha():
            return True

    if re.search(r"(?i)\b(fulltime|parttime|deliverydriver|deliveryexperience|servicearea)\b", text):
        return True

    if re.search(r"[.!?][A-Za-z]", text):
        return True

    return False


async def extract_and_summarize(conversation_id: str) -> dict:
    """Run the extraction pass and the recruiter-summary pass for a conversation.

    The extraction pass asks the LLM for structured JSON. If that fails, the
    function falls back to the incrementally stored database state. The summary
    pass then tries to generate recruiter-facing prose and falls back to a
    deterministic summary when the model output is malformed.

    Args:
        conversation_id: UUID of the completed or terminal conversation.

    Returns:
        Dict containing ``extracted_data`` and ``summary`` keys ready to be
        persisted back onto the conversation record.
    """
    # Import here to avoid circular imports
    from backend.agent import call_llm

    cid = conversation_id[:8]
    messages = storage.get_messages(conversation_id)
    transcript = format_transcript(messages)
    logger.debug("[{}] Starting extract_and_summarize ({} messages, transcript_len={})", cid, len(messages), len(transcript))

    # Pass 1: extraction
    extraction_start = time.monotonic()
    try:
        raw = await call_llm(
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(
                conversation_transcript=transcript
            )}],
            temperature=0,
            is_extraction=True,
        )
        extracted = json.loads(raw.strip())
        validate_extraction_schema(extracted)
        elapsed = time.monotonic() - extraction_start
        logger.info("[{}] Extraction pass completed in {:.0f}ms — fields: {}", cid, elapsed * 1000, list(extracted.keys()))
    except Exception as e:
        elapsed = time.monotonic() - extraction_start
        logger.warning("[{}] Extraction LLM failed after {:.0f}ms: {} — using DB fallback", cid, elapsed * 1000, e)
        extracted = storage.build_extracted_data_from_db(conversation_id)

    # Pass 2: summary
    summary_start = time.monotonic()
    try:
        summary_text = await call_llm(
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(
                conversation_transcript=transcript,
                extracted_data=json.dumps(extracted, ensure_ascii=False),
            )}],
            temperature=0.3,
            is_extraction=True,
        )
        conv = storage.get_conversation(conversation_id)
        status = conv["status"] if conv else "unknown"
        if _summary_needs_fallback(summary_text, extracted):
            logger.warning("[{}] Summary output looked malformed; using deterministic fallback", cid)
            summary_text = _build_fallback_summary(extracted, status)
        elapsed = time.monotonic() - summary_start
        logger.info("[{}] Summary pass completed in {:.0f}ms (len={})", cid, elapsed * 1000, len(summary_text))
    except Exception as e:
        elapsed = time.monotonic() - summary_start
        logger.warning("[{}] Summary LLM failed after {:.0f}ms: {}", cid, elapsed * 1000, e)
        conv = storage.get_conversation(conversation_id)
        status = conv["status"] if conv else "unknown"
        summary_text = f"Summary unavailable. Status: {status}."

    return {"extracted_data": extracted, "summary": summary_text}
