"""Core conversation agent — orchestrates the screening flow.

Handles language detection, guardrail checks (exit intent, offensive content),
LLM calls with function calling, field validation, stage advancement, and
conversation finalization.
"""

import asyncio
import json
import time
from collections import defaultdict

from lingua import Language, LanguageDetectorBuilder

from backend.logging_config import logger
from backend.llm.service import LLMService, LLM_BUSY_RESPONSE
from backend.config import (
    MAX_CLARIFICATION_ATTEMPTS,
    HARD_DISQUALIFIERS,
    MAX_OFFENSIVE_STRIKES,
    ConversationStage,
    ConversationStatus,
    STAGE_TO_FIELD,
    FIELD_TO_STAGE,
    STAGE_ORDER,
    get_next_stage,
)
from backend.models import LLMScreeningResponse
from backend.prompts import SYSTEM_PROMPT
from backend.privacy import sanitize_input, log_event
from backend.validator import VALIDATORS
from backend.guardrails import (
    detect_confirmation,
    detect_exit_intent,
    detect_offensive_content,
    FRONTEND_STOP_SIGNAL,
    get_exit_confirmation_message,
    get_withdrawal_message,
    get_offensive_warning_message,
    get_offensive_termination_message,
)
from backend import storage, summary, faq

# Lingua detector restricted to ES/EN — high accuracy on short text
_lingua_detector = LanguageDetectorBuilder.from_languages(
    Language.SPANISH, Language.ENGLISH
).build()

# Per-conversation locks to serialize rapid messages
_conversation_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# LLM service — injected from main.py lifespan via set_llm_service()
_llm_service: LLMService | None = None


def set_llm_service(service: LLMService) -> None:
    """Inject the LLM service instance (called from ``main.py`` lifespan)."""
    global _llm_service
    _llm_service = service


def cleanup_conversation_lock(conversation_id: str) -> None:
    """Remove the per-conversation lock entry to prevent unbounded growth.

    Called by the re-engagement scheduler when marking a conversation as
    abandoned, since that code path doesn't go through ``process_message``.
    """
    _conversation_locks.pop(conversation_id, None)


# Load service areas for prompt
_SERVICE_AREAS_CACHE: str | None = None


def _get_service_areas_str() -> str:
    """Return a comma-separated string of service-area cities (cached)."""
    global _SERVICE_AREAS_CACHE
    if _SERVICE_AREAS_CACHE is None:
        from backend.validator import CITIES

        _SERVICE_AREAS_CACHE = ", ".join(CITIES)
    return _SERVICE_AREAS_CACHE


# ── Per-stage field schemas for function calling ──

STAGE_FIELD_SCHEMAS: dict[str, dict] = {
    "name": {
        "type": "string",
        "description": (
            "The candidate's full name. Must be a real, plausible human name. "
            "If the candidate provides gibberish, repetitive text, or something "
            "that is clearly not a name, do NOT include this field — omit it "
            "so the system can re-ask."
        ),
    },
    "license": {"type": "boolean", "description": "Whether the candidate has a driver's license"},
    "city": {
        "type": "string",
        "description": (
            "The city the candidate currently lives in. Extract ONLY the city "
            "name (e.g., 'Madrid', 'Barcelona'). If the candidate does not "
            "provide an actual city (e.g., they say 'everywhere', 'anywhere', "
            "'I can relocate', or give a vague/non-city answer), do NOT "
            "include this field — omit it so the system can re-ask."
        ),
    },
    "availability": {
        "type": "string",
        "enum": ["full-time", "part-time", "weekends"],
        "description": "The candidate's work availability",
    },
    "schedule": {
        "type": "string",
        "enum": ["morning", "afternoon", "evening", "flexible"],
        "description": "The candidate's preferred work schedule",
    },
    "experience_years": {"type": "number", "description": "Years of delivery experience as a decimal (e.g. 2 for two years, 0.5 for six months, 0.25 for three months). Use 0 only when the candidate explicitly says they have no experience."},
    "experience_platform": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Specific delivery platform names the candidate has worked with "
            "(e.g., 'Glovo', 'Uber Eats', 'Deliveroo'). If the candidate gives "
            "a vague answer like 'many', 'several', or 'a lot' without naming "
            "specific platforms, do NOT include this field — omit it so the "
            "system can re-ask for specific names."
        ),
    },
    "start_date": {"type": "string", "description": "When the candidate can start (YYYY-MM-DD or ASAP)"},
}


def _build_tool_schema(stage: str, extracted_data: dict | None = None) -> list[dict]:
    """Build the function calling tool definition with multi-field extraction.

    Includes optional properties for all uncollected fields from the current
    stage onward, allowing the LLM to extract multiple fields in a single turn.
    """
    if extracted_data is None:
        extracted_data = {}

    # Build extracted_fields properties for all uncollected stages from current onward
    extracted_fields_props: dict[str, dict] = {}
    try:
        current_idx = STAGE_ORDER.index(ConversationStage(stage))
    except (ValueError, KeyError):
        current_idx = 0

    for s in STAGE_ORDER[current_idx:]:
        if s in (ConversationStage.GREETING, ConversationStage.CLOSING):
            continue
        field_name = STAGE_TO_FIELD.get(s)
        if field_name is None:
            continue
        # Skip already-collected fields
        if field_name in extracted_data and extracted_data[field_name] is not None:
            continue
        # Skip experience_platform if experience_years is known to be 0
        if s == ConversationStage.EXPERIENCE_PLATFORM:
            exp_years = extracted_data.get("experience_years")
            if exp_years is not None and exp_years == 0:
                continue
        schema = STAGE_FIELD_SCHEMAS.get(s.value)
        if schema:
            extracted_fields_props[field_name] = schema

    return [
        {
            "type": "function",
            "function": {
                "name": "extract_screening_field",
                "description": (
                    "Extract screening field values from the candidate's message and generate a response. "
                    "Include ALL fields the candidate clearly mentioned, even if they answer future questions early."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "extracted_fields": {
                            "type": "object",
                            "description": (
                                "All screening fields found in the candidate's message. "
                                "Only include fields the candidate actually mentioned."
                            ),
                            "properties": extracted_fields_props,
                        },
                        "response": {
                            "type": "string",
                            "description": "Your message to the candidate, written in their language",
                        },
                        "detected_language": {
                            "type": "string",
                            "enum": ["es", "en"],
                            "description": "The language of the candidate's last message",
                        },
                        "response_language": {
                            "type": "string",
                            "enum": ["es", "en"],
                            "description": "The language you actually wrote your response in",
                        },
                        "exit_intent": {
                            "type": "boolean",
                            "description": "True if the candidate wants to stop or leave the screening",
                        },
                        "is_offensive": {
                            "type": "boolean",
                            "description": "True if the candidate's message contains insults, profanity, or abusive language",
                        },
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "neutral", "frustrated", "confused"],
                            "description": "The candidate's emotional tone in this message",
                        },
                    },
                    "required": ["extracted_fields", "response", "detected_language", "response_language", "exit_intent", "is_offensive", "sentiment"],
                },
            },
        }
    ]


# ── LLM call helpers ──


async def call_llm(
    messages: list[dict],
    temperature: float = 0.3,
    is_extraction: bool = False,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> str | dict:
    """Call the LLM via the configured provider chain.

    Returns:
        Raw LLM response string (content mode) or parsed dict (function calling mode),
        or ``LLM_BUSY_RESPONSE`` (``""``) if all providers are exhausted.
    """
    call_type = "extraction" if is_extraction else "conversation"
    logger.debug("LLM call started | type={} temp={} tools={}", call_type, temperature, bool(tools))
    call_start = time.monotonic()

    result = await _llm_service.generate(
        messages, temperature,
        tools=tools, tool_choice=tool_choice,
    )

    elapsed = time.monotonic() - call_start
    if result:
        logger.info(
            "LLM call completed | type={} ms={:.0f} result_type={}",
            call_type, elapsed * 1000, type(result).__name__,
        )
    else:
        logger.warning(
            "LLM call returned empty | type={} ms={:.0f}",
            call_type, elapsed * 1000,
        )
    return result


def parse_llm_response(raw_response: str | dict, stage: str = "") -> LLMScreeningResponse:
    """Parse LLM output into an LLMScreeningResponse.

    Handles both function calling (dict) and content mode (JSON string).
    If the response uses the legacy single ``field_value`` format, it is
    converted to ``extracted_fields`` keyed by the current stage's field name.
    """
    if isinstance(raw_response, dict):
        # Function calling mode — already parsed
        logger.debug(
            "Parsed tool call response: extracted_fields={} field_value={} sentiment={}",
            repr(raw_response.get("extracted_fields")),
            repr(raw_response.get("field_value")),
            raw_response.get("sentiment"),
        )
        result = LLMScreeningResponse(**raw_response)
    else:
        # Content mode fallback (used by extraction pass, or if tools not available)
        cleaned = raw_response.strip()
        logger.trace("Parsing LLM JSON (first 200 chars): {}", cleaned[:200])
        try:
            data = json.loads(cleaned)
            result = LLMScreeningResponse(**data)
            logger.debug(
                "Parsed LLM response: field_value={} response_len={}",
                repr(result.field_value), len(result.response),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Failed to parse LLM response as JSON — falling back to empty response. Error: {} Raw: {}",
                e, raw_response[:200],
            )
            return LLMScreeningResponse(field_value=None, response=cleaned)

    # Backward compat: if only field_value was provided (no extracted_fields),
    # convert to extracted_fields keyed by the current stage's field name.
    if not result.extracted_fields and result.field_value is not None:
        stage_enum = ConversationStage(stage) if stage else None
        field_name = STAGE_TO_FIELD.get(stage_enum) if stage_enum else None
        if field_name:
            result.extracted_fields = {field_name: result.field_value}

    return result


# ── Language detection ──


def _detect_response_language(text: str) -> str:
    """Detect the actual language of an LLM response using Lingua.

    Restricted to Spanish/English for maximum accuracy on short text.
    Used to determine the appended question's language so a single message
    never mixes languages — regardless of what the LLM self-reports.
    """
    lang = _lingua_detector.detect_language_of(text)
    if lang == Language.SPANISH:
        return "es"
    # English or None (undetectable) — default to English
    return "en"


# ── Greeting ──


def generate_greeting(language: str = "es") -> str:
    """Return the deterministic opening greeting message."""
    if language == "en":
        return (
            "Hi! I'm the screening assistant for Grupo Sazón. "
            "We're looking for delivery drivers and I'd like to ask you "
            "a few quick questions. What's your name? First and last name, please."
        )
    return (
        "¡Hola! Soy el asistente de selección de Grupo Sazón. "
        "Estamos buscando repartidores y me gustaría hacerte "
        "unas preguntas rápidas. ¿Cómo te llamas? Nombre y apellidos, por favor."
    )


# ── Fallback questions and helpers ──


def _get_stage_question(stage: str, language: str) -> str:
    """Return a hardcoded fallback question for a stage when the LLM response is empty."""
    questions_es = {
        "name": "¿Cómo te llamas? Nombre y apellidos, por favor.",
        "license": "¿Tienes carnet de conducir?",
        "city": "¿En qué ciudad vives?",
        "availability": "¿Qué disponibilidad tienes? Tiempo completo, medio tiempo, o fines de semana.",
        "schedule": "¿Qué horario prefieres? Mañana, tarde, noche, o flexible.",
        "experience_years": "¿Cuánta experiencia en reparto tienes?",
        "experience_platform": "¿En qué plataformas de reparto has trabajado?",
        "start_date": "¿Cuándo podrías empezar?",
    }
    questions_en = {
        "name": "What's your name? First and last name, please.",
        "license": "Do you have a driver's license?",
        "city": "What city do you live in?",
        "availability": "What's your availability? Full-time, part-time, or weekends.",
        "schedule": "What schedule do you prefer? Morning, afternoon, evening, or flexible.",
        "experience_years": "How much delivery experience do you have?",
        "experience_platform": "Which delivery platforms have you worked with?",
        "start_date": "When could you start?",
    }
    questions = questions_en if language == "en" else questions_es
    return questions.get(stage, "")


def _get_stage_clarification(stage: str, language: str) -> str:
    """Return a clarification prompt when the candidate's answer can't be validated."""
    clarifications_es = {
        "name": "¿Me podrías decir tu nombre completo?",
        "license": "¿Tienes carnet de conducir? Sí o no.",
        "city": "¿En qué ciudad vives exactamente?",
        "availability": "¿Podrías elegir entre tiempo completo, medio tiempo, o fines de semana?",
        "schedule": "¿Qué horario prefieres? Mañana, tarde, noche, o flexible.",
        "experience_years": "¿Podrías darme un número aproximado de años? Por ejemplo, 1, 2, 5...",
        "experience_platform": "¿En qué plataformas de reparto has trabajado? Por ejemplo, Uber Eats, Glovo...",
        "start_date": "¿Podrías darme una fecha concreta o decirme si podrías empezar de inmediato?",
    }
    clarifications_en = {
        "name": "Could you tell me your full name?",
        "license": "Do you have a driver's license? Yes or no.",
        "city": "Which city do you live in exactly?",
        "availability": "Could you pick between full-time, part-time, or weekends?",
        "schedule": "What schedule do you prefer? Morning, afternoon, evening, or flexible.",
        "experience_years": "Could you give me a rough number of years? For example, 1, 2, 5...",
        "experience_platform": "Which delivery platforms have you worked with? For example, Uber Eats, Glovo...",
        "start_date": "Could you give me a specific date or let me know if you could start right away?",
    }
    clarifications = clarifications_en if language == "en" else clarifications_es
    return clarifications.get(stage, "")


def _ensure_response(
    llm_result: LLMScreeningResponse, stage: str, language: str
) -> LLMScreeningResponse:
    """Guarantee the LLM result contains a non-empty response string."""
    if llm_result.response and llm_result.response.strip():
        return llm_result
    fallback = _get_stage_question(stage, language)
    if fallback:
        logger.warning(
            "LLM returned empty response for stage={}, using hardcoded fallback question",
            stage,
        )
        return LLMScreeningResponse(
            field_value=llm_result.field_value, response=fallback,
            sentiment=llm_result.sentiment,
        )
    logger.error(
        "LLM returned empty response for stage={} and no fallback available", stage
    )
    return llm_result


def _build_system_prompt(
    stage: str, extracted_data: dict, faq_context: str = ""
) -> str:
    """Build the system prompt with current conversation context."""
    name = extracted_data.get("full_name", "candidato/a")
    return SYSTEM_PROMPT.format(
        service_areas=_get_service_areas_str(),
        current_stage=stage,
        collected_fields=json.dumps(extracted_data, ensure_ascii=False),
        name=name,
        faq_context=faq_context,
    )


# ── Guardrail helpers (extracted from _process_message_inner) ──


def _handle_offensive(
    conversation_id: str,
    extracted_data: dict,
    current_stage: str,
    language: str,
    process_start: float,
    source: str,
) -> dict | None:
    """Handle offensive content detection (shared by LLM-detected path).

    Returns a response dict if the message was handled, or None to continue.
    """
    cid = conversation_id[:8]
    log_event(conversation_id, current_stage, f"offensive_input_{source}")

    strikes = extracted_data.get("_offensive_strikes", 0) + 1
    extracted_data["_offensive_strikes"] = strikes
    storage.update_conversation(conversation_id, extracted_data=extracted_data)
    logger.warning(
        "[{}] Offensive content detected via {} (strike {}/{})",
        cid, source, strikes, MAX_OFFENSIVE_STRIKES,
    )

    if strikes >= MAX_OFFENSIVE_STRIKES:
        termination_msg = get_offensive_termination_message(language)
        storage.save_message(conversation_id, "assistant", termination_msg, language)
        extracted_data["terminated_offensive"] = True
        extracted_data.pop("_offensive_strikes", None)
        extracted_data.pop("_pending_withdrawal", None)
        extracted_data["disqualification_reason"] = "offensive_behavior"
        storage.update_conversation(
            conversation_id,
            status=ConversationStatus.DISQUALIFIED,
            stage=current_stage,
            extracted_data=extracted_data,
        )
        log_event(conversation_id, current_stage, "terminated_offensive")
        elapsed = time.monotonic() - process_start
        logger.info("[{}] Offensive termination in {:.0f}ms", cid, elapsed * 1000)
        return {
            "response": termination_msg,
            "status": ConversationStatus.DISQUALIFIED,
            "stage": current_stage,
            "extracted_data": extracted_data,
            "language": language,
            "_finalize": True,
        }

    warning_msg = get_offensive_warning_message(language)
    storage.save_message(conversation_id, "assistant", warning_msg, language)
    elapsed = time.monotonic() - process_start
    logger.info("[{}] Offensive warning issued in {:.0f}ms", cid, elapsed * 1000)
    return {
        "response": warning_msg,
        "status": ConversationStatus.IN_PROGRESS,
        "stage": current_stage,
        "extracted_data": extracted_data,
        "language": language,
    }


async def _handle_withdrawal_flow(
    conversation_id: str,
    user_message: str,
    extracted_data: dict,
    current_stage: str,
    language: str,
    process_start: float,
) -> dict | None:
    """Handle pending withdrawal confirmation. Returns response dict or None."""
    if not extracted_data.get("_pending_withdrawal"):
        return None

    cid = conversation_id[:8]
    is_stop_signal = user_message.strip() == FRONTEND_STOP_SIGNAL

    if detect_confirmation(user_message) or is_stop_signal:
        logger.info("[{}] Withdrawal confirmed", cid)
        storage.save_message(conversation_id, "user", user_message, language)
        name = extracted_data.get("full_name")
        farewell = get_withdrawal_message(language, name)
        storage.save_message(conversation_id, "assistant", farewell, language)
        extracted_data.pop("_pending_withdrawal", None)
        extracted_data.pop("_offensive_strikes", None)
        storage.update_conversation(
            conversation_id,
            status=ConversationStatus.WITHDRAWN,
            stage=current_stage,
            extracted_data=extracted_data,
        )
        log_event(conversation_id, current_stage, "withdrawn")
        await finalize_conversation(conversation_id)
        updated = storage.get_conversation(conversation_id)
        elapsed = time.monotonic() - process_start
        logger.info("[{}] Withdrawal complete in {:.0f}ms", cid, elapsed * 1000)
        return {
            "response": farewell,
            "status": ConversationStatus.WITHDRAWN,
            "stage": current_stage,
            "extracted_data": updated["extracted_data"],
            "language": language,
        }

    # Denial or normal message — clear flag and continue screening
    logger.info("[{}] Withdrawal cancelled, resuming screening", cid)
    extracted_data.pop("_pending_withdrawal", None)
    storage.update_conversation(conversation_id, extracted_data=extracted_data)
    return None


async def _handle_stop_signal(
    conversation_id: str,
    user_message: str,
    extracted_data: dict,
    current_stage: str,
    language: str,
    process_start: float,
) -> dict | None:
    """Handle frontend stop button signal. Returns response dict or None."""
    if user_message.strip() != FRONTEND_STOP_SIGNAL:
        return None

    cid = conversation_id[:8]
    logger.info("[{}] Frontend stop signal received, immediate withdrawal", cid)
    storage.save_message(conversation_id, "user", user_message, language)
    name = extracted_data.get("full_name")
    farewell = get_withdrawal_message(language, name)
    storage.save_message(conversation_id, "assistant", farewell, language)
    extracted_data.pop("_pending_withdrawal", None)
    extracted_data.pop("_offensive_strikes", None)
    storage.update_conversation(
        conversation_id,
        status=ConversationStatus.WITHDRAWN,
        stage=current_stage,
        extracted_data=extracted_data,
    )
    log_event(conversation_id, current_stage, "withdrawn")
    await finalize_conversation(conversation_id)
    updated = storage.get_conversation(conversation_id)
    elapsed = time.monotonic() - process_start
    logger.info("[{}] Immediate withdrawal complete in {:.0f}ms", cid, elapsed * 1000)
    return {
        "response": farewell,
        "status": ConversationStatus.WITHDRAWN,
        "stage": current_stage,
        "extracted_data": updated["extracted_data"],
        "language": language,
    }


def _handle_exit_intent(
    conversation_id: str,
    extracted_data: dict,
    current_stage: str,
    language: str,
    process_start: float,
) -> dict:
    """Handle LLM-detected exit intent by requesting confirmation."""
    cid = conversation_id[:8]
    logger.info("[{}] LLM flagged exit intent", cid)
    extracted_data["_pending_withdrawal"] = True
    storage.update_conversation(conversation_id, extracted_data=extracted_data)
    confirm_msg = get_exit_confirmation_message(language)
    storage.save_message(conversation_id, "assistant", confirm_msg, language)
    elapsed = time.monotonic() - process_start
    logger.info("[{}] Exit confirmation requested in {:.0f}ms", cid, elapsed * 1000)
    return {
        "response": confirm_msg,
        "status": ConversationStatus.IN_PROGRESS,
        "stage": current_stage,
        "extracted_data": extracted_data,
        "language": language,
    }


async def _handle_field_validation(
    conversation_id: str,
    llm_result: LLMScreeningResponse,
    current_stage: str,
    extracted_data: dict,
    language: str,
    conv_status: str,
    process_start: float,
) -> dict:
    """Validate the extracted field, handle DQ/retry/advance."""
    cid = conversation_id[:8]
    validator_fn = VALIDATORS.get(current_stage)
    field_name = STAGE_TO_FIELD.get(ConversationStage(current_stage))

    if validator_fn is None or field_name is None:
        logger.debug("[{}] No validator for stage={}, returning as-is", cid, current_stage)
        storage.save_message(conversation_id, "assistant", llm_result.response, language)
        elapsed = time.monotonic() - process_start
        logger.debug("[{}] Total process_message time: {:.0f}ms", cid, elapsed * 1000)
        return {
            "response": llm_result.response,
            "status": conv_status,
            "stage": current_stage,
            "extracted_data": extracted_data,
            "language": language,
        }

    is_valid, normalized_value, should_disqualify = validator_fn(llm_result.field_value)
    logger.debug(
        "[{}] Validation for field={}: raw={} -> valid={} normalized={} disqualify={}",
        cid, field_name, repr(llm_result.field_value), is_valid, repr(normalized_value), should_disqualify,
    )

    # Disqualification
    if should_disqualify and ConversationStage(current_stage) in HARD_DISQUALIFIERS:
        return await _handle_disqualification(
            conversation_id, llm_result, current_stage, field_name,
            normalized_value, language, process_start,
        )

    # Invalid/unclear value
    if not is_valid:
        return await _handle_invalid_field(
            conversation_id, llm_result, current_stage, field_name,
            extracted_data, language, process_start,
        )

    # Valid answer — store and advance
    return await _handle_valid_field(
        conversation_id, llm_result, current_stage, field_name,
        normalized_value, language, process_start,
    )


_DISQUALIFICATION_MESSAGES = {
    "no_license": {
        "es": "Gracias por tu interés en Grupo Sazón, {name}. Desafortunadamente, el carnet de conducir es un requisito imprescindible para este puesto. Te animamos a que te pongas en contacto con nosotros cuando lo tengas. ¡Mucha suerte!",
        "en": "Thank you for your interest in Grupo Sazón, {name}. Unfortunately, a driver's license is a required qualification for this role. We hope to hear from you again in the future. Good luck!",
    },
    "outside_area": {
        "es": "Gracias por tu tiempo, {name}. Por ahora solo operamos en ciertas ciudades de España y México, y tu zona no está en nuestra área de servicio actual. Te tendremos en mente si expandimos. ¡Hasta pronto!",
        "en": "Thanks for your time, {name}. We currently only operate in select cities across Spain and Mexico, and your area isn't in our current service zone. We'll keep you in mind as we expand. Take care!",
    },
}


async def _handle_disqualification(
    conversation_id: str,
    llm_result: LLMScreeningResponse,
    current_stage: str,
    field_name: str,
    normalized_value: object,
    language: str,
    process_start: float,
) -> dict:
    """Handle hard disqualification (no license, wrong city)."""
    cid = conversation_id[:8]
    reason = "no_license" if current_stage == ConversationStage.LICENSE else "outside_area"
    logger.warning("[{}] DISQUALIFIED at stage={} reason={}", cid, current_stage, reason)

    store_value = normalized_value if normalized_value is not None else llm_result.field_value
    storage.update_extracted_field(conversation_id, field_name, store_value)
    storage.update_extracted_field(conversation_id, "disqualification_reason", reason)
    storage.update_conversation(
        conversation_id,
        status=ConversationStatus.DISQUALIFIED,
        stage=current_stage,
    )

    # Use the system disqualification template so the candidate always
    # receives a proper farewell, even when the LLM's response was just
    # an acknowledgement (e.g. "Thanks!") because it didn't know the
    # validator would disqualify.
    conv_data = storage.get_conversation(conversation_id)
    candidate_name = (conv_data.get("extracted_data") or {}).get("full_name", "candidato/a")
    lang_key = language if language in ("es", "en") else "es"
    response = _DISQUALIFICATION_MESSAGES[reason][lang_key].format(name=candidate_name)

    storage.save_message(conversation_id, "assistant", response, language)
    log_event(conversation_id, current_stage, "disqualified")

    await finalize_conversation(conversation_id)
    updated = storage.get_conversation(conversation_id)
    elapsed = time.monotonic() - process_start
    logger.info("[{}] Disqualification complete in {:.0f}ms", cid, elapsed * 1000)
    return {
        "response": response,
        "status": ConversationStatus.DISQUALIFIED,
        "stage": current_stage,
        "extracted_data": updated["extracted_data"],
        "language": language,
    }


async def _handle_invalid_field(
    conversation_id: str,
    llm_result: LLMScreeningResponse,
    current_stage: str,
    field_name: str,
    extracted_data: dict,
    language: str,
    process_start: float,
) -> dict:
    """Handle invalid/unclear field value with retry logic."""
    cid = conversation_id[:8]
    attempts = storage.get_field_attempts(conversation_id, field_name)
    logger.debug(
        "[{}] Invalid value for field={}, attempt {}/{}",
        cid, field_name, attempts + 1, MAX_CLARIFICATION_ATTEMPTS,
    )

    if attempts < MAX_CLARIFICATION_ATTEMPTS:
        storage.increment_field_attempts(conversation_id, field_name)
        # Append a clarification question so the candidate knows what to provide
        clarification = _get_stage_clarification(current_stage, language)
        combined_response = (
            f"{llm_result.response} {clarification}"
            if clarification
            else llm_result.response
        )
        storage.save_message(conversation_id, "assistant", combined_response, language)
        log_event(conversation_id, current_stage, "retry")
        elapsed = time.monotonic() - process_start
        logger.info("[{}] Retry requested for field={} ({:.0f}ms)", cid, field_name, elapsed * 1000)
        return {
            "response": combined_response,
            "status": ConversationStatus.IN_PROGRESS,
            "stage": current_stage,
            "extracted_data": extracted_data,
            "language": language,
        }

    # Max retries exceeded — store null, move on
    logger.warning("[{}] Max retries exceeded for field={}, storing null and advancing", cid, field_name)
    storage.update_extracted_field(conversation_id, field_name, None)
    log_event(conversation_id, current_stage, "max_retries_exceeded")

    next_stage = get_next_stage(ConversationStage(current_stage), extracted_data)
    if next_stage is None or next_stage == ConversationStage.CLOSING:
        return await _handle_closing(
            conversation_id, llm_result.response, language,
            "needs_review", process_start,
        )

    storage.update_conversation(conversation_id, stage=next_stage.value)
    # Append the next stage's question so the candidate knows the conversation moved on
    next_question = _get_stage_question(next_stage.value, language)
    combined_response = (
        f"{llm_result.response} {next_question}"
        if next_question
        else llm_result.response
    )
    storage.save_message(conversation_id, "assistant", combined_response, language)
    return {
        "response": combined_response,
        "status": ConversationStatus.IN_PROGRESS,
        "stage": next_stage.value,
        "extracted_data": storage.get_conversation(conversation_id)["extracted_data"],
        "language": language,
    }


async def _handle_valid_field(
    conversation_id: str,
    llm_result: LLMScreeningResponse,
    current_stage: str,
    field_name: str,
    normalized_value: object,
    language: str,
    process_start: float,
) -> dict:
    """Store a valid field value and advance to the next stage."""
    cid = conversation_id[:8]
    logger.info("[{}] Field collected: {}={}", cid, field_name, repr(normalized_value))
    storage.update_extracted_field(conversation_id, field_name, normalized_value)
    log_event(conversation_id, current_stage, "field_collected")

    updated_data = storage.get_conversation(conversation_id)["extracted_data"]
    next_stage = get_next_stage(ConversationStage(current_stage), updated_data)
    logger.debug("[{}] Stage transition: {} -> {}", cid, current_stage, next_stage)

    if next_stage is None or next_stage == ConversationStage.CLOSING:
        has_null = any(
            updated_data.get(f) is None
            for f in [
                "full_name", "driver_license", "city_zone",
                "availability", "preferred_schedule",
                "experience_years", "start_date",
            ]
        )
        final_status = "needs_review" if has_null else "qualified"
        return await _handle_closing(
            conversation_id, None, language, final_status, process_start,
        )

    storage.update_conversation(conversation_id, stage=next_stage.value)
    # Append the correct next-stage question so the system (not the LLM)
    # controls which question is asked, preventing stage desync.
    next_question = _get_stage_question(next_stage.value, language)
    combined_response = (
        f"{llm_result.response} {next_question}"
        if next_question
        else llm_result.response
    )
    storage.save_message(conversation_id, "assistant", combined_response, language)
    elapsed = time.monotonic() - process_start
    logger.info("[{}] Advanced to stage={} in {:.0f}ms", cid, next_stage.value, elapsed * 1000)
    return {
        "response": combined_response,
        "status": ConversationStatus.IN_PROGRESS,
        "stage": next_stage.value,
        "extracted_data": storage.get_conversation(conversation_id)["extracted_data"],
        "language": language,
    }


def _get_next_uncollected_stage(
    after_stage: ConversationStage, extracted_data: dict
) -> ConversationStage | None:
    """Return the first stage after *after_stage* whose field is not yet collected.

    Applies the experience_platform skip logic.  Returns ``CLOSING`` if all
    fields are collected, or ``None`` if *after_stage* is already at the end.
    """
    try:
        start_idx = STAGE_ORDER.index(after_stage) + 1
    except ValueError:
        start_idx = 0

    for s in STAGE_ORDER[start_idx:]:
        if s in (ConversationStage.GREETING, ConversationStage.CLOSING):
            if s == ConversationStage.CLOSING:
                return ConversationStage.CLOSING
            continue
        # Skip experience_platform when experience_years == 0
        if s == ConversationStage.EXPERIENCE_PLATFORM:
            exp_years = extracted_data.get("experience_years")
            if exp_years is not None and exp_years == 0:
                continue
        field_name = STAGE_TO_FIELD.get(s)
        if field_name and (field_name not in extracted_data or extracted_data[field_name] is None):
            return s

    return ConversationStage.CLOSING


async def _handle_multi_field_validation(
    conversation_id: str,
    llm_result: LLMScreeningResponse,
    current_stage: str,
    extracted_data: dict,
    language: str,
    conv_status: str,
    process_start: float,
) -> dict:
    """Validate all extracted fields, handle DQ/retry/advance.

    Processes fields in stage order.  Valid fields are stored immediately.
    If a disqualifying field is encountered, processing stops.  Invalid
    'bonus' fields (from future stages) are silently discarded.
    """
    cid = conversation_id[:8]
    ef = llm_result.extracted_fields or {}

    # If nothing was extracted, treat as the current stage field being unclear
    if not ef:
        logger.debug("[{}] No extracted_fields, falling back to current-stage retry", cid)
        # Ensure field_value is None so _handle_field_validation triggers a retry
        retry_result = LLMScreeningResponse(
            field_value=None,
            response=llm_result.response,
            sentiment=llm_result.sentiment,
        )
        return await _handle_field_validation(
            conversation_id, retry_result, current_stage, extracted_data,
            language, conv_status, process_start,
        )

    current_stage_enum = ConversationStage(current_stage)
    current_field = STAGE_TO_FIELD.get(current_stage_enum)

    # Sort extracted fields in stage order for deterministic processing
    ordered_fields: list[tuple[str, object]] = []
    for s in STAGE_ORDER:
        fn = STAGE_TO_FIELD.get(s)
        if fn and fn in ef:
            ordered_fields.append((fn, ef[fn]))

    collected = []
    current_field_invalid = False

    for field_name, raw_value in ordered_fields:
        stage_enum = FIELD_TO_STAGE.get(field_name)
        if stage_enum is None:
            continue
        stage_str = stage_enum.value

        validator_fn = VALIDATORS.get(stage_str)
        if validator_fn is None:
            # No validator — store as-is
            storage.update_extracted_field(conversation_id, field_name, raw_value)
            collected.append(field_name)
            log_event(conversation_id, stage_str, "field_collected")
            logger.info("[{}] Field collected (no validator): {}={}", cid, field_name, repr(raw_value))
            continue

        is_valid, normalized_value, should_disqualify = validator_fn(raw_value)
        logger.debug(
            "[{}] Multi-field validation: field={} raw={} valid={} normalized={} dq={}",
            cid, field_name, repr(raw_value), is_valid, repr(normalized_value), should_disqualify,
        )

        # Disqualification — stop immediately
        if should_disqualify and stage_enum in HARD_DISQUALIFIERS:
            return await _handle_disqualification(
                conversation_id, llm_result, stage_str, field_name,
                normalized_value, language, process_start,
            )

        if is_valid:
            storage.update_extracted_field(conversation_id, field_name, normalized_value)
            collected.append(field_name)
            log_event(conversation_id, stage_str, "field_collected")
            logger.info("[{}] Field collected: {}={}", cid, field_name, repr(normalized_value))
        else:
            # Invalid current-stage field → retry logic
            if field_name == current_field:
                current_field_invalid = True
            # Invalid bonus field → silently discard (will be asked later)
            else:
                logger.debug("[{}] Discarding invalid bonus field: {}={}", cid, field_name, repr(raw_value))

    # If the current stage's field was invalid, handle retry
    if current_field_invalid:
        return await _handle_invalid_field(
            conversation_id, llm_result, current_stage, current_field,
            extracted_data, language, process_start,
        )

    # Determine next uncollected stage
    updated_data = storage.get_conversation(conversation_id)["extracted_data"]
    next_stage = _get_next_uncollected_stage(current_stage_enum, updated_data)
    logger.debug("[{}] Multi-field advance: collected={} next_stage={}", cid, collected, next_stage)

    if next_stage is None or next_stage == ConversationStage.CLOSING:
        has_null = any(
            updated_data.get(f) is None
            for f in [
                "full_name", "driver_license", "city_zone",
                "availability", "preferred_schedule",
                "experience_years", "start_date",
            ]
        )
        final_status = "needs_review" if has_null else "qualified"
        return await _handle_closing(
            conversation_id, None, language, final_status, process_start,
        )

    storage.update_conversation(conversation_id, stage=next_stage.value)
    next_question = _get_stage_question(next_stage.value, language)
    combined_response = (
        f"{llm_result.response} {next_question}"
        if next_question
        else llm_result.response
    )
    storage.save_message(conversation_id, "assistant", combined_response, language)
    elapsed = time.monotonic() - process_start
    logger.info(
        "[{}] Multi-field: collected {} field(s), advanced to stage={} in {:.0f}ms",
        cid, len(collected), next_stage.value, elapsed * 1000,
    )
    return {
        "response": combined_response,
        "status": ConversationStatus.IN_PROGRESS,
        "stage": next_stage.value,
        "extracted_data": storage.get_conversation(conversation_id)["extracted_data"],
        "language": language,
    }


async def _handle_closing(
    conversation_id: str,
    llm_response: str | None,
    language: str,
    status: str,
    process_start: float,
) -> dict:
    """Finalize conversation with closing message."""
    cid = conversation_id[:8]
    updated_data = storage.get_conversation(conversation_id)["extracted_data"]
    final_status = ConversationStatus(status)

    logger.info("[{}] Conversation closing with status={}", cid, final_status.value)
    closing_msg = _get_closing_message(language, updated_data, final_status.value)
    storage.update_conversation(
        conversation_id,
        status=final_status,
        stage=ConversationStage.CLOSING.value,
    )
    storage.save_message(conversation_id, "assistant", closing_msg, language)
    log_event(conversation_id, ConversationStage.CLOSING.value, final_status.value)

    await finalize_conversation(conversation_id)
    updated = storage.get_conversation(conversation_id)
    elapsed = time.monotonic() - process_start
    logger.success("[{}] Conversation finalized as {} in {:.0f}ms", cid, final_status.value, elapsed * 1000)
    return {
        "response": closing_msg,
        "status": final_status,
        "stage": ConversationStage.CLOSING,
        "extracted_data": updated["extracted_data"],
        "language": language,
    }


# ── Sentiment tracking ──


def _record_sentiment(conversation_id: str, sentiment: str | None) -> None:
    """Append sentiment to the conversation's sentiment_history."""
    if not sentiment:
        return
    conv = storage.get_conversation(conversation_id)
    if not conv:
        return
    data = conv["extracted_data"]
    history = data.get("sentiment_history", [])
    history.append(sentiment)
    data["sentiment_history"] = history
    storage.update_conversation(conversation_id, extracted_data=data)


# ── Main entry point ──


async def process_message(conversation_id: str, user_message: str, *, whisper_language: str | None = None) -> dict:
    """Main conversation entry point that serializes concurrent calls."""
    logger.debug("Acquiring lock for conv={}", conversation_id[:8])
    async with _conversation_locks[conversation_id]:
        result = await _process_message_inner(conversation_id, user_message, whisper_language=whisper_language)
    # Clean up lock for terminal conversations to prevent unbounded growth
    if result.get("status") in (
        ConversationStatus.QUALIFIED,
        ConversationStatus.DISQUALIFIED,
        ConversationStatus.NEEDS_REVIEW,
        ConversationStatus.ABANDONED,
        ConversationStatus.WITHDRAWN,
    ):
        _conversation_locks.pop(conversation_id, None)
    return result


async def _process_message_inner(conversation_id: str, user_message: str, *, whisper_language: str | None = None) -> dict:
    """Inner message handler — orchestrates the full pipeline."""
    process_start = time.monotonic()
    cid = conversation_id[:8]

    # Sanitize input
    user_message = sanitize_input(user_message)
    logger.debug("[{}] Processing message: '{}'", cid, user_message[:80])

    # Load conversation state
    conv = storage.get_conversation(conversation_id)
    if conv is None:
        logger.error("[{}] Conversation not found in DB", cid)
        return {
            "response": "Conversation not found.",
            "status": ConversationStatus.IN_PROGRESS,
            "stage": ConversationStage.GREETING,
            "extracted_data": {},
            "language": "es",
        }

    # Check if conversation is already terminal
    if conv["status"] in (
        ConversationStatus.QUALIFIED, ConversationStatus.DISQUALIFIED,
        ConversationStatus.NEEDS_REVIEW, ConversationStatus.ABANDONED,
        ConversationStatus.WITHDRAWN,
    ):
        logger.info("[{}] Message received on terminal conversation (status={})", cid, conv["status"])
        return {
            "response": "This conversation has already ended.",
            "status": conv["status"],
            "stage": conv["stage"],
            "extracted_data": conv["extracted_data"],
            "language": conv.get("language", "es"),
        }

    current_stage = conv["stage"]
    language = conv["language"]
    extracted_data = conv["extracted_data"]
    logger.info(
        "[{}] Current state: stage={} language={} fields_collected={}",
        cid, current_stage, language, list(extracted_data.keys()),
    )

    # ── GUARDRAIL: Pending withdrawal confirmation ──
    withdrawal_result = await _handle_withdrawal_flow(
        conversation_id, user_message, extracted_data,
        current_stage, language, process_start,
    )
    if withdrawal_result is not None:
        return withdrawal_result

    # Whisper audio-level language detection is highly reliable — trust it
    if whisper_language and whisper_language in ("es", "en") and whisper_language != language:
        logger.info("[{}] Language switched via Whisper audio: {} -> {}", cid, language, whisper_language)
        language = whisper_language
        storage.update_conversation(conversation_id, language=language)

    # ── GUARDRAIL: Frontend stop signal (immediate withdrawal) ──
    stop_result = await _handle_stop_signal(
        conversation_id, user_message, extracted_data,
        current_stage, language, process_start,
    )
    if stop_result is not None:
        return stop_result

    # Save user message
    storage.save_message(conversation_id, "user", user_message, language)
    log_event(conversation_id, current_stage, "user_message")

    # ── PRE-LLM GUARDRAIL: Keyword-based offensive content (fast path) ──
    if detect_offensive_content(user_message):
        offensive_result = _handle_offensive(
            conversation_id, extracted_data, current_stage,
            language, process_start, "keyword",
        )
        if offensive_result:
            if offensive_result.pop("_finalize", False):
                await finalize_conversation(conversation_id)
                updated = storage.get_conversation(conversation_id)
                offensive_result["extracted_data"] = updated["extracted_data"]
            return offensive_result

    # ── PRE-LLM GUARDRAIL: Keyword-based exit intent (fast path) ──
    if detect_exit_intent(user_message):
        return _handle_exit_intent(
            conversation_id, extracted_data, current_stage,
            language, process_start,
        )

    # Load full conversation history
    messages = storage.get_messages(conversation_id)
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    logger.debug("[{}] Loaded {} messages for LLM context", cid, len(history))

    # FAQ retrieval
    faq_context = ""
    faq_result = faq.search(user_message, language=language)
    if faq_result is not None:
        faq_context = (
            f"\nFAQ CONTEXT (use this to answer the candidate's question):\n"
            f"Q: {faq_result['question']}\n"
            f"A: {faq_result['answer']}\n"
            f"\nAnswer the candidate's question naturally using the FAQ context above. "
            f"Do NOT ask the next screening question — the system will append it automatically.\n"
        )
        log_event(conversation_id, current_stage, "faq_match")

    # Build system prompt and call LLM with function calling
    system_prompt = _build_system_prompt(
        current_stage, extracted_data, faq_context=faq_context
    )

    # Pre-detect the candidate's message language so we can give the LLM
    # an explicit directive instead of asking it to "detect" (which it ignores).
    # Lingua is unreliable on very short texts (e.g. "Yes I do", "Morning"),
    # so only trust it when the message has enough words for a confident call.
    _MIN_WORDS_FOR_LINGUA = 4
    word_count = len(user_message.split())
    if word_count >= _MIN_WORDS_FOR_LINGUA:
        candidate_lang = _detect_response_language(user_message)
        lang_name = "English" if candidate_lang == "en" else "Spanish"
        lang_hint = (
            f"The candidate's last message is in {lang_name.upper()}. "
            f"You MUST write your ENTIRE response in {lang_name} — "
            f"including the acknowledgment. Do NOT mix languages. "
            f"Do NOT start with Spanish phrases like '¡Perfecto!' or '¡Genial!' if the candidate wrote in English."
        )
    else:
        lang_hint = (
            "CRITICAL: Detect the language of the candidate's LAST message above. "
            "You MUST write your ENTIRE response — including the acknowledgment — in that same language. "
            "Do NOT default to the language used earlier in the conversation. "
            "Do NOT start with Spanish phrases like '¡Perfecto!' or '¡Genial!' if the candidate wrote in English."
        )

    llm_messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "system", "content": lang_hint},
    ]

    tools = _build_tool_schema(current_stage, extracted_data)
    llm_start = time.monotonic()
    raw_response = await call_llm(
        llm_messages, temperature=0.3,
        tools=tools, tool_choice={"type": "function", "function": {"name": "extract_screening_field"}},
    )
    llm_elapsed = time.monotonic() - llm_start
    logger.debug("[{}] LLM call completed in {:.0f}ms", cid, llm_elapsed * 1000)

    # All models exhausted
    if raw_response == LLM_BUSY_RESPONSE:
        logger.warning("[{}] All LLM models busy/exhausted, returning busy message", cid)
        busy_msg = (
            "Estamos experimentando alta demanda en este momento. "
            "Por favor, intenta de nuevo en unos segundos. / "
            "We're experiencing high demand right now. "
            "Please try again in a few seconds."
        )
        return {
            "response": busy_msg,
            "status": conv["status"],
            "stage": current_stage,
            "extracted_data": extracted_data,
            "language": language,
        }

    llm_result = parse_llm_response(raw_response, stage=current_stage)
    llm_result = _ensure_response(llm_result, current_stage, language)

    # Record sentiment
    _record_sentiment(conversation_id, llm_result.sentiment)

    # Language handling:
    # 1. Detect the ACTUAL language of the LLM's response text via Lingua.
    # 2. Trust the candidate's detected language (det_lang) for the appended
    #    question and DB persistence — the candidate's language is the source
    #    of truth, not whatever language the LLM chose to respond in.
    # 3. When the LLM responds in the wrong language, log it but still use
    #    the candidate's language for the next question.
    actual_resp_lang = _detect_response_language(llm_result.response)
    det_lang = llm_result.detected_language

    if det_lang in ("es", "en"):
        # Candidate's language is the source of truth
        if det_lang != language:
            logger.info("[{}] Candidate language switched: {} -> {}", cid, language, det_lang)
        if actual_resp_lang != det_lang:
            logger.warning(
                "[{}] LLM responded in {} but candidate wrote in {} — using candidate language for next question",
                cid, actual_resp_lang, det_lang,
            )
        language = det_lang
        storage.update_conversation(conversation_id, language=det_lang)
    elif actual_resp_lang in ("es", "en"):
        # Fallback: no reliable candidate language, use LLM response language
        if actual_resp_lang != language:
            logger.info("[{}] LLM actually responded in {} (conversation was {})", cid, actual_resp_lang, language)
        language = actual_resp_lang
        storage.update_conversation(conversation_id, language=actual_resp_lang)

    # ── POST-LLM GUARDRAIL: LLM-detected offensive content ──
    if llm_result.is_offensive:
        offensive_result = _handle_offensive(
            conversation_id, extracted_data, current_stage,
            language, process_start, "llm",
        )
        if offensive_result:
            if offensive_result.pop("_finalize", False):
                await finalize_conversation(conversation_id)
                updated = storage.get_conversation(conversation_id)
                offensive_result["extracted_data"] = updated["extracted_data"]
            return offensive_result

    # ── POST-LLM GUARDRAIL: LLM-detected exit intent ──
    if llm_result.exit_intent:
        return _handle_exit_intent(
            conversation_id, extracted_data, current_stage,
            language, process_start,
        )

    # ── FAQ-only turn: answer the question without burning retry attempts ──
    was_faq_turn = faq_result is not None
    ef = llm_result.extracted_fields or {}
    if was_faq_turn and not ef:
        stage_question = _get_stage_question(current_stage, language)
        combined_response = f"{llm_result.response} {stage_question}"
        logger.info("[{}] FAQ-only turn, re-asking stage={}", cid, current_stage)

        storage.save_message(conversation_id, "assistant", combined_response, language)
        log_event(conversation_id, current_stage, "faq_only_turn")
        return {
            "response": combined_response,
            "status": ConversationStatus.IN_PROGRESS,
            "stage": current_stage,
            "extracted_data": extracted_data,
            "language": language,
        }

    # ── Field validation, DQ, retry, and stage advancement (multi-field) ──
    return await _handle_multi_field_validation(
        conversation_id, llm_result, current_stage, extracted_data,
        language, conv["status"], process_start,
    )


# ── Closing messages ──


def _get_closing_message(language: str, extracted_data: dict, status: str) -> str:
    """Return a deterministic closing message based on outcome and language."""
    name = extracted_data.get("full_name", "")

    if status == "qualified":
        if language == "en":
            return (
                f"Excellent, {name}! You have everything we're looking for. "
                "A Grupo Sazón recruiter will be in touch within the next 2 business days. "
                "We're excited you applied!"
            )
        return (
            f"¡Excelente, {name}! Tienes todo lo que necesitamos. "
            "Un recruiter de Grupo Sazón se pondrá en contacto contigo "
            "en los próximos 2 días hábiles. ¡Estamos encantados de que te hayas postulado!"
        )

    if status == "needs_review":
        if language == "en":
            return (
                f"Thank you for your time, {name}. We have your information and a recruiter "
                "will review your application. We'll be in touch soon!"
            )
        return (
            f"Gracias por tu tiempo, {name}. Tenemos tu información y un recruiter "
            "revisará tu solicitud. ¡Te contactaremos pronto!"
        )

    # Generic fallback
    if language == "en":
        return "Thank you for your time. We'll be in touch!"
    return "¡Gracias por tu tiempo! ¡Estaremos en contacto!"


async def finalize_conversation(conversation_id: str) -> None:
    """Run the two-pass extraction + summary and store results.

    Called when a conversation reaches a terminal state. Strips internal
    underscore-prefixed metadata keys (except sentiment_history) before persisting.
    """
    cid = conversation_id[:8]
    finalize_start = time.monotonic()
    try:
        result = await summary.extract_and_summarize(conversation_id)
        # Strip internal guardrail metadata (underscore-prefixed keys)
        # but preserve sentiment_history
        clean_data = {
            k: v for k, v in result["extracted_data"].items()
            if not k.startswith("_")
        }
        # Preserve sentiment_history from current conversation data
        conv = storage.get_conversation(conversation_id)
        if conv and "sentiment_history" in conv["extracted_data"]:
            clean_data["sentiment_history"] = conv["extracted_data"]["sentiment_history"]
        storage.update_conversation(
            conversation_id,
            extracted_data=clean_data,
            summary=result["summary"],
        )
        elapsed = time.monotonic() - finalize_start
        logger.success(
            "[{}] Finalization complete in {:.0f}ms (summary_len={})",
            cid, elapsed * 1000, len(result["summary"]),
        )
        log_event(conversation_id, "finalization", "complete")
    except Exception as e:
        elapsed = time.monotonic() - finalize_start
        logger.error(
            "[{}] Finalization failed after {:.0f}ms: {}", cid, elapsed * 1000, e
        )
        # Conversation still has incrementally-stored data; summary is just missing
