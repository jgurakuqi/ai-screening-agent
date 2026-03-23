"""FastAPI application entry point for the Grupo Sazón screening agent.

Defines the REST API, initializes all services (DB, FAQ, LLM, TTS, STT)
during the lifespan event, and serves the frontend static files.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from backend.logging_config import logger, setup_logging
from backend import storage, faq
from backend.config import (
    generate_conversation_id,
    ConversationStage,
    ENABLE_SCHEDULER,
    REENGAGEMENT_TIMEOUT_MINUTES,
    LLM_PROVIDER,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_PRIMARY_MODEL,
    AZURE_FALLBACK_MODEL,
    OPENROUTER_API_KEY,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    FALLBACK_MODEL_2,
    EXTRA_FREE_MODELS,
    LLM_MAX_RETRIES_PER_MODEL,
    LLM_RETRY_BASE_DELAY_SECONDS,
    LLM_RATE_LIMIT_BASE_DELAY_SECONDS,
    LLM_RATE_LIMIT_MAX_WAIT_SECONDS,
    LLM_PASS_BASE_DELAY_SECONDS,
    LLM_PASS_MAX_WAIT_SECONDS,
    LLM_MAX_FULL_PASSES,
    LLM_JITTER_FRACTION,
    LLM_EMPTY_CONTENT_COOLDOWN_SECONDS,
)
from backend.models import (
    MessageRequest,
    TTSRequest,
    CreateConversationResponse,
    MessageResponse,
    ConversationResponse,
    ConversationListItem,
    MessageRecord,
    AnalyticsResponse,
)
from backend.tts.service import TTSService
from backend.stt.service import STTService
from backend.llm.service import LLMService
from backend.llm.azure_provider import AzureProvider
from backend.llm.openrouter_provider import OpenRouterProvider
from backend.agent import generate_greeting, process_message, set_llm_service
from backend.reengagement import check_and_reengage, reengage_conversation
from backend.privacy import run_data_retention

# Initialize loguru as the sole logging backend (must happen before uvicorn starts logging)
setup_logging()

scheduler = AsyncIOScheduler()


def _build_llm_service() -> LLMService:
    """Build the LLM service and register provider(s) based on LLM_PROVIDER.

    When ``LLM_PROVIDER`` is ``"azure"``, Azure is the primary provider and
    OpenRouter is added as a fallback (if configured).  When ``"openrouter"``,
    only OpenRouter is used.

    Returns:
        Configured ``LLMService`` instance.
    """
    service = LLMService()

    if LLM_PROVIDER == "azure":
        if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
            azure_models = [m for m in [AZURE_PRIMARY_MODEL, AZURE_FALLBACK_MODEL] if m]
            service.add_provider(
                AzureProvider(
                    api_key=AZURE_OPENAI_API_KEY,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_version=AZURE_OPENAI_API_VERSION,
                    models=azure_models,
                    max_retries_per_model=LLM_MAX_RETRIES_PER_MODEL,
                    retry_base_delay=LLM_RETRY_BASE_DELAY_SECONDS,
                    rate_limit_base_delay=LLM_RATE_LIMIT_BASE_DELAY_SECONDS,
                    rate_limit_max_wait=LLM_RATE_LIMIT_MAX_WAIT_SECONDS,
                )
            )
            logger.info("Azure OpenAI provider registered (models={})", azure_models)
        else:
            logger.warning(
                "LLM_PROVIDER=azure but AZURE_OPENAI_API_KEY or "
                "AZURE_OPENAI_ENDPOINT is missing — check .env"
            )

    # OpenRouter: primary when LLM_PROVIDER=openrouter, fallback when azure
    if OPENROUTER_API_KEY:
        service.add_provider(
            OpenRouterProvider(
                api_key=OPENROUTER_API_KEY,
                models=[PRIMARY_MODEL, FALLBACK_MODEL, FALLBACK_MODEL_2] + EXTRA_FREE_MODELS,
                max_retries_per_model=LLM_MAX_RETRIES_PER_MODEL,
                retry_base_delay=LLM_RETRY_BASE_DELAY_SECONDS,
                rate_limit_base_delay=LLM_RATE_LIMIT_BASE_DELAY_SECONDS,
                rate_limit_max_wait=LLM_RATE_LIMIT_MAX_WAIT_SECONDS,
                pass_base_delay=LLM_PASS_BASE_DELAY_SECONDS,
                pass_max_wait=LLM_PASS_MAX_WAIT_SECONDS,
                max_full_passes=LLM_MAX_FULL_PASSES,
                jitter_fraction=LLM_JITTER_FRACTION,
                empty_content_cooldown=LLM_EMPTY_CONTENT_COOLDOWN_SECONDS,
            )
        )
        if LLM_PROVIDER == "azure":
            logger.info("OpenRouter registered as fallback provider")
        else:
            logger.info("OpenRouter registered as primary provider")
    elif LLM_PROVIDER != "azure":
        logger.warning("No OPENROUTER_API_KEY configured! Set it in .env")

    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services for the FastAPI application lifecycle.

    Args:
        app: FastAPI application instance whose ``state`` stores runtime
            services such as TTS and STT.

    Yields:
        Control back to FastAPI once startup is complete. On exit, scheduler
        resources are released.
    """
    # Startup
    logger.info("Starting Grupo Sazón Screening Agent...")
    storage.init_db()
    logger.success("Database initialized")

    # Run data retention purge
    purged = run_data_retention()
    if purged:
        logger.info("Purged {} old conversations on startup", purged)

    # Start scheduler if enabled
    if ENABLE_SCHEDULER:
        scheduler.add_job(
            check_and_reengage,
            "interval",
            minutes=REENGAGEMENT_TIMEOUT_MINUTES,
            args=[REENGAGEMENT_TIMEOUT_MINUTES],
            id="reengagement_check",
        )
        scheduler.start()
        logger.info("Re-engagement scheduler started (every {} min)", REENGAGEMENT_TIMEOUT_MINUTES)
    else:
        logger.debug("Scheduler disabled via ENABLE_SCHEDULER=false")

    # Initialize FAQ knowledge base
    faq.initialize()
    logger.success("FAQ knowledge base initialized")

    # Initialize TTS service
    app.state.tts_service = TTSService()
    logger.info("TTS service initialized")

    # Initialize STT service (loads Whisper model — takes a few seconds on first run)
    app.state.stt_service = STTService()
    logger.info("STT service initialized")

    # Initialize LLM service with provider chain
    llm_service = _build_llm_service()
    set_llm_service(llm_service)
    logger.info("LLM service initialized (provider={})", LLM_PROVIDER)

    logger.success("Application startup complete — serving at http://localhost:8000")
    yield

    # Shutdown
    logger.info("Shutting down...")
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.debug("Scheduler shut down")


app = FastAPI(
    title="Grupo Sazón Screening Agent",
    description="AI-powered candidate screening agent for delivery driver recruitment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.post("/conversations", response_model=CreateConversationResponse)
async def create_conversation():
    """Create a new screening conversation and send the opening greeting.

    Returns:
        ``CreateConversationResponse`` containing the new conversation ID and
        the first assistant message.
    """
    conv_id = generate_conversation_id()
    logger.debug("Generating new conversation with ID: {}", conv_id)

    storage.create_conversation(conv_id)

    # Generate and store greeting
    greeting = generate_greeting("es")
    storage.save_message(conv_id, "assistant", greeting, "es")

    # Advance stage to NAME (greeting is done)
    storage.update_conversation(conv_id, stage=ConversationStage.NAME.value)

    logger.info("New conversation created: {}", conv_id)
    return CreateConversationResponse(
        conversation_id=conv_id,
        greeting_message=greeting,
    )


@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def send_message(conversation_id: str, body: MessageRequest):
    """Process a candidate message and return the updated conversation state.

    Args:
        conversation_id: UUID of the conversation receiving the message.
        body: Request payload containing the message text and optional Whisper
            language hint.

    Returns:
        ``MessageResponse`` with the assistant reply and updated state. If
        processing fails unexpectedly, the function returns a graceful fallback
        response instead of propagating a raw server error.
    """
    logger.debug("Incoming message for conv={}: '{}'", conversation_id[:8], body.message[:80])

    conv = storage.get_conversation(conversation_id)
    if conv is None:
        logger.warning("Message sent to non-existent conversation: {}", conversation_id)
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        result = await process_message(conversation_id, body.message, whisper_language=body.whisper_language)
    except Exception as e:
        logger.error("LLM/processing error for conv={}: {}", conversation_id[:8], e)
        # Return a graceful response instead of a raw 500
        return MessageResponse(
            response="Lo siento, estoy teniendo problemas técnicos. ¿Podrías repetir tu respuesta? / Sorry, I'm experiencing technical issues. Could you repeat your answer?",
            status=conv["status"],
            stage=conv["stage"],
            extracted_data=conv.get("extracted_data", {}),
            language=conv.get("language", "es"),
        )

    logger.info(
        "Response sent for conv={} | status={} stage={} | response_len={}",
        conversation_id[:8], result["status"], result["stage"], len(result["response"]),
    )
    return MessageResponse(
        response=result["response"],
        status=result["status"],
        stage=result["stage"],
        extracted_data=result["extracted_data"],
        language=result.get("language", "es"),
    )


@app.get("/conversations", response_model=list[ConversationListItem])
async def list_conversations():
    """List conversations for the frontend sidebar.

    Returns:
        ``ConversationListItem`` records ordered by most recent activity.
    """
    logger.trace("Listing all conversations")
    rows = storage.list_conversations()
    items = []
    for row in rows:
        extracted = row.get("extracted_data", {})
        full_name = extracted.get("full_name")
        display_name = full_name if full_name else "New candidate"
        last_activity = row.get("last_message_at") or row.get("created_at", "")
        items.append(ConversationListItem(
            id=row["id"],
            status=row["status"],
            display_name=display_name,
            last_activity=last_activity,
        ))
    logger.debug("Returning {} conversations", len(items))
    return items


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Return a single conversation with extracted data and summary fields.

    Args:
        conversation_id: UUID of the conversation to fetch.

    Returns:
        ``ConversationResponse`` for the requested conversation.
    """
    logger.trace("Fetching conversation: {}", conversation_id[:8])
    conv = storage.get_conversation(conversation_id)
    if conv is None:
        logger.warning("GET conversation not found: {}", conversation_id)
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv["id"],
        status=conv["status"],
        stage=conv["stage"],
        language=conv["language"],
        created_at=conv["created_at"],
        last_message_at=conv.get("last_message_at"),
        extracted_data=conv["extracted_data"],
        summary=conv.get("summary"),
    )


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageRecord])
async def get_messages(conversation_id: str):
    """Return the full message history for a conversation.

    Args:
        conversation_id: UUID of the conversation whose messages are needed.

    Returns:
        Chronological ``MessageRecord`` entries for the conversation.
    """
    logger.trace("Fetching messages for conv={}", conversation_id[:8])
    conv = storage.get_conversation(conversation_id)
    if conv is None:
        logger.warning("GET messages for non-existent conversation: {}", conversation_id)
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = storage.get_messages(conversation_id)
    logger.debug("Returning {} messages for conv={}", len(messages), conversation_id[:8])
    return [
        MessageRecord(
            role=m["role"],
            content=m["content"],
            language=m.get("language"),
            created_at=m["created_at"],
        )
        for m in messages
    ]


@app.post("/conversations/{conversation_id}/reengage")
async def trigger_reengagement(conversation_id: str):
    """Manually send a re-engagement message for an in-progress conversation.

    Args:
        conversation_id: UUID of the conversation to nudge.

    Returns:
        Dict containing the re-engagement message text that was sent.
    """
    logger.info("Manual re-engagement triggered for conv={}", conversation_id[:8])
    message = await reengage_conversation(conversation_id)
    if message is None:
        logger.warning("Re-engagement failed: conv={} not found or not in_progress", conversation_id[:8])
        raise HTTPException(
            status_code=400,
            detail="Cannot re-engage: conversation not found or not in progress",
        )
    logger.success("Re-engagement message sent for conv={}", conversation_id[:8])
    return {"message": message}


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Return aggregate analytics across all stored conversations.

    Returns:
        ``AnalyticsResponse`` built from the persistence-layer aggregates.
    """
    logger.trace("Analytics requested")
    data = storage.get_analytics()
    logger.debug("Analytics: total={} qualified={}", data["total_conversations"], data["by_status"].get("qualified", 0))
    return AnalyticsResponse(**data)


@app.post("/tts/synthesize")
async def synthesize_speech(body: TTSRequest):
    """Synthesize audio for a piece of text.

    Args:
        body: Request payload containing the text and target language.

    Returns:
        Audio ``Response`` when server-side synthesis succeeds, or a JSON
        payload instructing the frontend to fall back to browser TTS.
    """
    logger.debug("TTS request: lang={} text_len={}", body.language, len(body.text))

    tts_service: TTSService = app.state.tts_service
    audio_bytes, content_type, use_browser = await tts_service.synthesize(
        body.text, body.language
    )

    if use_browser:
        return JSONResponse({"use_browser_tts": True, "text": body.text})

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={"Content-Disposition": "inline"},
    )


@app.post("/stt/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("es"),
):
    """Transcribe uploaded audio using the configured STT service.

    Args:
        audio: Uploaded audio file from the client.
        language: Requested language hint such as ``"es"`` or ``"en"``.

    Returns:
        JSON payload containing the transcript and detected language. When the
        transcript is empty, the response includes ``empty=True`` instead.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    logger.debug("STT request: lang={} size={} content_type={}",
                 language, len(audio_bytes), audio.content_type)

    stt_service: STTService = app.state.stt_service
    try:
        transcript, detected_language = await stt_service.transcribe(audio_bytes, language)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not transcript.strip():
        return JSONResponse({"transcript": "", "empty": True})

    return {"transcript": transcript.strip(), "detected_language": detected_language}


@app.delete("/reset")
async def reset_all_data():
    """Delete all persisted screening data.

    Returns:
        Dict containing the number of deleted conversation records.
    """
    logger.warning("RESET requested — wiping all data")
    count = storage.delete_all()
    logger.info("Reset complete: {} conversations deleted", count)
    return {"deleted_conversations": count}


# ---- Serve frontend ----

@app.get("/")
async def serve_frontend():
    """Serve the frontend application's main HTML document."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Static files (JS, CSS, etc.) — mounted last so API routes take precedence
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
