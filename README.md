# FreshRoute Screening Agent

AI-powered candidate screening agent for FreshRoute, a restaurant chain hiring delivery drivers across 45 locations in Spain and Mexico.

## Setup

1. **Clone the repo**
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env — add your Azure OpenAI API key (or OpenRouter key for fallback)
   ```
4. **Start the FastAPI backend:**
   ```bash
   uvicorn backend.main:app --reload
   ```
   API docs available at http://localhost:8000/docs
5. **Open the web app:**
   Visit http://localhost:8000/ once the backend is running.

## Architecture Overview

```
Vanilla SPA (index.html/app.js/voice.js)
        ↓ HTTP
FastAPI (backend/main.py)
        ↓
agent.py -> AsyncOpenAI (Azure OpenAI) -> SQLite
        ↓
faq.py (RAG) and summary.py (two-pass extraction + summary)
        ↓
APScheduler (reengagement.py) inside FastAPI lifespan

FAQ RAG pipeline:
  User asks question → looks_like_question() heuristic → faq.search()
    → sentence-transformers embedding → cosine similarity vs pre-computed FAQ vectors
    → best match above threshold → injected into system prompt as FAQ CONTEXT
    → LLM answers naturally + continues screening in same message

Voice mode (cross-browser):
  User speaks → MediaRecorder (browser) → audio blob
    → POST /stt/transcribe → faster-whisper (server) → transcribed text
    → POST /conversations/{id}/messages (same endpoint)
    → response text → POST /tts/synthesize → edge-tts audio
    → browser plays audio → mic reactivates after playback
```

- **Vanilla HTML/JS SPA** provides a smoother real-time chat experience, served by FastAPI at `/`
- **Voice mode** adds cross-browser speech input (MediaRecorder + server-side faster-whisper) and neural TTS output (edge-tts) as an overlay on the text chat — works in all modern browsers, no WebSockets required
- **FastAPI** exposes REST endpoints for conversation management
- **agent.py** is the core conversation engine: LLM calls, state machine, validation, FAQ context injection
- **faq.py** is the RAG module: bilingual FAQ retrieval using sentence-transformers embeddings and cosine similarity
- **guardrails.py** handles safety checks: keyword-based fast paths for offensive content (130+ bilingual word regex) and exit intent (bilingual phrase matching on short messages), plus confirmation/denial matching. These fast paths complement the LLM-based detection via function calling fields (`is_offensive`, `exit_intent`).
- **summary.py** handles two-pass extraction (structured JSON) + natural language summary
- **SQLite** stores conversations, messages, and field attempts
- **APScheduler** polls for inactive conversations and sends re-engagement messages
- **loguru** provides unified structured logging across all modules (stdlib interception, colored output, third-party noise suppression)

## Key Design Decisions

- **LLM-first validation with function calling (tool use)** — The LLM returns structured output via function calling with per-stage tool schemas; validators enforce business rules on parsed output. No custom text parsing needed.
- **Pydantic models throughout** — Request/response validation, auto-generated Swagger docs at `/docs`, type safety.
- **Single LLM call per turn** — System prompt instructs the LLM to include the next question when confirming a valid answer. Halves typical response latency.
- **Disqualifying fields first** — Driver's license and city are checked before non-disqualifying fields to avoid wasted turns.
- **`needs_review` instead of disqualifying on ambiguity** — Unclear answers after max retries route to human review, avoiding false negatives.
- **AsyncOpenAI client** — Non-blocking, compatible with FastAPI's async event loop.
- **Full conversation history** sent on every LLM call for coherent context.
- **Two-pass extraction** — Conversational model stays focused on dialogue; extraction is a dedicated LLM call with temperature=0.
- **Incremental field storage** — `extracted_data` updated field-by-field during conversation, not only at finalization. Survives crashes.
- **Per-conversation async locks** — Serializes concurrent messages to prevent race conditions.
- **`get_next_stage()` helper** — Centralizes all stage transition logic including experience-platform skip.
- **RAG via cosine similarity, no vector DB** — For a small FAQ corpus (~48 entries), a full vector database (ChromaDB, FAISS) is unnecessary overhead. Pre-computed numpy embeddings with `np.dot` cosine similarity is simpler, faster, and dependency-free. The bilingual model (`paraphrase-multilingual-MiniLM-L12-v2`) handles Spanish and English queries against the same index. FAQs are stored in both languages (not translated at retrieval time) for maximum match quality.
- **On-demand FAQ injection, not always-on context** — FAQ context is only injected into the system prompt when the user's message looks like a question AND a match is found above threshold. This avoids bloating the prompt on every turn and keeps token usage low.
- **Azure OpenAI with fallback chain** — Azure GPT-5.4-mini as primary provider with OpenRouter fallback for resilience. All configurable via environment variables.
- **Deterministic greeting** — Not LLM-generated, ensures consistent first impression.
- **ASAP accepted as valid start_date** — Messaging UX should not force date specificity.

## LLM Choice

| Priority | Model | Rationale |
|---|---|---|
| Primary | Azure GPT-5.4-mini | Function calling support, reliable structured output, free Azure credits |
| Fallback | OpenRouter free-tier chain | 22-model deep fallback for resilience if Azure is unavailable |

The primary model is Azure OpenAI GPT-5.4-mini, accessed via the Azure OpenAI API with function calling (tool use) for structured output. An OpenRouter fallback chain provides resilience. The architecture is model-agnostic — switching providers requires only environment variable changes.

## Guardrails + Privacy

- **Offensive input detection** — Dual detection: a keyword-based fast path (pre-compiled regex with 130+ bilingual offensive words using word-boundary matching) catches obvious cases without an LLM call, while the LLM detects subtler offensiveness via the `is_offensive` field in function calling output. First offense triggers a de-escalation warning; repeated offenses terminate the conversation as `disqualified` with reason `offensive_behavior`.
- **Voluntary withdrawal** — Dual detection: a keyword-based fast path checks short messages (≤6 words) against bilingual exit phrases, while the LLM catches nuanced phrasing via the `exit_intent` field in function calling output. The agent asks for confirmation before ending as `withdrawn`. A "Stop" button provides immediate withdrawal without confirmation.
- **FAQ answering via RAG** — Candidate questions about pay, benefits, vehicle requirements, training, etc. are answered mid-conversation using retrieval-augmented generation. A bilingual FAQ knowledge base (48 entries in ES/EN across 10 categories) is embedded at startup using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. When a question is detected, the closest FAQ is retrieved via cosine similarity and injected into the system prompt. The LLM answers naturally and continues the screening flow. Questions below the similarity threshold (default 0.65) fall back to "ask your recruiter."
- **Input sanitization** — Control character stripping, 500-character limit.
- **No raw message content in logs** — Event metadata only (conversation_id, stage, timestamp).
- **Data retention** — Conversations older than `DATA_RETENTION_DAYS` (default: 30) are purged on startup.
- **PII minimization** — Only screening fields stored; no IP addresses, device info, or extra identifiers.

## Bonus Features Implemented

1. **Multi-language (ES/EN)** — Two-layer language detection: Lingua (Spanish/English specialized) for text messages of 4+ words; for shorter messages the LLM detects the language via its `detected_language` function-calling field. Whisper language detection from STT overrides in voice mode. Agent responds in the candidate's most recent language. Code-switching defaults to Spanish.
2. **Guardrails + Privacy** — Dual-detection system for offensive input and exit intent (keyword-based fast paths + LLM function calling fields), escalating warnings → termination, voluntary withdrawal with confirmation flow, prompt-level off-topic redirection, input sanitization, metadata-only logging, data retention purge.
3. **Sentiment / Tone adjustment** — LLM-based sentiment detection via function calling. Each message's sentiment (positive/neutral/frustrated/confused) is tracked in `sentiment_history`. Average sentiment displayed in analytics dashboard.
4. **Re-engagement** — APScheduler polls every 30 min; up to 3 follow-up attempts; conversations abandoned after 24 hours. Manual trigger endpoint for demo.
5. **Analytics** — `GET /analytics` endpoint with completion rate, qualification rate, drop-off by stage (displayed in UI), average turns by outcome (qualified, disqualified, needs review). Withdrawn conversations tracked separately for voluntary dropout analytics.
6. **Voice mode** — Cross-browser voice agent using MediaRecorder + server-side faster-whisper for STT and edge-tts for neural TTS. Works in all modern browsers (Chrome, Edge, Brave, Firefox, Safari). Candidates can complete the entire screening flow by speaking. Voice mode is additive (text input always available), with automatic mic activation after TTS playback and echo prevention. Provider abstraction layer enables trivial swap to paid services (ElevenLabs, Deepgram) via environment variables. First startup downloads ~500MB Whisper model (cached in `~/.cache/huggingface/`).
7. **FAQ answering via RAG** — Bilingual knowledge base with 48 FAQ entries across 10 categories covering compensation, benefits, vehicle requirements, training, schedule, job details, requirements, company info, application process, and policies. Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for embedding, cosine similarity for retrieval, and system prompt injection for LLM-grounded answers. Candidates can ask questions mid-screening and get accurate answers without breaking the flow.
8. **ATS integration design** — Full API spec for pushing screened candidates to external ATS platforms (Greenhouse, Lever, Workday, BambooHR, Teamtailor). Includes REST endpoint definitions, field mapping tables, webhook handling, retry/dead-letter logic, and sequence diagrams. See [`docs/ats_integration_design.md`](docs/ats_integration_design.md).

## Running Tests

```bash
cd screening-agent
pytest tests/ -v
```

## Known Limitations

- SQLite write contention at high concurrency (see scaling notes below)
- `dateparser` edge cases on highly colloquial relative date phrasing
- Function calling support requires compatible models (Azure GPT-5.4-mini or equivalent)
- Re-engagement sends stored messages; in production a Twilio/WhatsApp push is required
- Voice mode STT transcription takes 2-4 seconds on CPU (not real-time streaming); first startup downloads ~500MB Whisper model
- edge-tts requires internet connectivity and may be rate-limited under heavy use (falls back to browser SpeechSynthesis automatically)

## Potential Improvements

1. **Twilio WhatsApp integration** — Real async push messaging, the actual channel candidates would use.
2. **Fine-tune sentiment categories** — Add more granular sentiment labels beyond the current four (positive/neutral/frustrated/confused).
3. **Horizontal scaling** — Replace SQLite with Postgres, add Redis queue for async LLM calls. FastAPI already scales horizontally; the DB is the bottleneck.
4. **ATS integration implementation** — The design spec is complete ([`docs/ats_integration_design.md`](docs/ats_integration_design.md)). Next step: implement the adapter layer and REST endpoints to push qualified candidates to Greenhouse, Lever, etc.
5. **Advanced recruiter dashboard** — The current sidebar provides a basic recruiter view (conversation list, status badges, quick stats, analytics). A full dashboard would add search/filter, detailed candidate profiles, and bulk actions.
6. **A/B testing conversation flows** — Track which field order or tone variant produces higher completion rates.
7. **Date parsing hardening** — Handle highly colloquial relative dates beyond what `dateparser` covers.
8. **Paid voice providers** — Swap edge-tts for ElevenLabs (TTS) or faster-whisper for Deepgram (STT) by implementing the `TTSProvider`/`STTProvider` interfaces in `backend/tts/` and `backend/stt/` and updating environment variables. The provider abstraction is already in place.

## Scaling to 10K Candidates/Week

The current architecture hits limits at ~200 concurrent sessions (SQLite has write contention and the app is still deployed as a single service). For 10K/week:

- Replace the vanilla SPA with a React frontend communicating via WebSocket to FastAPI
- Replace SQLite with Postgres behind a connection pool
- Add a Redis queue so LLM calls are processed asynchronously (candidate sees a typing indicator)
- The FastAPI layer is already stateless — scales horizontally with zero changes
- LLM calls go through Azure OpenAI, which handles rate limiting and scaling
- Estimated infrastructure cost at 10K/week: ~$50–100/month

## Sample Conversations

See `data/sample_conversations/` for annotated example transcripts demonstrating:
- Happy path (qualified candidate, Spanish)
- Early disqualification (no license)
- Disqualification (city outside service area)
- Language switching (EN → ES mid-conversation)
