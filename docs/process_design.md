# Process Design Document — Grupo Sazón Candidate Screening Agent

## 1. Conversation Stages

The screening conversation proceeds through ordered stages. Disqualifying fields come first to avoid wasting turns on ineligible candidates.

```
STAGE 0 — Greeting (auto-sent on conversation creation)
  └─> STAGE 1 — Full Name
        └─> STAGE 2 — Driver's License          ← HARD DISQUALIFIER
              ├─ No / unclear after retries ──> DISQUALIFIED / NEEDS_REVIEW
              └─ Yes ──> STAGE 3 — City / Zone  ← HARD DISQUALIFIER
                    ├─ Outside area / unclear ──> DISQUALIFIED / NEEDS_REVIEW
                    └─ Inside area ──> STAGE 4 — Availability
                          └─> STAGE 5 — Preferred Schedule
                                └─> STAGE 6 — Prior Delivery Experience
                                      ├─> SUB-STAGE 6a — Experience Years
                                      └─> SUB-STAGE 6b — Experience Platform (if years > 0)
                                            └─> STAGE 7 — Start Date
                                                  └─> STAGE 8 — Qualified Closing
```

| Stage | Purpose | Required | Disqualifies |
|---|---|---|---|
| 0 — Greeting | Auto-sent. Set tone, introduce the process, ask for name | — | No |
| 1 — Full Name | Collect full name from greeting response | Yes | No |
| 2 — Driver's License | Yes/No question | Yes | Yes (No answer) |
| 3 — City / Zone | Free text, validated against service areas | Yes | Yes (outside area) |
| 4 — Availability | Full-time / part-time / weekends | Yes | No |
| 5 — Preferred Schedule | Morning / afternoon / evening / flexible | Yes | No |
| 6a — Experience Years | How many years of delivery experience | Yes | No |
| 6b — Experience Platform | Which platforms (only asked if years > 0) | Conditional | No |
| 7 — Start Date | When can they begin | Yes | No |
| 8 — Qualified Closing | Confirm next steps | — | No |

**One question per turn.** The agent never asks two fields in the same message. When confirming a valid answer, the agent includes the next question in the same response to keep the flow natural and minimize round-trips.

**Multi-field extraction.** If a candidate volunteers information about future stages in a single message (e.g., "I'm María, I have a license and live in Madrid"), the LLM extracts all mentioned fields at once. Bonus fields from future stages are validated and stored if valid, but discarded silently if invalid. The current-stage field still governs stage advancement — a retry on the current field blocks progression regardless of bonus fields.

**Greeting trigger:** The greeting is generated and stored as an assistant message immediately when a conversation is created. The candidate's first reply should contain their name.

> *"¡Hola! Soy el asistente de selección de Grupo Sazón. Estamos buscando repartidores y me gustaría hacerte unas preguntas rápidas. ¿Cómo te llamas?"*

---

## 2. Data Fields and Validation Rules

The system uses an **LLM-first validation architecture**: the LLM interprets the user's natural language message and extracts a structured value (via JSON mode), then business-rule validators check the extracted value — not the raw text. This means a user saying *"I drive for a living, of course I have one"* is correctly interpreted as `driver_license: true` before the validator ever sees it.

| Field | LLM Extracts | Validator Checks | Hard Disqualifier | After Max Retries |
|---|---|---|---|---|
| full_name | Name string or null | Non-empty, ≥2 chars | No | Store as-is, continue |
| driver_license | true / false / null | Boolean check | Yes (false) | needs_review |
| city_zone | City name or null | Fuzzy match against service areas (score ≥ 80), synonym lookup | Yes (no match) | needs_review |
| availability | Enum value or null | Enum membership | No | needs_review |
| preferred_schedule | Enum value or null | Enum membership | No | needs_review |
| experience_years | Integer ≥ 0 or null | Integer ≥ 0 | No | Store null, continue |
| experience_platforms | List or [] | Known platforms list (unknown accepted) | No | Store [], continue |
| start_date | ISO date, "ASAP", or null | ISO format or "ASAP" | No | Store null, continue |

**Retry logic:** Each field tracks a `clarification_attempts` counter. When attempts reach the maximum (default: 3), the field is stored as null and the conversation advances. If any required field is null at the end, the final status is `needs_review`.

**Disqualification logic:** Only triggered when a disqualifying field returns a definitive negative value (false for license, confirmed non-match for city). Unclear/null values after retries result in `needs_review`, not `disqualified`.

---

## 3. Edge Cases

### Candidate asks a question about the job or company

- A question-detection heuristic checks for `?` and bilingual question starters ("cuánto", "how", "what", etc.).
- If detected, the FAQ module (`faq.py`) embeds the query and searches the bilingual FAQ knowledge base via cosine similarity.
- If a match is found above the similarity threshold (default 0.65), the FAQ Q&A pair is injected into the system prompt as `FAQ CONTEXT`.
- The LLM answers the candidate's question naturally using the FAQ context, then continues with the current screening question in the same message.
- If no FAQ match is found (below threshold), the LLM follows rule 6: "say you don't have that information and suggest they ask their recruiter after screening."
- The screening stage does **not** advance or change when a FAQ question is asked — it's handled as a side-channel within the current turn.
- Example: Candidate at the license stage asks "¿Cuánto pagan?" → Agent answers with the FAQ-grounded salary info, then re-asks the license question.

### Candidate stops responding mid-conversation

- A re-engagement scheduler polls every 30 minutes for conversations with no recent activity.
- Up to 3 follow-up messages are sent, spaced by the timeout interval.
- After 3 hours of inactivity, the conversation is marked as `abandoned`.
- In production this would trigger a Twilio/WhatsApp push notification. In the demo, the message is stored and becomes visible when the candidate returns to the chat.

### Invalid or ambiguous answers

- The agent never repeats the same question verbatim. On retry, it rephrases and gives an example of a valid answer.
- Example: *"¿Tienes carnet de conducir? Por favor responde sí o no."*
- After max retries the agent moves on and marks the field null. The final status becomes `needs_review`.

### Candidate switches language (Spanish ↔ English)

- Language detection uses a two-layer system:
  1. **Lingua** (Spanish/English specialized detector) for text messages of 4+ words. For shorter messages, the LLM detects the language itself via the `detected_language` function-calling field.
  2. **Whisper** language detection from STT when the candidate uses voice mode — this overrides the text-based detection for the current turn.
- The agent always responds in the language of the candidate's **most recent message**.
- Code-switching within a single message defaults to Spanish.
- Detected language is stored per message and per conversation. It is passed to the LLM as a system prompt directive; the LLM returns its detected language via function calling output.

### Candidate wants to stop mid-conversation

- Exit intent is detected via a **dual-detection** system:
  1. **Keyword-based fast path** (`guardrails.py`): checks short messages (≤6 words) against bilingual exit phrases ("quit", "stop", "bye", "no quiero continuar", "quiero parar", "adiós", etc.). This catches obvious cases without an LLM call.
  2. **LLM-based detection**: the `exit_intent` boolean field in function calling output catches more nuanced phrasing.
- On detection, the agent asks for confirmation: *"Are you sure you want to stop the screening process?"*
- If confirmed (yes/sí/ok/claro), the conversation is marked `withdrawn` and finalized with whatever data was collected.
- If denied (no/nope), the confirmation flag is cleared and screening resumes normally.
- The "Stop" button in the UI sends a `[STOP]` signal that triggers immediate withdrawal without confirmation (the button click itself is an explicit intent).
- `withdrawn` is distinct from `abandoned`: withdrawn = voluntary, abandoned = timeout after inactivity.

### Offensive or inappropriate input

- Offensive content is detected via a **dual-detection** system:
  1. **Keyword-based fast path** (`guardrails.py`): a pre-compiled regex with 130+ bilingual offensive words using word-boundary matching for precision.
  2. **LLM-based detection**: the `is_offensive` field in function calling output catches subtler or contextual offensiveness.
- **First offense:** The agent issues a de-escalation warning: *"I understand you might be frustrated, but I'd appreciate it if we could keep things respectful..."* Conversation continues on the same stage.
- **Repeated offenses** (configurable via `MAX_OFFENSIVE_STRIKES`, default: 2): The conversation is terminated as `disqualified` with reason `offensive_behavior`.

### Multiple rapid messages

- An async lock per conversation ensures message processing is serialized. A second message arriving while the first is processing waits for the lock.
- The UI disables the send button while awaiting a response.

---

## 4. Qualified vs. Disqualified Paths

| Status | Trigger | Agent Action | Data Stored |
|---|---|---|---|
| `qualified` | All required fields collected, no disqualifying answer | Send qualified closing message. Finalize. | Full extracted data JSON + summary |
| `disqualified` | Definitive false on a hard-disqualifier field | Send empathetic disqualification message. Stop. Finalize. | Partial data up to disqualification point + summary |
| `needs_review` | Required field null after max retries, or ambiguous disqualifier | Send neutral closing. Flag for human recruiter. Finalize. | All collected fields + null markers + summary |
| `disqualified` (offensive) | Repeated offensive behavior (≥ `MAX_OFFENSIVE_STRIKES`) | Send termination message. Finalize. | Partial data + `disqualification_reason: offensive_behavior` + summary |
| `abandoned` | No activity for 24 hours after re-engagement attempts exhausted | No agent message. Status updated by scheduler. | All collected fields + null markers |
| `withdrawn` | Candidate explicitly requests to stop (text or Stop button), confirmed | Send empathetic farewell. Finalize. | All collected fields up to withdrawal point + summary |

**Finalization always runs** regardless of outcome. Every conversation produces a structured JSON record and a summary, even if cut short at stage 2.

---

## 5. Message Tone and Length Guidelines

| Guideline | Rule |
|---|---|
| Length | Maximum 2–3 sentences per message. Never a wall of text. |
| One question per turn | Never ask two fields in the same message |
| Tone | Warm but efficient. Professional but human. Never robotic or corporate. |
| Disqualification | Always empathetic. Thank the candidate. Mention future opportunities. Never abrupt. |
| Validation errors | Rephrase, give a valid example. Never scold or repeat word-for-word. |
| Closing (qualified) | Enthusiastic. Confirm next steps (recruiter contact within 2 business days). |
| Language | Match the candidate's language. Accents and informal contractions are fine in Spanish. |

**Example — good disqualification message (ES):**
> "Gracias por tu interés en Grupo Sazón, María. Por ahora el carnet de conducir es un requisito imprescindible, pero te animamos a que te pongas en contacto cuando lo tengas. ¡Mucha suerte!"

**Example — good validation retry:**
> "Perdona, no entendí bien tu respuesta. ¿Tienes carnet de conducir? Por favor responde sí o no."

**Example — good qualified closing (EN):**
> "Excellent, John! You have everything we're looking for. A Grupo Sazón recruiter will be in touch within the next 2 business days. We're excited you applied!"

---

## 6. LLM Integration

| Setting | Value |
|---|---|
| Primary provider | Azure OpenAI |
| Primary model | GPT-5.4 Mini |
| Fallback provider | OpenRouter (free-tier model chain) |
| Fallback models | Llama 3.1 8B Instruct (primary) + 22 additional free models with automatic rotation |
| Structured output | Function calling (tool_choice forced) with per-stage field schemas |
| Temperature | 0 for extraction passes, default for conversational responses |

The LLM is called with a **single function tool** whose schema changes per stage. Required output fields on every call: `extracted_fields`, `response`, `detected_language`, `response_language`, `exit_intent`, `is_offensive`, `sentiment`.

---

## 7. Finalization and Summary

Every conversation — regardless of outcome — goes through a **two-pass finalization** when it reaches a terminal status:

1. **Extraction pass** (temperature=0): Re-reads the full conversation history and produces a structured JSON with all screening fields. Deterministic fallbacks ensure output even if the LLM produces malformed JSON.
2. **Summary pass**: Generates a recruiter-facing prose summary highlighting key data points, flags, and recommended next steps.

Both outputs are stored alongside the conversation record.

---

## 8. Bonus Features

### Sentiment tracking

- Per-message sentiment is classified by the LLM via the `sentiment` field in function calling output (values: positive, neutral, frustrated, confused).
- Sentiment history is stored per conversation and aggregated in analytics.

### Voice mode

- **Speech-to-text**: Browser-based audio capture → Faster Whisper transcription on the backend. Whisper's detected language overrides text-based language detection.
- **Text-to-speech**: Agent responses are synthesized via Edge TTS and streamed back as audio.
- The same screening flow and validation logic applies — voice is an input/output layer, not a separate agent.

### Analytics

- Aggregated metrics available via API endpoint: completion rate, qualification rate, drop-off by stage (showing which screening stage candidates were on when they abandoned/withdrew), average turns by outcome (qualified, disqualified, needs review), and average sentiment scores. All metrics are displayed in the frontend analytics breakdown.
