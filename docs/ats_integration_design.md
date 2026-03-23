# ATS Integration API Design

> Design spec for pushing screened candidates from the FreshRoute screening agent to external ATS platforms, receiving status-change webhooks, and exposing recruiter-facing summaries.

---

## Table of Contents

1. [ATS Platform Analysis](#1-ats-platform-analysis)
2. [Universal Candidate Payload](#2-universal-candidate-payload)
3. [REST API Specification](#3-rest-api-specification)
4. [Field Mapping Table](#4-field-mapping-table)
5. [Metadata Specification](#5-metadata-specification)
6. [Sequence Diagrams](#6-sequence-diagrams)
7. [Database Schema Additions](#7-database-schema-additions)
8. [Adapter Architecture](#8-adapter-architecture)
9. [Configuration](#9-configuration)
10. [Integration Point in Existing Code](#10-integration-point-in-existing-code)
11. [File Structure for Implementation](#11-file-structure-for-implementation)

---

## 1. ATS Platform Analysis

The integration layer targets five ATS platforms. Each has distinct API characteristics; the adapter pattern abstracts these differences behind a unified interface.

### Greenhouse (Harvest API)

| Attribute | Detail |
|-----------|--------|
| **Protocol** | REST, JSON |
| **Auth** | API key as HTTP Basic username (password blank), scoped per user |
| **Candidate ingestion** | `POST /v1/candidates` creates a candidate. Applications attached separately via `POST /v1/candidates/{id}/applications` with a `job_id`. Custom fields passed as `[{id, value}]` — IDs must be pre-configured in Greenhouse admin. |
| **Webhooks** | Configurable for candidate stage changes. HMAC signature in `Signature` header. |
| **Rate limits** | 50 requests / 10 seconds |

### Lever

| Attribute | Detail |
|-----------|--------|
| **Protocol** | REST, JSON |
| **Auth** | OAuth2 client credentials grant. Tokens expire and must be refreshed. |
| **Candidate ingestion** | `POST /v1/opportunities` creates an opportunity (Lever's candidate-in-pipeline). Requires a `posting` ID. Custom fields as key-value pairs under `customFields`. |
| **Webhooks** | Stage changes, archive events, hires. HMAC-SHA256 via `X-Lever-Signature` header. |
| **Rate limits** | 10 requests / second |

### Workday

| Attribute | Detail |
|-----------|--------|
| **Protocol** | SOAP primary, REST available for some resources. Complex XML payloads for SOAP, JSON for REST. |
| **Auth** | OAuth2 or WS-Security. Requires a registered Integration System User (ISU) in the tenant. |
| **Candidate ingestion** | `Put_Applicant` SOAP operation or `POST /recruiting/v1/applicants` (REST, if enabled). Deeply nested payloads referencing Workday-internal IDs. |
| **Webhooks** | Business Process Notifications or Workday Studio integrations — not simple webhooks. |
| **Rate limits** | Tenant-specific, typically 100 calls/minute |
| **Recommendation** | Handle via middleware (Workato, MuleSoft, or custom adapter service) rather than direct API calls. The `workday` adapter delegates to a configurable `middleware_url`. |

### BambooHR

| Attribute | Detail |
|-----------|--------|
| **Protocol** | REST, JSON |
| **Auth** | API key as HTTP Basic username (password blank), company subdomain in URL: `https://api.bamboohr.com/api/gateway.php/{subdomain}/v1/` |
| **Candidate ingestion** | `POST /v1/applicant_tracking/applications` creates an applicant attached to a job opening. Custom fields via separate table calls or inline if ATS module is enabled. |
| **Webhooks** | Supported (beta on some plans). Payload includes event type and entity data. |
| **Rate limits** | Undocumented, generally permissive |

### Teamtailor

| Attribute | Detail |
|-----------|--------|
| **Protocol** | REST, JSON:API format (`{data: {type, attributes, relationships}}`) |
| **Auth** | API key in `Authorization: Token token=<key>` header, scoped per company |
| **Candidate ingestion** | `POST /v1/candidates` creates a candidate. Job applications created separately via `POST /v1/job-applications` with relationships to both `candidate` and `job`. Custom fields set as `custom-field-values` relationships. |
| **Webhooks** | Supported for candidate events. Signed with shared secret. |
| **Rate limits** | 100 requests / minute |

---

## 2. Universal Candidate Payload

The internal canonical format used by the screening agent before transformation to any ATS-specific format. Maps directly from `extracted_data` JSON and conversation metadata.

```json
{
  "first_name": "string",
  "last_name": "string",
  "full_name": "string",

  "email": "string | null",
  "phone": "string | null",

  "location": {
    "city": "string",
    "country": "string",
    "region": "string | null"
  },

  "custom_fields": {
    "driver_license": "boolean",
    "availability": "full-time | part-time | weekends",
    "preferred_schedule": "morning | afternoon | evening | flexible",
    "experience_years": "integer",
    "experience_platforms": ["string"],
    "start_date": "string (ISO date or ASAP)"
  },

  "tags": [
    "screening:qualified",
    "source:ai-agent",
    "language:es",
    "availability:full-time",
    "experience:3y"
  ],

  "attachments": [
    {
      "type": "screening_transcript",
      "content_type": "text/plain",
      "filename": "screening_transcript_{conversation_id}.txt",
      "data": "base64-encoded transcript"
    }
  ],

  "metadata": {
    "conversation_id": "uuid",
    "screening_status": "qualified | needs_review",
    "screening_score": "float | null",
    "screening_duration_seconds": "integer",
    "total_messages": "integer",
    "language": "es | en",
    "disqualification_reason": "string | null",
    "screened_at": "ISO 8601",
    "agent_version": "string",
    "summary": "string"
  }
}
```

### Name splitting logic

- If the name contains a space: everything before the last space is `first_name`, the last token is `last_name`.
- If no space: `first_name` = full string, `last_name` = empty string.
- Spanish compound surnames (e.g. "Maria Garcia Lopez") split as first="Maria Garcia", last="Lopez". Acceptable — recruiters can correct in ATS.

### Country derivation

Static map from `city_zone` to country, derived from `data/service_areas.json`:

- **Spain:** Madrid, Barcelona, Valencia, Sevilla, Malaga, Bilbao, Zaragoza, Alicante, Murcia, Palma de Mallorca
- **Mexico:** Mexico City, Guadalajara, Monterrey, Puebla, Cancun, Merida, Queretaro, Leon, Tijuana, Toluca

### Contact fields

`email` and `phone` are **not collected** by the screening agent. They are marked as `null` in the payload and expected to be supplied by the recruiter post-push inside the ATS.

---

## 3. REST API Specification

All endpoints live under `/api/v1/ats/` to separate from existing conversation endpoints.

### 3.1 `POST /api/v1/ats/candidates`

Push a screened candidate to the configured ATS.

**Trigger:** After `finalize_conversation()` completes and status is `qualified` or `needs_review`. Disqualified candidates are NOT pushed (configurable).

#### Request

```json
{
  "conversation_id": "uuid",
  "ats_provider": "greenhouse | lever | workday | bamboohr | teamtailor",
  "ats_config": {
    "api_key": "string",
    "oauth_client_id": "string",
    "oauth_client_secret": "string",
    "subdomain": "string",
    "job_id": "string",
    "middleware_url": "string",
    "custom_field_mapping": {
      "driver_license": "custom_field_12345",
      "availability": "custom_field_12346"
    }
  },
  "options": {
    "include_transcript": true,
    "transcript_format": "txt | pdf",
    "push_tags": true,
    "idempotency_key": "string"
  }
}
```

#### Response `201 Created`

```json
{
  "sync_id": "uuid",
  "ats_candidate_id": "string",
  "ats_provider": "greenhouse",
  "status": "synced",
  "synced_at": "2026-03-20T14:30:00Z",
  "conversation_id": "uuid"
}
```

#### Response `202 Accepted` (async/queued)

```json
{
  "sync_id": "uuid",
  "status": "pending",
  "message": "Candidate push queued. Poll GET /api/v1/ats/sync/{sync_id} for status."
}
```

#### Errors

| Code | Meaning |
|------|---------|
| `400` | Invalid body, unknown provider, conversation not found |
| `404` | Conversation ID does not exist |
| `409` | Duplicate push (idempotency key collision or already synced) |
| `422` | Conversation not in pushable status (`in_progress`, `disqualified`) |
| `502` | ATS API returned an error |
| `503` | ATS API unreachable (triggers retry queue) |

---

### 3.2 `POST /api/v1/ats/webhooks/status-change`

Inbound webhook for ATS to notify of candidate status changes.

#### Headers

| Header | Purpose |
|--------|---------|
| `X-Webhook-Signature` | HMAC-SHA256 of raw body using pre-shared secret |
| `X-Webhook-Event-ID` | Unique event ID for idempotency |
| `Content-Type` | `application/json` |

#### Request (normalized — each adapter transforms raw ATS payload to this)

```json
{
  "event_id": "string",
  "event_type": "candidate.status_changed | candidate.hired | candidate.rejected | candidate.interview_scheduled",
  "ats_provider": "greenhouse",
  "ats_candidate_id": "string",
  "conversation_id": "string",
  "new_status": "string",
  "changed_at": "ISO 8601",
  "raw_payload": {}
}
```

#### Response `200 OK`

```json
{
  "received": true,
  "event_id": "string"
}
```

#### Signature verification

1. Read raw bytes from request body before JSON parsing.
2. Compute `HMAC-SHA256(raw_body, webhook_secret)`.
3. Constant-time comparison with `X-Webhook-Signature`.
4. If mismatch → `401 Unauthorized`.

#### Idempotency

- Store processed `event_id` values in `webhook_events` table.
- On duplicate → `200 OK` with `{"received": true, "duplicate": true}`, no reprocessing.

---

### 3.3 `GET /api/v1/ats/candidates/{conversation_id}/summary`

Recruiter-facing screening summary + transcript.

#### Query Parameters

| Param | Default | Options |
|-------|---------|---------|
| `format` | `json` | `json`, `pdf` |
| `include_transcript` | `true` | `true`, `false` |

#### Response `200 OK` (format=json)

```json
{
  "conversation_id": "uuid",
  "candidate": {
    "first_name": "Maria",
    "last_name": "Garcia",
    "location": { "city": "Madrid", "country": "Spain" }
  },
  "screening": {
    "status": "qualified",
    "summary": "Maria Garcia confirmed a valid driver's license and lives in Madrid...",
    "duration_seconds": 245,
    "total_messages": 18,
    "language": "es",
    "screened_at": "2026-03-20T10:15:00Z"
  },
  "extracted_data": {
    "full_name": "Maria Garcia",
    "driver_license": true,
    "city_zone": "Madrid",
    "availability": "full-time",
    "preferred_schedule": "morning",
    "experience_years": 3,
    "experience_platforms": ["Glovo", "Uber Eats"],
    "start_date": "2026-04-01",
    "disqualification_reason": null
  },
  "transcript": [
    { "role": "assistant", "content": "Hola! Soy el asistente...", "timestamp": "..." },
    { "role": "user", "content": "Hola, me llamo Maria Garcia", "timestamp": "..." }
  ],
  "ats_sync": {
    "synced": true,
    "ats_provider": "greenhouse",
    "ats_candidate_id": "12345",
    "synced_at": "2026-03-20T10:16:00Z"
  }
}
```

For `format=pdf`, returns `Content-Type: application/pdf` with a recruiter-friendly layout.

---

### 3.4 `GET /api/v1/ats/sync/{sync_id}`

Check status of an async candidate push.

#### Response `200 OK`

```json
{
  "sync_id": "uuid",
  "conversation_id": "uuid",
  "ats_provider": "greenhouse",
  "ats_candidate_id": "string | null",
  "status": "synced | pending | failed | retrying",
  "attempts": 3,
  "last_error": "string | null",
  "created_at": "ISO 8601",
  "updated_at": "ISO 8601"
}
```

---

### 3.5 Authentication

| Direction | Method | Detail |
|-----------|--------|--------|
| **Agent → ATS** | Per-provider credentials | OAuth2 (Lever) with token cache + auto-refresh. API keys (Greenhouse, BambooHR, Teamtailor) stored server-side or passed per-request. |
| **ATS → Agent webhook** | HMAC-SHA256 | Per-provider webhook secrets. Reject missing/invalid signatures with `401`. |
| **Recruiter → Agent** | API key in `X-API-Key` header | Validated via middleware. Only `/api/v1/ats/*` endpoints require auth; candidate-facing conversation endpoints remain unauthenticated. |

---

### 3.6 Error Handling and Retry Strategy

Mirrors the existing LLM retry pattern in `backend/agent.py`.

| Scenario | Action |
|----------|--------|
| `502`, `503`, `429`, network timeout | Retry with exponential backoff |
| Other `4xx` | Do not retry; mark `failed` with error |

**Retry parameters:**

| Parameter | Value |
|-----------|-------|
| Base delay | 2 seconds |
| Max retries | 5 (configurable via `ATS_MAX_RETRIES`) |
| Jitter | +/- 30% |
| Max wait | 120 seconds between retries |

**Dead letter queue:**

- After all retries exhausted → sync record marked `failed` with error + timestamp.
- `GET /api/v1/ats/sync/failed` lists failed pushes for manual review.
- `POST /api/v1/ats/sync/{sync_id}/retry` re-queues a failed push.

**Idempotency:**

- `idempotency_key` prevents duplicate candidate creation.
- If a push with the same key already succeeded → return cached `201`.
- Keys expire after 24 hours.

---

## 4. Field Mapping Table

| Agent Field | Universal Field | Greenhouse | Lever | Workday | BambooHR | Teamtailor |
|---|---|---|---|---|---|---|
| `full_name` → split | `first_name` | `first_name` | `name` (single) | `Legal_Name_Data.First_Name` | `firstName` | `attributes.first-name` |
| `full_name` → split | `last_name` | `last_name` | (in `name`) | `Legal_Name_Data.Last_Name` | `lastName` | `attributes.last-name` |
| _(not collected)_ | `email` | `email_addresses[0].value` | `emails[0]` | `Email_Address_Data.Email_Address` | `email` | `attributes.email` |
| _(not collected)_ | `phone` | `phone_numbers[0].value` | `phones[0]` | `Phone_Data.Phone_Number` | `mobilePhone` | `attributes.phone` |
| `city_zone` | `location.city` | `addresses[0].value` | `location` | `Address_Data.Municipality` | `city` | `attributes.city` (custom) |
| `driver_license` | `custom_fields` | `custom_fields[{id, value}]` | `customFields.driver_license` | `Worker_Custom_Field` | Custom table field | `custom-field-values` |
| `availability` | `custom_fields` | `custom_fields[{id, value}]` | `customFields.availability` | `Worker_Custom_Field` | Custom table field | `custom-field-values` |
| `preferred_schedule` | `custom_fields` | `custom_fields[{id, value}]` | `customFields.preferred_schedule` | `Worker_Custom_Field` | Custom table field | `custom-field-values` |
| `experience_years` | `custom_fields` | `custom_fields[{id, value}]` | `customFields.experience_years` | `Worker_Custom_Field` | Custom table field | `custom-field-values` |
| `experience_platforms` | `custom_fields` | `custom_fields[{id, value}]` | `customFields.experience_platforms` | `Worker_Custom_Field` | Custom table field | `custom-field-values` |
| `start_date` | `custom_fields` | `custom_fields[{id, value}]` | `customFields.start_date` | `Worker_Custom_Field` | Custom table field | `custom-field-values` |
| `status` | `tags[]` | `tags[]` | `tags[]` | N/A (notes) | `tags` | `tags` relationship |
| transcript | `attachments[0]` | `attachments` (multipart) | `files` | Separate attachment call | Notes field | Notes field |

**Notes:**
- Greenhouse custom fields require numeric IDs pre-configured in admin. Use `custom_field_mapping` in ATS config.
- Lever uses a single `name` field — adapter concatenates `first_name + " " + last_name`.
- Workday field references use internal naming (e.g. `Legal_Name_Data`). Middleware handles transformation.

---

## 5. Metadata Specification

Every candidate push includes the following metadata for recruiter context.

| Field | Type | Source | Description |
|---|---|---|---|
| `conversation_id` | UUID | `conversations.id` | Internal reference to full conversation |
| `screening_status` | enum | `conversations.status` | `qualified` or `needs_review` |
| `screening_score` | float \| null | Computed | Qualification confidence 0.0–1.0 (see below) |
| `screening_duration_seconds` | integer | Computed | `last_message_at - created_at` |
| `total_messages` | integer | `COUNT(messages)` | Both roles combined |
| `language` | enum | `conversations.language` | `es` or `en` |
| `disqualification_reason` | string \| null | `extracted_data` | `no_license`, `outside_area`, or null |
| `screened_at` | ISO 8601 | `conversations.last_message_at` | When screening concluded |
| `agent_version` | string | FastAPI `app.version` | Agent version that conducted the screening |
| `summary` | string | `conversations.summary` | Recruiter-facing natural language summary |

### Screening score calculation (Phase 1: field completeness)

```
required_fields = [full_name, driver_license, city_zone, availability,
                   preferred_schedule, experience_years, start_date]
filled = count of non-null fields
score = filled / len(required_fields)
```

- **1.0** = all fields cleanly extracted
- **0.71** (5/7) = two fields ambiguous/missing
- Future phases: ML-derived scoring from conversation quality signals

---

## 6. Sequence Diagrams

### Happy path: Screening completes → candidate pushed → recruiter notified

```
Candidate        Agent (FastAPI)       SQLite DB        ATS Adapter        ATS (Greenhouse)
   |                   |                   |                 |                    |
   |-- final answer -->|                   |                 |                    |
   |                   |-- update status ->|                 |                    |
   |                   |   (qualified)     |                 |                    |
   |                   |-- finalize ------>|                 |                    |
   |                   |  (extract+summary)|                 |                    |
   |<-- closing msg ---|                   |                 |                    |
   |                   |                   |                 |                    |
   |                   |== ATS PUSH (async, non-blocking) ===|                    |
   |                   |                   |                 |                    |
   |                   |-- build payload --|                 |                    |
   |                   |  (UniversalCandidate)               |                    |
   |                   |                   |-- create sync ->|                    |
   |                   |                   |   record        |                    |
   |                   |                   |                 |-- transform ------>|
   |                   |                   |                 |   (GH format)      |
   |                   |                   |                 |                    |
   |                   |                   |                 |-- POST /candidates>|
   |                   |                   |                 |<-- 201 + id -------|
   |                   |                   |                 |                    |
   |                   |                   |                 |-- POST /apps ----->|
   |                   |                   |                 |   (attach to job)  |
   |                   |                   |                 |<-- 201 ------------|
   |                   |                   |                 |                    |
   |                   |                   |<- update sync --|                    |
   |                   |                   |   (synced, id)  |                    |
   |                   |                   |                 |                    |
   |                   |======= LATER: STATUS WEBHOOK =======|                    |
   |                   |                   |                 |                    |
   |                   |<-- POST /webhooks/status-change ----------- (hired) ----|
   |                   |-- verify HMAC --->|                 |                    |
   |                   |-- check idemp. -->|                 |                    |
   |                   |-- update sync --->|                 |                    |
   |                   |----------> 200 OK ---------------------------------------->|
```

### Failure path: ATS push fails → dead letter queue

```
Agent (FastAPI)       ATS Adapter        ATS Platform       Dead Letter Queue
   |                     |                    |                    |
   |-- push candidate -->|                    |                    |
   |                     |-- POST /cand. ---->|                    |
   |                     |<-- 503 ------------|                    |
   |                     |-- retry (2s) ----->|                    |
   |                     |<-- 503 ------------|                    |
   |                     |-- retry (~4s) ---->|                    |
   |                     |<-- 503 ------------|                    |
   |                     |-- retry (~8s) ---->|                    |
   |                     |<-- 503 ------------|                    |
   |                     |-- retry (~16s) --->|                    |
   |                     |<-- 503 ------------|                    |
   |                     |                    |                    |
   |                     |-- max retries exceeded ------------->  |
   |                     |                    |  mark "failed"    |
   |<-- 202 (pending) --|                    |                    |
   |                     |                    |                    |
   | ... recruiter calls POST /sync/{id}/retry to re-queue ...   |
```

---

## 7. Database Schema Additions

Two new tables for ATS sync state and webhook idempotency.

```sql
CREATE TABLE IF NOT EXISTS ats_sync_records (
    id TEXT PRIMARY KEY,                    -- UUID
    conversation_id TEXT NOT NULL,
    ats_provider TEXT NOT NULL,             -- greenhouse, lever, workday, bamboohr, teamtailor
    ats_candidate_id TEXT,                  -- Returned by ATS on success
    status TEXT DEFAULT 'pending',          -- pending, synced, failed, retrying
    idempotency_key TEXT UNIQUE,
    attempts INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS webhook_events (
    event_id TEXT PRIMARY KEY,              -- From X-Webhook-Event-ID
    ats_provider TEXT NOT NULL,
    event_type TEXT NOT NULL,
    conversation_id TEXT,
    processed_at TEXT NOT NULL,
    raw_payload TEXT                         -- JSON string for debugging
);
```

---

## 8. Adapter Architecture

Each ATS platform is handled by an adapter implementing a common interface, mirroring the `VALIDATORS` dict pattern in `backend/validator.py`.

```
ATSAdapter (abstract):
    transform(payload: UniversalCandidatePayload) -> provider-specific dict
    push(transformed_payload, ats_config) -> ATSPushResult
    parse_webhook(raw_body, headers) -> NormalizedWebhookEvent
    verify_signature(raw_body, signature, secret) -> bool

Concrete adapters:
    GreenhouseAdapter
    LeverAdapter
    WorkdayAdapter        # Delegates to middleware_url
    BambooHRAdapter
    TeamtailorAdapter

Registry:
    ATS_ADAPTERS = {
        "greenhouse": GreenhouseAdapter,
        "lever": LeverAdapter,
        "workday": WorkdayAdapter,
        "bamboohr": BambooHRAdapter,
        "teamtailor": TeamtailorAdapter,
    }
```

---

## 9. Configuration

New environment variables following the existing pattern in `backend/config.py`.

```env
# ATS Integration
ATS_PUSH_ENABLED=true
ATS_DEFAULT_PROVIDER=greenhouse
ATS_MAX_RETRIES=5
ATS_RETRY_BASE_DELAY_SECONDS=2.0
ATS_RETRY_MAX_WAIT_SECONDS=120.0
ATS_WEBHOOK_SECRET=<hmac-secret>
ATS_API_KEY=<internal-recruiter-api-key>
ATS_PUSH_ON_QUALIFIED=true
ATS_PUSH_ON_NEEDS_REVIEW=false

# Provider-specific (example: Greenhouse)
ATS_GREENHOUSE_API_KEY=<key>
ATS_GREENHOUSE_JOB_ID=<job-id>
ATS_GREENHOUSE_CUSTOM_FIELD_MAP={"driver_license":"12345","availability":"12346"}
```

---

## 10. Integration Point in Existing Code

The ATS push triggers from `finalize_conversation()` in `backend/agent.py`. After the summary is stored, if `ATS_PUSH_ENABLED` is true and the conversation status matches push criteria:

```python
# In finalize_conversation(), after summary is stored:
if ATS_PUSH_ENABLED:
    conv = storage.get_conversation(conversation_id)
    should_push = (
        (conv["status"] == "qualified" and ATS_PUSH_ON_QUALIFIED) or
        (conv["status"] == "needs_review" and ATS_PUSH_ON_NEEDS_REVIEW)
    )
    if should_push:
        # Fire-and-forget — don't block the candidate's closing message
        asyncio.create_task(push_to_ats(conversation_id))
```

The candidate receives their closing message immediately. The ATS push happens in the background with its own retry logic.

---

## 11. File Structure for Implementation

```
screening-agent/
  backend/
    ats/
      __init__.py
      models.py              # Pydantic: UniversalCandidatePayload, ATSPushResult, etc.
      adapter.py             # Abstract adapter + registry
      adapters/
        __init__.py
        greenhouse.py
        lever.py
        workday.py
        bamboohr.py
        teamtailor.py
      service.py             # push_to_ats(), handle_webhook(), get_summary()
      routes.py              # FastAPI router for /api/v1/ats/* endpoints
      auth.py                # API key middleware, HMAC verification
    config.py                # Extend with ATS_* env vars
    storage.py               # Extend with new tables
  tests/
    test_ats_service.py
    test_ats_adapters.py
    test_ats_routes.py
```

Mount in `main.py`:

```python
from backend.ats.routes import ats_router
app.include_router(ats_router, prefix="/api/v1/ats")
```
