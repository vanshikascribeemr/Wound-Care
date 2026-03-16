# WoundCare AI — Integration Guide

> **Audience:** Dev Team / Integrators
> **Last Updated:** 2026-02-19
> **Architecture:** Lambda + Step Functions

---

## Technology Stack — 4 Layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — Custom Pipeline (Application Code)               │
│                                                             │
│  src/models.py            ← Data models (EncounterState)    │
│  src/transcriber.py       ← Whisper transcription           │
│  src/scriberyte_client.py ← Patient info fetch              │
│  src/parser.py            ← Gemini 3 Pro LLM parsing        │
│  src/html_generator.py    ← HTML chart generation           │
│  src/manager.py           ← Orchestrates all above          │
│  app.py                   ← FastAPI REST endpoints          │
│                                                             │
│  Written in: Python       Tests: pytest (8/8 passing)       │
└────────────────────────────┬────────────────────────────────┘
                             │ wrapped by
┌────────────────────────────▼────────────────────────────────┐
│  LAYER 2 — Lambda Architecture (Cloud Execution)            │
│                                                             │
│  src/lambda_trigger.py      ← S3 event → start pipeline    │
│  src/lambda_transcribe.py   ← Step 1: Whisper              │
│  src/lambda_patient_info.py ← Step 2: Scriberyte API       │
│  src/lambda_parse.py        ← Step 3: Gemini parse         │
│  src/lambda_generate.py     ← Step 4: HTML → S3            │
│  src/lambda_api.py          ← FastAPI via Mangum            │
│                                                             │
│  Trigger: S3 upload → instant, event-driven                 │
│  Orchestration: Step Functions (retries, error handling)    │
└────────────────────────────┬────────────────────────────────┘
                             │ provisioned by
┌────────────────────────────▼────────────────────────────────┐
│  LAYER 3 — IaC (Infrastructure Definition)                  │
│                                                             │
│  infra/stacks/woundcare_stack.py                            │
│    ├── S3 Bucket                                            │
│    ├── 6 Lambda Functions                                   │
│    ├── Step Functions State Machine                         │
│    ├── API Gateway                                          │
│    ├── Lambda Layer (shared deps)                           │
│    ├── Secrets Manager (API keys)                           │
│    ├── IAM Roles + Scriberyte IAM User                      │
│    └── CloudWatch Logs + X-Ray tracing                     │
│                                                             │
│  Tool: AWS CDK (Python)   Cost: ~$8/month fixed             │
└────────────────────────────┬────────────────────────────────┘
                             │ deployed via
┌────────────────────────────▼────────────────────────────────┐
│  LAYER 4 — CDK (Deployment Tool)                            │
│                                                             │
│  cdk bootstrap  → one-time AWS account setup               │
│  cdk diff       → preview changes before deploy            │
│  cdk deploy     → creates/updates all AWS resources        │
│                                                             │
│  CI/CD: GitHub Actions → auto-deploys on push to main      │
└─────────────────────────────────────────────────────────────┘
```

| Layer | What it is | Owner | Status |
|-------|-----------|-------|--------|
| **Custom Pipeline** | Business logic — transcribe, parse, generate | Dev team | ✅ Done |
| **Lambda Architecture** | Cloud execution wrappers | Dev team | ✅ Done |
| **IaC** | AWS resource definitions | Dev team | ✅ Done |
| **CDK** | Deployment tool (runs IaC) | Cloud Engineer | 🔲 Pending deploy |

---

## System Overview

```
Provider dictates → Scriberyte records audio
                         │
                         │ S3 Upload
                         ▼
              woundcare-bucket/woundcare/audio/
                         │
                         │ S3 Event (instant)
                         ▼
                 lambda_trigger.py
                         │
                         │ Step Functions
                         ▼
          ┌──────────────────────────────┐
          │  1. Transcribe (Whisper)     │
          │  2. Patient Info (Scriberyte)│
          │  3. Parse (Gemini 3 Pro)     │
          │  4. Generate HTML            │
          └──────────────────────────────┘
                         │
                         ▼
              S3: woundcare/{provider_id}/chatgpt_htmls/{id}/latest.html
```

---

## API Endpoints

Base URL: `https://{API_GATEWAY_ID}.execute-api.us-east-1.amazonaws.com/prod`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check → `{"status": "healthy"}` |
| `GET` | `/appointments` | List all appointments |
| `POST` | `/appointments/book` | Book a new appointment |
| `POST` | `/process-s3-audio` | Manually trigger processing for an S3 key |
| `POST` | `/process-s3-addendum` | Manually trigger addendum processing |

### Book Appointment
```bash
curl -X POST https://API_ENDPOINT/appointments/book \
  -H "Content-Type: application/json" \
  -d '{
    "patient_name": "John Doe",
    "dob": "05/12/1965",
    "facility": "Grace Wound Center",
    "physician": "Dr. Smith"
  }'
```

### Manually Trigger Processing
```bash
curl -X POST https://API_ENDPOINT/process-s3-audio \
  -H "Content-Type: application/json" \
  -d '{
    "s3_key": "woundcare/audio/P1005_2026-02-19.wav",
    "appointment_id": "P1005_2026-02-19"
  }'
```

---

## S3 Folder Structure

```
woundcare-bucket/
└── woundcare/
    └── {provider_id}/                  ← Provider-specific folders
        ├── split-audio-files/          ← Scriberyte uploads here
        ├── audio/                      ← Archived audio
        ├── transcribed-speaker-label/  ← Whisper transcripts (.json)
        ├── clinical_data_jsons/        ← Structured clinical results (.json)
        └── chatgpt_htmls/
            └── {appointment_id}/
                ├── {Traceable_Name}.html ← Professional chart
                └── history/
                    ├── v1.html
                    └── v2.html         ← After addendum
```

---

The system generates filenames optimized for traceability and clinical audit.

**Audio Input Example:**
`2026-02-26-11-19-18-427383-{provider_id}-chart-vt1_s60DB.mp3`

**Generated Output Examples:**
*   **HTML Chart**: `...-mp3-prod01-wisper-transcript-SL-gpt-4o-chatgpt-report.html` (in `chatgpt_htmls/`)
*   **Raw Transcript**: `...-mp3-prod01-wisper-transcript-SL.json` (in `transcribed-speaker-label/`)
*   **Clinical JSON**: `...-mp3-prod01-wisper-clinical_classification.json` (in `clinical_data_jsons/`)

> The `_add` suffix is detected by `lambda_trigger.py` and routes to addendum processing.

---

## Appointment ID Format

```
{patient_id}_{date}
Example: P1005_2026-02-19
```

- Must be **unique per visit**
- Must match between audio filename and Scriberyte API
- Date format: `YYYY-MM-DD`

---

## Environment Variables

| Variable | Where Set | Description |
|----------|-----------|-------------|
| `GOOGLE_API_KEY` | Secrets Manager | Gemini 3 Pro API key |
| `OPENAI_API_KEY` | Secrets Manager | OpenAI Whisper key |
| `SCRIBERYTE_API_KEY` | Secrets Manager | Scriberyte API key |
| `SCRIBERYTE_API_URL` | `.env` / Lambda env | Scriberyte base URL |
| `S3_BUCKET_NAME` | Lambda env (CDK) | `woundcare-bucket` |

---

## Scriberyte API Contract

We call this endpoint to pre-populate patient info:

```
GET {SCRIBERYTE_API_URL}/appointments/{appointment_id}
Authorization: Bearer {SCRIBERYTE_API_KEY}
```

Expected response:
```json
{
  "patient_name": "John Doe",
  "dob": "05/12/1965",
  "facility": "Grace Wound Center",
  "physician": "Dr. Smith"
}
```

If Scriberyte API is unavailable, we fall back to extracting patient info from the transcript.

---

## IAM Credentials for Scriberyte

Scriberyte needs these to upload audio to S3:

```bash
# Retrieve from Secrets Manager after CDK deploy
aws secretsmanager get-secret-value \
  --secret-id woundcare/scriberyte-s3-credentials \
  --query SecretString --output text
```

Permissions: **write-only** to `woundcare/audio/*`. Cannot read charts or other folders.

---

## Cost Per Visit

| Component | Cost |
|-----------|------|
| Whisper transcription (~5 min audio) | ~$0.030 |
| Gemini 3 Pro parsing | ~$0.024 |
| AWS Lambda + Step Functions | ~$0.001 |
| S3 storage | ~$0.001 |
| **Total per visit** | **~$0.056** |

---

## Deployment Checklist

- [ ] CDK stack deployed (`cdk deploy`)
- [ ] Google API key set in Secrets Manager
- [ ] OpenAI API key set in Secrets Manager
- [ ] Scriberyte API key set in Secrets Manager
- [ ] Scriberyte S3 credentials shared with Scriberyte team
- [ ] Scriberyte API URL set in Lambda environment
- [ ] End-to-end test: upload audio → verify chart generated
- [ ] GitHub secrets set for CI/CD
