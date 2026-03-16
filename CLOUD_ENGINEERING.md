# WoundCare AI — Cloud Engineering Guide

> **Audience:** Cloud Engineer / DevOps
> **Last Updated:** 2026-02-19
> **Architecture:** Lambda + Step Functions (Option B)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WoundCare AI Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scriberyte                                                 │
│  (IAM User) ──PUT──► S3: woundcare/audio/*.wav              │
│                            │                               │
│                            │ S3 Event                      │
│                            ▼                               │
│                    lambda_trigger.py                        │
│                            │                               │
│                            │ StartExecution                │
│                            ▼                               │
│                   Step Functions Pipeline                   │
│                            │                               │
│              ┌─────────────┼─────────────┐                 │
│              ▼             ▼             ▼                  │
│         Step 1         Step 2        Step 3                 │
│       Transcribe    Patient Info      Parse                 │
│       (Whisper)    (Scriberyte API)  (Gemini)               │
│              └─────────────┼─────────────┘                 │
│                            ▼                               │
│                         Step 4                             │
│                        Generate                            │
│                    (HTML → S3 chart)                       │
│                                                             │
│  API Gateway ──────────► lambda_api.py (FastAPI/Mangum)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## AWS Resources Provisioned by CDK

| Resource | Name | Purpose |
|----------|------|---------|
| S3 Bucket | `woundcare-bucket` | Audio inbox + chart storage |
| Lambda | `woundcare-trigger` | S3 event → starts pipeline |
| Lambda | `woundcare-transcribe` | Whisper transcription |
| Lambda | `woundcare-patient-info` | Scriberyte API fetch |
| Lambda | `woundcare-parse` | Gemini 3 Pro LLM parsing |
| Lambda | `woundcare-generate` | HTML chart generation |
| Lambda | `woundcare-api` | FastAPI via Mangum |
| Step Functions | `woundcare-pipeline` | Pipeline orchestration |
| API Gateway | `woundcare-api` | Public REST API |
| Lambda Layer | `woundcare-deps` | Shared Python dependencies |
| Secrets Manager | `woundcare/google-api-key` | Gemini API key |
| Secrets Manager | `woundcare/openai-api-key` | OpenAI Whisper key |
| Secrets Manager | `woundcare/scriberyte-api-key` | Scriberyte API key |
| Secrets Manager | `woundcare/scriberyte-s3-credentials` | S3 creds for Scriberyte |
| IAM Role | `woundcare-lambda-role` | Lambda execution role |
| IAM User | `scriberyte-audio-uploader` | Scriberyte S3 write access |
| CloudWatch Logs | `/woundcare/pipeline` | Step Functions logs |

---

## Cost Breakdown

### Fixed Monthly (always running)

| Resource | Cost/Month |
|----------|-----------|
| API Gateway (1M requests free tier) | ~$0.00–$3.50 |
| Step Functions (4K free executions/month) | ~$0.00–$1.00 |
| Lambda (1M free invocations/month) | ~$0.00–$1.00 |
| S3 (storage + requests) | ~$2.00 |
| Secrets Manager (4 secrets) | ~$2.00 |
| CloudWatch Logs | ~$1.00 |
| **Total Fixed** | **~$6–$10/month** |

### Per Visit Variable

| Step | Service | Cost/Visit |
|------|---------|-----------|
| Audio transcription (~5 min) | OpenAI Whisper | ~$0.030 |
| LLM chart parsing | Gemini 3 Pro | ~$0.024 |
| S3 storage + requests | AWS S3 | ~$0.001 |
| Step Functions execution | AWS SFN | ~$0.001 |
| **Per-visit total** | | **~$0.056** |

### Total at Scale

| Visits/Month | Fixed | Variable | **Total/Month** |
|-------------|-------|----------|-----------------|
| 0 (idle) | $8 | $0 | **$8** |
| 100 | $8 | $5.60 | **$13.60** |
| 500 | $8 | $28 | **$36** |
| 1,000 | $8 | $56 | **$64** |

> **Savings vs ECS architecture: ~$68/month = $816/year**

---

## First-Time Deployment

### Prerequisites

```bash
# 1. Install AWS CLI
# https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

# 2. Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)

# 3. Install Node.js (required for CDK CLI)
# https://nodejs.org/

# 4. Install CDK CLI
npm install -g aws-cdk

# 5. Install Python CDK dependencies
cd infra
pip install -r requirements.txt
```

### Step 1 — Build Lambda Layer

The Lambda layer bundles all Python dependencies so they're available to all Lambda functions.

```bash
# From project root
mkdir -p lambda_layer/python

pip install \
  fastapi uvicorn pydantic pydantic-settings jinja2 python-dotenv \
  openai google-genai jsonpatch pydub python-multipart \
  boto3 mangum \
  --target lambda_layer/python \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 3.11 \
  --only-binary=:all: \
  --upgrade
```

### Step 2 — Bootstrap CDK (One-Time)

```bash
# Replace with your AWS Account ID and region
cdk bootstrap aws://123456789012/us-east-1 --profile your-aws-profile
```

### Step 3 — Preview Changes

```bash
cd infra
cdk diff
```

Review the output — you should see all Lambda functions, Step Functions, API Gateway, S3, and IAM resources.

### Step 4 — Deploy

```bash
cd infra
cdk deploy --require-approval never
```

Deployment takes ~3–5 minutes. On success, outputs are printed:

```
Outputs:
WoundCareStack.BucketName = woundcare-bucket
WoundCareStack.ApiEndpoint = https://abc123.execute-api.us-east-1.amazonaws.com/prod/
WoundCareStack.StateMachineArn = arn:aws:states:us-east-1:...
WoundCareStack.GoogleSecretArn = arn:aws:secretsmanager:...
WoundCareStack.OpenAiSecretArn = arn:aws:secretsmanager:...
WoundCareStack.ScriberyteS3CredentialsArn = arn:aws:secretsmanager:...
```

**Save these outputs** — you'll need them for the next steps.

### Step 5 — Set API Keys in Secrets Manager

```bash
# Google Gemini API key
aws secretsmanager put-secret-value \
  --secret-id woundcare/google-api-key \
  --secret-string "AIza..."

# OpenAI API key (for Whisper)
aws secretsmanager put-secret-value \
  --secret-id woundcare/openai-api-key \
  --secret-string "sk-..."

# Scriberyte API key (when available)
aws secretsmanager put-secret-value \
  --secret-id woundcare/scriberyte-api-key \
  --secret-string "scriberyte-key-..."
```

### Step 6 — Share Scriberyte S3 Credentials

```bash
# Retrieve the credentials generated by CDK
aws secretsmanager get-secret-value \
  --secret-id woundcare/scriberyte-s3-credentials \
  --query SecretString \
  --output text
```

Share the output with the Scriberyte team. They need:
- `access_key_id`
- `secret_access_key`
- `bucket_name`
- `region`
- `allowed_prefix` → `woundcare/audio/`

### Step 7 — Verify Deployment

```bash
# Test API health
curl https://YOUR_API_ENDPOINT/health
# Expected: {"status": "healthy"}

# Test appointments endpoint
curl https://YOUR_API_ENDPOINT/appointments
# Expected: []

# Test pipeline by uploading a test audio file
aws s3 cp sample-audios/test.wav s3://woundcare-bucket/woundcare/audio/TEST001_2026-02-19.wav

# Check Step Functions execution
aws stepfunctions list-executions \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT:stateMachine:woundcare-pipeline \
  --status-filter RUNNING
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/main.yml`) runs on every push to `main`:

1. **Test** — runs `pytest tests/`
2. **Deploy Lambda Layer** — rebuilds and uploads dependency layer
3. **Deploy CDK** — runs `cdk deploy` with updated Lambda code

### Required GitHub Secrets

Set these in: `GitHub repo → Settings → Secrets → Actions`

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | CI/CD IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | CI/CD IAM user secret key |
| `GOOGLE_API_KEY` | For running tests |
| `OPENAI_API_KEY` | For running tests |
| `S3_BUCKET_NAME` | `woundcare-bucket` |

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
                ├── {Traceable_Name}.html ← Professional chart (with tags)
                └── history/
                    ├── v1.html
                    └── v2.html         ← After addendum
```

---

## Monitoring & Logs

### Step Functions Console
```
AWS Console → Step Functions → woundcare-pipeline → Executions
```
- See each visit's pipeline run
- Click any execution to see step-by-step progress
- Failed steps show exact error + input/output

### Lambda Logs
```
AWS Console → CloudWatch → Log Groups
/aws/lambda/woundcare-transcribe
/aws/lambda/woundcare-parse
/aws/lambda/woundcare-generate
/aws/lambda/woundcare-api
/woundcare/pipeline   ← Step Functions errors
```

### CLI Log Tailing
```bash
# Tail API logs in real-time
aws logs tail /aws/lambda/woundcare-api --follow

# Tail pipeline logs
aws logs tail /aws/lambda/woundcare-transcribe --follow
```

---

## IAM Security Model

| Principal | Permissions | Scope |
|-----------|------------|-------|
| `woundcare-lambda-role` | S3 read/write, Secrets read, SFN start | `woundcare-bucket` only |
| `scriberyte-audio-uploader` | S3 PutObject only | `woundcare/audio/*` only |
| CI/CD IAM user | CDK deploy, ECR push | Scoped to WoundCare resources |

---

## Tear Down

```bash
# Remove all AWS resources (CAUTION: deletes data)
cd infra
cdk destroy

# S3 bucket has RemovalPolicy.RETAIN — delete manually if needed
aws s3 rb s3://woundcare-bucket --force
```

---

## Runbook — Common Operations

### Re-deploy after code change
```bash
cd infra && cdk deploy
```

### Rotate API keys
```bash
aws secretsmanager put-secret-value \
  --secret-id woundcare/google-api-key \
  --secret-string "NEW_KEY_HERE"
# Lambda picks up new key on next cold start (no redeploy needed)
```

### Manually trigger pipeline for a file already in S3
```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT:stateMachine:woundcare-pipeline \
  --input '{"s3_key":"woundcare/audio/P1005_2026-02-19.wav","appointment_id":"P1005_2026-02-19","is_addendum":false}'
```

### Check pipeline execution status
```bash
aws stepfunctions describe-execution \
  --execution-arn arn:aws:states:us-east-1:ACCOUNT:execution:woundcare-pipeline:EXECUTION_ID
```

### Download generated chart
```bash
# Note: Filename is dynamic based on original audio, e.g. ...-report.html
aws s3 cp \
  s3://woundcare-bucket/woundcare/{provider_id}/chatgpt_htmls/{appointment_id}/ \
  ./downloads/ --recursive --exclude "*" --include "*-report.html"
```

### View all charts for a provider
```bash
aws s3 ls s3://woundcare-bucket/woundcare/{provider_id}/chatgpt_htmls/ --recursive
```
