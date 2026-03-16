"""
Lambda Step 3 — LLM Parse Transcript (src/lambda_parse.py)
-----------------------------------------------------------
Triggered by Step Functions after patient info fetch.
Parses transcript using Gemini 3 Pro to extract structured
clinical data (wounds, measurements, treatment plan, etc.)

Input event: output from lambda_patient_info
Output: adds "encounter_state" (serialized JSON) to event
"""
import os
import asyncio
import boto3
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client("s3")
BUCKET = os.environ.get("S3_BUCKET_NAME", "woundcare-bucket")
S3_OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "woundcare")


def _load_secret(secret_name: str) -> str:
    """Fetch secret from AWS Secrets Manager at runtime."""
    try:
        sm = boto3.client("secretsmanager")
        resp = sm.get_secret_value(SecretId=secret_name)
        return resp["SecretString"]
    except Exception as e:
        print(f"[Parse] Warning: Could not load secret {secret_name}: {e}")
        return ""


def handler(event, context):
    """Lambda entry point — called by Step Functions."""
    appointment_id = event["appointment_id"]
    transcript = event["transcript"]
    patient_info = event.get("patient_info")  # May be None
    is_addendum = event.get("is_addendum", False)
    patient_context = event.get("patient_context", "")  # Clinical sentence from Scriberyte DB
    event.get("date_folder")

    # ── Ensure API key is set (from Secrets Manager in prod) ──────
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = _load_secret("woundcare/google-api-key")

    print(f"[Parse] Parsing transcript for: {appointment_id} (addendum={is_addendum})")

    # ── Prepend patient context sentence to transcript for LLM ────
    if patient_context and not is_addendum:
        transcript = f"{patient_context}\n\n{transcript}"
        print("[Parse] Patient context prepended to transcript")

    from src.manager import EncounterManager
    from src.models import PatientInformation

    manager = EncounterManager()

    pi = PatientInformation(**patient_info) if patient_info else None
    provider_id = event.get("provider_id", "default")
    original_audio_filename = event.get("original_audio_filename")

    if is_addendum:
        state = asyncio.run(manager.apply_addendum(appointment_id, transcript, provider_id=provider_id, original_filename=original_audio_filename))
    else:
        state = asyncio.run(manager.create_from_transcript(transcript, appointment_id, provider_id=provider_id, original_filename=original_audio_filename, pre_patient_info=pi, pre_patient_context=patient_context))

    print(f"[Parse] Parsed {len(state.wounds)} wound(s) for {appointment_id}")

    # ── Save state JSON to S3 ──────────────────────────────────────
    # Naming: same base name as original audio, .json extension
    if original_audio_filename:
        base_name = os.path.splitext(original_audio_filename)[0]
        state_key = f"{S3_OUTPUT_PREFIX}/{provider_id}/clinical_data_jsons/{base_name}.json"
    else:
        state_key = f"{S3_OUTPUT_PREFIX}/{provider_id}/clinical_data_jsons/{appointment_id}.json"

    s3_client.put_object(
        Bucket=BUCKET,
        Key=state_key,
        Body=state.model_dump_json(indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"[Parse] State saved to s3://{BUCKET}/{state_key}")

    return {
        **event,
        "state_s3_key": state_key,
        "wound_count": len(state.wounds),
        "encounter_state": state.model_dump(),
    }
