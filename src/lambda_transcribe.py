"""
Lambda Step 1 — Transcribe Audio (src/lambda_transcribe.py)
------------------------------------------------------------
Triggered by Step Functions.
Downloads audio from S3, transcribes via OpenAI Whisper,
uploads transcript to S3, returns transcript text.

Filename convention:
    {timestamp}-{visitID}-{providerUUID}-{chart|addendum}-{version}.mp3

Input event (from lambda_trigger):
    {
        "s3_key": "woundcare/{providerUUID}/split-audio-files/{filename}",
        "appointment_id": "{timestamp}-{visitID}-{providerUUID}",
        "is_addendum": false,
        "provider_id": "{providerUUID}",
        "original_audio_filename": "{filename}"
    }

Output: adds "transcript" and "transcript_s3_key" to event
"""
import os
import boto3
import asyncio
import tempfile
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client("s3")
BUCKET = os.environ.get("S3_BUCKET_NAME", "woundcare-bucket")
S3_OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "woundcare")


def handler(event, context):
    """Lambda entry point — called by Step Functions."""
    s3_key = event["s3_key"]
    appointment_id = event["appointment_id"]
    event.get("is_addendum", False)

    # ── 1. Download audio to /tmp (Lambda ephemeral storage) ──────
    ext = os.path.splitext(s3_key)[1] or ".wav"
    tmp_audio = os.path.join(tempfile.gettempdir(), f"{appointment_id}{ext}")

    print(f"[Transcribe] Downloading s3://{BUCKET}/{s3_key}")
    s3_client.download_file(BUCKET, s3_key, tmp_audio)

    # ── 2. Transcribe via Whisper ──────────────────────────────────
    from src.transcriber import Transcriber
    transcriber = Transcriber()
    transcript = asyncio.run(transcriber.transcribe(tmp_audio))
    print(f"[Transcribe] Transcript length: {len(transcript)} chars")

    # ── 3. Upload transcript to S3 ─────────────────────────────────
    # Naming: same base name as audio, .txt extension
    # e.g. "timestamp-visitID-providerUUID-chart-1.mp3" → "…chart-1.txt"
    provider_id = event.get("provider_id", "default")
    original_audio_filename = event.get("original_audio_filename") or os.path.basename(s3_key)
    
    # Strip audio extension to get base name, then add .txt
    base_name = os.path.splitext(original_audio_filename)[0]
    transcript_filename = f"{base_name}.txt"
    
    # Folder: transcribed-speaker-label
    transcript_key = f"{S3_OUTPUT_PREFIX}/{provider_id}/transcribed-speaker-label/{transcript_filename}"

    s3_client.put_object(
        Bucket=BUCKET,
        Key=transcript_key,
        Body=transcript.encode("utf-8"),
        ContentType="text/plain",
    )
    print(f"[Transcribe] Transcript uploaded to s3://{BUCKET}/{transcript_key}")

    # ── 4. Archive audio to provider folder ───────────────────────────
    # Keeping it simple: move/copy to an 'audio' subfolder or archived-audio within provider root
    # Since user didn't specify archive folder, I'll use {provider_id}/audio/ to keep it organized
    audio_dest_key = f"{S3_OUTPUT_PREFIX}/{provider_id}/audio/{os.path.basename(s3_key)}"
    if s3_key != audio_dest_key:
        s3_client.copy_object(
            Bucket=BUCKET,
            CopySource={"Bucket": BUCKET, "Key": s3_key},
            Key=audio_dest_key,
        )
        print(f"[Transcribe] Audio copied to archive at {audio_dest_key}")

    # ── 5. Cleanup tmp ─────────────────────────────────────────────
    if os.path.exists(tmp_audio):
        os.remove(tmp_audio)

    date_folder = datetime.utcnow().strftime("%Y-%m-%d") # Fallback for later steps
    return {
        **event,
        "transcript": transcript,
        "transcript_s3_key": transcript_key,
        "date_folder": date_folder,
    }

