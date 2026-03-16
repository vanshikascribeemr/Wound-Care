"""
Lambda Step 4 — Generate HTML Chart (src/lambda_generate.py)
-------------------------------------------------------------
Final step in the Step Functions pipeline.
Generates the HTML chart from parsed encounter state
and uploads it to S3 as the deliverable.

Input event: output from lambda_parse
Output: adds "chart_s3_key" to event, returns final summary
"""
import os
import tempfile
import boto3
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client("s3")
BUCKET = os.environ.get("S3_BUCKET_NAME", "woundcare-bucket")
S3_OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "woundcare")


def handler(event, context):
    """Lambda entry point — called by Step Functions."""
    appointment_id = event["appointment_id"]
    date_folder = event["date_folder"]
    encounter_state_dict = event["encounter_state"]
    event.get("is_addendum", False)

    print(f"[Generate] Generating HTML for: {appointment_id}")

    from src.models import EncounterState
    from src.html_generator import HtmlGenerator

    # ── Reconstruct EncounterState from dict ───────────────────────
    state = EncounterState(**encounter_state_dict)

    # ── Generate HTML to /tmp ──────────────────────────────────────
    generator = HtmlGenerator()
    tmp_html = os.path.join(tempfile.gettempdir(), f"{appointment_id}.html")
    generator.generate(state, tmp_html)
    print(f"[Generate] HTML generated: {tmp_html}")

    # ── Upload chart HTML to S3 ────────────────────────────────────
    # Naming: same base name as original audio, .html extension
    # No history/ subfolder: version is embedded in filename (chart-1, addendum-1)
    provider_id = event.get("provider_id", "default")
    
    if state.original_audio_filename:
        base_name = os.path.splitext(state.original_audio_filename)[0]
    else:
        base_name = appointment_id

    chart_s3_key = f"{S3_OUTPUT_PREFIX}/{provider_id}/chatgpt_htmls/{base_name}.html"

    with open(tmp_html, "rb") as f:
        s3_client.put_object(
            Bucket=BUCKET,
            Key=chart_s3_key,
            Body=f.read(),
            ContentType="text/html",
        )
    print(f"[Generate] Chart uploaded to s3://{BUCKET}/{chart_s3_key}")

    # ── Cleanup tmp ────────────────────────────────────────────────
    if os.path.exists(tmp_html):
        os.remove(tmp_html)

    # ── 5. Delete original audio from split-audio-files inbox ───────
    # We only delete the audio from the inbox now that the HTML chart has been successfully generated.
    # The audio was already copied to the 'audio/' archive folder in lambda_transcribe.
    s3_key = event.get("s3_key")
    if s3_key and "split-audio-files" in s3_key:
        try:
            s3_client.delete_object(Bucket=BUCKET, Key=s3_key)
            print(f"[Generate] Deleted processed audio from inbox: {s3_key}")
        except Exception as e:
            print(f"[Generate] Warning - Failed to delete inbox audio {s3_key}: {e}")

    return {
        "appointment_id": appointment_id,
        "date_folder": date_folder,
        "chart_s3_key": chart_s3_key,
        "version": state.version,
        "wound_count": event.get("wound_count", 0),
        "status": "success",
    }
