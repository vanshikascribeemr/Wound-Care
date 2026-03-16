"""
Lambda S3 Trigger (src/lambda_trigger.py)
------------------------------------------
Fired instantly by S3 when audio is uploaded.
Parses the S3 event, determines if new visit or addendum,
and starts the Step Functions pipeline.

Filename convention:
    {timestamp}-{visitID}-{providerUUID}-{chart|addendum}-{version}.mp3

S3 key format:
    woundcare/{providerUUID}/split-audio-files/{filename}

Examples:
    ...-chart-1.mp3  → new visit
    ...-addendum-1.mp3 → addendum
"""
import os
import re
import json
import boto3
import urllib.parse

sfn_client = boto3.client("stepfunctions")
STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN")

UUID_PATTERN = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'


def _extract_appointment_id(filename: str) -> str:
    """Strip extension AND type markers to get the base appointment ID.
    
    Input:  "timestamp-visitID-providerUUID-chart-1.mp3"
    Output: "timestamp-visitID-providerUUID"
    """
    name = os.path.splitext(filename)[0]
    
    # Check for markers used by Scriberyte naming convention
    marker = "-chart-" if "-chart-" in name.lower() else "-addendum-" if "-addendum-" in name.lower() else "_add"
    
    if marker in ["-chart-", "-addendum-"]:
        return re.split(marker, name, flags=re.IGNORECASE)[0]
    else:
        # Fallback for old formats (strip _add suffix)
        return re.split(r'(_add)', name, flags=re.IGNORECASE)[0].strip("-_ ")


def _is_addendum(filename: str) -> bool:
    """Check if filename contains addendum trigger word."""
    return bool(re.search(r"(-addendum-|_add)", filename, re.IGNORECASE))


def _extract_provider_info(s3_key: str, filename: str = None):
    """
    Extract provider UUID — tries S3 path first, then filename.
    
    S3 path format: woundcare/{providerUUID}/split-audio-files/{filename}
    Filename format: {timestamp}-{visitID}-{providerUUID}-{chart|addendum}-{version}.mp3
    """
    # 1. Try from S3 path (folder before 'split-audio-files')
    parts = s3_key.split('/')
    if 'split-audio-files' in parts:
        idx = parts.index('split-audio-files')
        if idx > 0:
            return parts[idx-1]
    
    # 2. Fallback: extract second UUID from filename (providerUUID)
    if filename:
        name = os.path.splitext(filename)[0]
        # Strip -chart-X or -addendum-X marker
        for marker in ['-chart-', '-addendum-']:
            if marker in name.lower():
                name = re.split(marker, name, flags=re.IGNORECASE)[0]
                break
        uuids = re.findall(UUID_PATTERN, name, re.IGNORECASE)
        if len(uuids) >= 2:
            return uuids[1]  # Second UUID = providerUUID
    
    return "default"


def handler(event, context):
    """Lambda entry point — triggered by S3 PUT event."""
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        s3_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

        # VALIDATION: Only trigger on files inside 'split-audio-files'
        if "split-audio-files" not in s3_key:
            print(f"[Trigger] Skipping non-input file: {s3_key}")
            continue

        filename = os.path.basename(s3_key)
        # Skip directory placeholders
        if not filename or s3_key.endswith("/"):
            continue

        appointment_id = _extract_appointment_id(filename)
        is_addendum = _is_addendum(filename)
        provider_id = _extract_provider_info(s3_key, filename)

        print(f"[Trigger] New file: s3://{bucket}/{s3_key}")
        print(f"[Trigger] Provider: {provider_id}, Appointment ID: {appointment_id}, Addendum: {is_addendum}")

        # Start Step Functions pipeline
        pipeline_input = {
            "s3_key": s3_key,
            "appointment_id": appointment_id,
            "is_addendum": is_addendum,
            "bucket": bucket,
            "provider_id": provider_id,
            "original_audio_filename": filename
        }

        response = sfn_client.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=f"{appointment_id}-{context.aws_request_id[:8]}",
            input=json.dumps(pipeline_input),
        )

        print(f"[Trigger] Pipeline started for provider {provider_id}: {response['executionArn']}")

    return {"status": "ok"}

