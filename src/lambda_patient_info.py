"""
Lambda Step 2 — Fetch Patient Info from Scriberyte DB (src/lambda_patient_info.py)
-----------------------------------------------------------------------------------
Triggered by Step Functions after transcription.
Queries Scriberyte's MSSQL database to fetch patient demographics.
Gracefully falls back to empty info if DB unavailable.

Input event: output from lambda_transcribe + original fields
Output: adds "patient_info" dict and "patient_context" sentence to event payload
"""
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def handler(event, context):
    """Lambda entry point — called by Step Functions."""
    appointment_id = event["appointment_id"]

    print(f"[PatientInfo] Fetching patient info for meeting_id: {appointment_id}")

    # ── Fetch from Scriberyte DB ──────────────────────────────────
    from src.scriberyte_client import ScriberyteClient
    client = ScriberyteClient()

    patient_info = None
    patient_context = ""

    if client.is_configured():
        try:
            patient_info_obj, patient_context = client.fetch_patient_info(appointment_id)
            if patient_info_obj:
                patient_info = patient_info_obj.model_dump()
                print(f"[PatientInfo] Pre-populated from Scriberyte DB: {patient_info.get('patient_name')}")
            else:
                print(f"[PatientInfo] Meeting {appointment_id} not found in Scriberyte DB — using transcript fallback")
        except Exception as e:
            print(f"[PatientInfo] Scriberyte DB error (non-fatal): {e}")
    else:
        print("[PatientInfo] Scriberyte DB not configured — using transcript fallback")

    return {
        **event,
        "patient_info": patient_info,         # None = fallback to LLM extraction
        "patient_context": patient_context,   # Clinical sentence prepended to transcript
    }
