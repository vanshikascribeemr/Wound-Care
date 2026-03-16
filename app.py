"""
WoundCare AI - Backend API Server (app.py)
-------------------------------------------
Entry point for the FastAPI backend. Exposes REST API endpoints consumed by
Scriberyte and any other external systems. There is NO UI served from here —
this is a pure backend service.

Key Endpoints:
  POST /appointments/book       - Pre-register a patient appointment (from Scriberyte)
  POST /process-s3-audio        - Manually trigger processing of an audio file in S3
  POST /process-s3-addendum     - Manually trigger addendum processing for an audio file in S3
  POST /dictate                 - Process a raw transcript directly (no audio)
  POST /addendum                - Apply a text addendum to an existing encounter
  GET  /appointments            - List all appointments
  GET  /health                  - Health check for AWS / load balancers

Primary Pipeline (automated):
  Audio uploaded to S3 → S3Watcher detects → Transcribe → Fetch patient info
  from Scriberyte → LLM parse → Generate HTML chart → Upload to S3
"""
from fastapi import FastAPI, HTTPException
from src.models import TranscriptProcessRequest, AddendumRequest, PatientInformation, S3ProcessRequest
from src.manager import EncounterManager
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="WoundCare AI Pipeline API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = EncounterManager()

# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check for AWS / Load Balancers."""
    return {"status": "healthy", "version": "2.0.0"}


# ─────────────────────────────────────────────
# Appointments
# ─────────────────────────────────────────────

@app.get("/appointments")
async def get_appointments():
    """List all appointments."""
    return manager.list_appointments()


@app.post("/appointments/book")
async def book_appointment(req: PatientInformation):
    """Create a new appointment (called by Scriberyte or manually)."""
    try:
        appointment = manager.create_appointment(req)
        return {
            "status": "success",
            "appointment_id": appointment.appointment_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/appointments/{appointment_id}")
async def delete_appointment(appointment_id: str):
    """Delete an appointment."""
    try:
        manager.delete_appointment(appointment_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# S3 Audio Processing (Primary Pipeline)
# ─────────────────────────────────────────────

@app.post("/process-s3-audio")
async def process_s3_audio(req: S3ProcessRequest):
    """Trigger processing for an audio file already in S3 (alternative to watcher)."""
    try:
        encounter = await manager.process_s3_audio_to_state(
            req.s3_key, 
            req.appointment_id, 
            provider_id=req.provider_id
        )
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="S3 file not found")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-s3-addendum")
async def process_s3_addendum(req: S3ProcessRequest):
    """Trigger addendum processing for an audio file already in S3."""
    try:
        encounter = await manager.process_s3_addendum_to_state(
            req.s3_key, 
            req.appointment_id, 
            provider_id=req.provider_id
        )
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="S3 file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Text-Based Processing (Advanced / Manual)
# ─────────────────────────────────────────────

@app.post("/dictate")
async def process_dictation(req: TranscriptProcessRequest):
    """Process a raw transcript directly (no audio needed)."""
    try:
        encounter = await manager.create_from_transcript(
            req.transcript, 
            req.appointment_id, 
            provider_id=req.provider_id
        )
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/addendum")
async def addendum(req: AddendumRequest):
    """Apply a text addendum to an existing encounter."""
    try:
        encounter = await manager.apply_addendum(
            req.appointment_id, 
            req.transcript, 
            provider_id=req.provider_id
        )
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
