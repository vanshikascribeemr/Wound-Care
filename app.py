from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from src.models import TranscriptProcessRequest, AddendumRequest, PatientInformation, AppointmentStatus
from src.manager import EncounterManager
import uvicorn
import os
from typing import List, Optional

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="WoundCare Voice Dictation")

# Enable CORS for production - allow all origins for now, can be restricted later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = EncounterManager()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health_check():
    """Health check for AWS / Load Balancers."""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/appointments")
async def get_appointments():
    """List all appointments."""
    return manager.list_appointments()

@app.post("/appointments/book")
async def book_appointment(req: PatientInformation):
    """Create a new appointment."""
    try:
        appointment = manager.create_appointment(req)
        return {
            "status": "success",
            "appointment_id": appointment.appointment_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dictate")
async def process_dictation(req: TranscriptProcessRequest):
    """Update an appointment from dictation."""
    try:
        encounter = await manager.create_from_transcript(req.transcript, req.appointment_id)
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload-audio")
async def upload_audio(appointment_id: str, file: UploadFile = File(...)):
    """Voice -> Transcript -> Structured Entry for a specific appointment."""
    try:
        # Save temp file
        temp_path = f"data/temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process
        encounter = await manager.process_audio_to_state(temp_path, appointment_id)
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-addendum")
async def upload_addendum(appointment_id: str, file: UploadFile = File(...)):
    """Addendum Voice -> Transcript -> Patch Entry."""
    try:
        temp_path = f"data/temp_add_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe
        transcript = await manager.transcriber.transcribe(temp_path)
        
        # Apply
        encounter = await manager.apply_addendum(appointment_id, transcript)
        
        os.remove(temp_path)
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/addendum")
async def addendum(req: AddendumRequest):
    """Update an existing encounter with new information."""
    try:
        encounter = await manager.apply_addendum(req.appointment_id, req.transcript)
        return {
            "status": "success",
            "appointment_id": encounter.appointment_id,
            "version": encounter.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view/{appointment_id}", response_class=HTMLResponse)
async def view_report(appointment_id: str, version: Optional[int] = None):
    """View the rendered clinical report, optionally at a specific version."""
    try:
        html = manager.render_encounter(appointment_id, version=version)
        return html
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Appointment not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{appointment_id}/docx")
async def download_docx(appointment_id: str):
    """Download the final note as a DOCX file."""
    try:
        path = manager.export_docx(appointment_id)
        state = manager.load_state(appointment_id)
        filename = f"WoundCare_{state.patient_information.patient_name or 'Note'}_{appointment_id[:8]}.docx"
        return FileResponse(path, filename=filename, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{appointment_id}/transcript")
async def download_transcript(appointment_id: str):
    """Download the full session transcript as a text file."""
    try:
        content = manager.get_full_transcript(appointment_id)
        state = manager.load_state(appointment_id)
        filename = f"Transcript_{state.patient_information.patient_name or 'Note'}_{appointment_id[:8]}.txt"
        
        # Save to temp file
        temp_path = f"data/transcript_{appointment_id}.txt"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return FileResponse(temp_path, filename=filename, media_type='text/plain')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcript/{appointment_id}")
async def get_transcript_text(appointment_id: str):
    """Get the full session transcript for preview."""
    try:
        content = manager.get_full_transcript(appointment_id)
        return {"transcript": content}
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

from pydantic import BaseModel as PydanticBaseModel

class RescheduleRequest(PydanticBaseModel):
    date_of_service: str

@app.patch("/appointments/{appointment_id}/reschedule")
async def reschedule_appointment(appointment_id: str, req: RescheduleRequest):
    """Reschedule an appointment to a new date."""
    try:
        state = manager.load_state(appointment_id)
        state.patient_information.date_of_service = req.date_of_service
        manager.save_state(state)
        return {"status": "success", "new_date": req.date_of_service}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Appointment not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
