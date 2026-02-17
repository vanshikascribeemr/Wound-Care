from typing import Optional, List, Dict, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from enum import Enum

class AppointmentStatus(str, Enum):
    BOOKED = "Booked"
    RECORDING_SAVED = "Recording Saved"
    SENT_TO_QA = "Sent to QA"
    QA_COMPLETED = "QA Completed"
    DELIVERED = "Delivered"

class PatientInformation(BaseModel):
    patient_name: Optional[str] = Field(None, alias="Patient Name")
    dob: Optional[str] = Field(None, alias="DOB")
    date_of_service: Optional[str] = Field(None, alias="Date of Service")
    physician: Optional[str] = Field(None, alias="Physician/Extender")
    transcriptionist: Optional[str] = Field(None, alias="Transcriptionist")
    facility: Optional[str] = Field(None, alias="Facility")

    model_config = {
        "populate_by_name": True,
        "extra": "allow"
    }

class WoundDetails(BaseModel):
    """Structured data for a single wound with all 21 clinical attributes."""
    number: str
    mist_therapy: Optional[str] = None
    location: Optional[str] = None
    outcome: Optional[str] = None
    type: Optional[str] = None
    status: Optional[str] = None
    measurements: Optional[str] = None
    area_sq_cm: Optional[str] = None
    volume_cu_cm: Optional[str] = None
    tunnels: Optional[str] = None
    max_depth: Optional[str] = None
    undermining: Optional[str] = None
    stage_grade: Optional[str] = None
    drainage: Optional[str] = None
    exudate_type: Optional[str] = None
    odor: Optional[str] = None
    wound_margin: Optional[str] = None
    periwound: Optional[str] = None
    necrotic_material: Optional[str] = None
    granulation: Optional[str] = None
    tissue_exposed: Optional[str] = None
    procedure: Optional[str] = None
    clinical_summary: Optional[str] = None
    treatment_plan: Optional[str] = None
    # Add other attributes from the table if needed
    attributes: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "populate_by_name": True,
        "extra": "allow"
    }

class EncounterState(BaseModel):
    appointment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    encounter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AppointmentStatus = AppointmentStatus.BOOKED
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    patient_information: PatientInformation = Field(default_factory=PatientInformation)
    wounds: List[WoundDetails] = Field(default_factory=list)
    provider_comments: Optional[str] = ""
    treatment_plan: Optional[str] = ""
    
    # Store the original transcripts and addendums
    history: List[Dict[str, Any]] = Field(default_factory=list)

class AppointmentCreateRequest(BaseModel):
    patient_name: str
    dob: str
    date_of_service: str
    physician: str
    facility: str
    transcriptionist: Optional[str] = None

class AddendumRequest(BaseModel):
    appointment_id: str
    transcript: str

class TranscriptProcessRequest(BaseModel):
    appointment_id: str
    transcript: str
