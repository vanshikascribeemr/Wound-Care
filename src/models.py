"""
WoundCare AI - Data Models (src/models.py)
-------------------------------------------
Pydantic models defining the data schema for the entire pipeline.
All data flowing through the system is validated against these models.

Key Models:
  PatientInformation  - Patient demographics (name, DOB, physician, facility)
  WoundDetails        - Per-wound clinical attributes (measurements, drainage, etc.)
  EncounterState      - Full encounter record: patient info + wounds + history + status
  AppointmentStatus   - Enum: BOOKED → RECORDING_SAVED → COMPLETED

  TranscriptProcessRequest  - API request body for /dictate
  AddendumRequest           - API request body for /addendum
  S3ProcessRequest          - API request body for /process-s3-audio and /process-s3-addendum
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import uuid
import re

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
    physician: Optional[str] = Field(None, alias="Physician")
    scribe: Optional[str] = Field(None, alias="Scribe")
    facility: Optional[str] = Field(None, alias="Facility")

    model_config = {
        "populate_by_name": True,
        "extra": "allow"
    }

class EMJustification(BaseModel):
    time_spent_examining: Optional[str] = "-"
    time_spent_documenting: Optional[str] = "-"
    time_spent_coordinating: Optional[str] = "-"
    resolved_wound_sign_off: Optional[str] = "-"
    total_time: Optional[str] = "-"

class WoundDetails(BaseModel):
    """Structured data for a single wound."""
    number: str
    mist_therapy: Optional[str] = None
    location: Optional[str] = None
    outcome: Optional[str] = None
    type: Optional[str] = None
    status: Optional[str] = None
    measurements: Optional[str] = None
    area_sq_cm: Union[str, float, None] = None
    volume_cu_cm: Union[str, float, None] = None
    tunnels: Optional[str] = None
    max_depth: Optional[str] = None
    undermining: Optional[str] = None
    stage_grade: Optional[str] = None
    exudate_amount: Optional[str] = None
    exudate_type: Optional[str] = None
    odor: Optional[str] = None
    wound_margin: Optional[str] = None
    periwound: Optional[str] = None
    necrotic_material: Optional[str] = None
    granulation: Optional[str] = None
    tissue_exposed: Optional[str] = None
    debridement: Optional[str] = None
    primary_dressing: Optional[str] = None
    secondary_dressing: Optional[str] = None
    frequency: Optional[str] = None
    special_equipment: Optional[str] = None

    # Detailed Visit Summary Specific Fields
    debridement_sharp: Optional[bool] = False
    debridement_mechanical: Optional[bool] = False
    debridement_enzymatic: Optional[bool] = False
    debridement_none: Optional[bool] = False
    debridement_details: Optional[str] = None
    offloading_equipment: Optional[str] = None
    additional_care_instructions: Optional[str] = None
    provider_notes: Optional[str] = None

    clinical_summary: Optional[str] = None
    treatment_plan: Optional[str] = None

    attributes: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def calculate_metrics(self) -> 'WoundDetails':
        if not self.measurements or self.measurements == "-":
            return self
        
        # Normalize: replace 'by' with 'x' and remove non-numeric chars except '.' and 'x'
        clean_m = self.measurements.lower().replace("by", "x")
        # Extract float-like numbers
        nums = re.findall(r"(\d+\.?\d*)", clean_m)
        
        if len(nums) >= 2:
            try:
                length = float(nums[0])
                width = float(nums[1])
                area = length * width
                self.area_sq_cm = str(round(area, 2))
                
                if len(nums) >= 3:
                    depth = float(nums[2])
                    volume = area * depth
                    self.volume_cu_cm = str(round(volume, 2))
                else:
                    self.volume_cu_cm = "-"
                
                # Debug print
                # print(f"   [Logic] Calculated Area: {self.area_sq_cm}, Volume: {self.volume_cu_cm} from {self.measurements}")

            except (ValueError, IndexError):
                pass
        return self

    model_config = {
        "populate_by_name": True,
        "extra": "allow"
    }

class EncounterState(BaseModel):
    appointment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider_id: str = "default"
    encounter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: AppointmentStatus = AppointmentStatus.BOOKED
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    patient_information: PatientInformation = Field(default_factory=PatientInformation)
    wounds: List[WoundDetails] = Field(default_factory=list)
    provider_comments: Optional[str] = ""
    treatment_plan: Optional[str] = ""
    em_justification: EMJustification = Field(default_factory=EMJustification)
    original_audio_filename: Optional[str] = None
    
    # Store the original transcripts and addendums
    history: List[Dict[str, Any]] = Field(default_factory=list)

class AddendumRequest(BaseModel):
    appointment_id: str
    transcript: str
    provider_id: Optional[str] = "default"

class TranscriptProcessRequest(BaseModel):
    appointment_id: str
    transcript: str
    provider_id: Optional[str] = "default"

class S3ProcessRequest(BaseModel):
    appointment_id: str
    s3_key: str
    provider_id: Optional[str] = "default"
