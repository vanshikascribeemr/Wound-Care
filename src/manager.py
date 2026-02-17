import os
import json
import jsonpatch
from typing import List
from datetime import datetime
from .models import EncounterState, WoundDetails, PatientInformation, AppointmentStatus
from .parser import ClinicalParser
from .renderer import NoteRenderer
from .transcriber import Transcriber
from .docx_generator import DocxGenerator
import boto3
from botocore.exceptions import ClientError

class EncounterManager:
    """Manages the lifecycle of an encounter: storage, updates, and rendering."""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.getenv("STORAGE_DIR", "data")
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # S3 Configuration
        self.s3_bucket = os.getenv("S3_BUCKET_NAME")
        self.s3_client = boto3.client('s3') if self.s3_bucket else None
        self.s3_prefix = "woundcare" # folder(woundcare)
        
        self.parser = ClinicalParser()
        self.renderer = NoteRenderer()
        self.transcriber = Transcriber()
        self.docx_gen = DocxGenerator()

    def _get_path(self, appointment_id: str) -> str:
        return os.path.join(self.storage_dir, f"{appointment_id}.json")

    def _upload_to_s3(self, local_path: str, s3_key: str):
        """Upload a file to S3 if configured."""
        if self.s3_client and self.s3_bucket:
            try:
                self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                print(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                print(f"S3 Upload Error: {e}")

    def _download_from_s3(self, s3_key: str, local_path: str) -> bool:
        """Download a file from S3 if it exists."""
        if self.s3_client and self.s3_bucket:
            try:
                self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
                print(f"S3 Download Error: {e}")
        return False

    def create_appointment(self, req: PatientInformation) -> EncounterState:
        """Create a new booked appointment."""
        state = EncounterState(
            patient_information=req,
            status=AppointmentStatus.BOOKED
        )
        self.save_state(state)
        return state

    def list_appointments(self) -> List[EncounterState]:
        """List all appointments in the system."""
        # 1. Sync list from S3 if available
        if self.s3_client and self.s3_bucket:
            try:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=f"{self.s3_prefix}/chart/"):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if key.endswith(".json"):
                            appt_id = key.split('/')[-1].replace(".json", "")
                            local_path = self._get_path(appt_id)
                            if not os.path.exists(local_path):
                                self._download_from_s3(key, local_path)
            except Exception as e:
                print(f"Error syncing list from S3: {e}")

        # 2. Scrape local data folder
        appointments = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.storage_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        appointments.append(EncounterState.model_validate(data))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return sorted(appointments, key=lambda x: x.created_at, reverse=True)

    def get_appointment(self, appointment_id: str) -> EncounterState:
        return self.load_state(appointment_id)

    def save_state(self, state: EncounterState):
        state.updated_at = datetime.now()
        local_path = self._get_path(state.appointment_id)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        
        # Sync to S3
        self._upload_to_s3(local_path, f"{self.s3_prefix}/chart/{state.appointment_id}.json")

    def delete_appointment(self, appointment_id: str):
        """Delete an appointment and its associated files. Only allowed for 'Booked' state."""
        state = self.load_state(appointment_id)
        if state.status != AppointmentStatus.BOOKED:
            raise ValueError("Cannot delete an appointment once a clinical record has been created.")

        path = self._get_path(appointment_id)
        if os.path.exists(path):
            os.remove(path)
        
        docx_path = os.path.join(self.storage_dir, f"{appointment_id}.docx")
        if os.path.exists(docx_path):
            os.remove(docx_path)

    def load_state(self, appointment_id: str) -> EncounterState:
        path = self._get_path(appointment_id)
        
        # If not local, try S3
        if not os.path.exists(path):
            s3_key = f"{self.s3_prefix}/chart/{appointment_id}.json"
            if not self._download_from_s3(s3_key, path):
                raise FileNotFoundError(f"Appointment {appointment_id} not found locally or in S3")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return EncounterState.model_validate(data)

    async def create_from_transcript(self, transcript: str, appointment_id: str) -> EncounterState:
        """Process dictation for an existing appointment."""
        state = self.load_state(appointment_id)
        
        parsed_data = await self.parser.parse_transcript(transcript)
        
        # Populate wounds
        state.wounds = [] # Clear if re-dictating
        for w in parsed_data.get("wounds", []):
            state.wounds.append(WoundDetails(**w))
            
        state.provider_comments = parsed_data.get("comments", "")
        state.treatment_plan = parsed_data.get("plan", "")
        state.status = AppointmentStatus.RECORDING_SAVED
        
        # Log history with state snapshot
        state.history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "initial_dictation",
            "transcript": transcript,
            "version": state.version,
            "snapshot": {
                "patient_info": state.patient_information.model_dump(),
                "wounds": [w.model_dump() for w in state.wounds],
                "provider_comments": state.provider_comments,
                "treatment_plan": state.treatment_plan,
                "status": state.status.value
            }
        })
        
        self.save_state(state)
        return state

    async def apply_addendum(self, encounter_id: str, addendum_transcript: str) -> EncounterState:
        """Apply a patch to existing encounter state."""
        state = self.load_state(encounter_id)
        
        # Generate patch from LLM
        patches = await self.parser.generate_patch(state.model_dump(), addendum_transcript)
        
        if patches:
            # Apply patch
            state_dict = state.model_dump()
            patch = jsonpatch.JsonPatch(patches)
            updated_dict = patch.apply(state_dict)
            
            # Re-validate
            updated_state = EncounterState.model_validate(updated_dict)
            updated_state.version += 1
            
            # Log history with state snapshot
            updated_state.history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "addendum",
                "transcript": addendum_transcript,
                "patches": patches,
                "version": updated_state.version,
                "snapshot": {
                    "patient_info": updated_state.patient_information.model_dump(),
                    "wounds": [w.model_dump() for w in updated_state.wounds],
                    "provider_comments": updated_state.provider_comments,
                    "treatment_plan": updated_state.treatment_plan,
                    "status": updated_state.status.value
                }
            })
            
            self.save_state(updated_state)
            return updated_state
        
        return state

    async def process_audio_to_state(self, audio_path: str, appointment_id: str) -> EncounterState:
        """Helper: Audio -> Transcript -> State with S3 storage for artifacts."""
        # 1. Upload raw audio
        self._upload_to_s3(audio_path, f"{self.s3_prefix}/audio/{appointment_id}.wav")
        
        # 2. Transcribe
        transcript = await self.transcriber.transcribe(audio_path)
        
        # 3. Store transcript as separate text file in S3
        transcript_path = os.path.join(self.storage_dir, f"transcript_{appointment_id}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        self._upload_to_s3(transcript_path, f"{self.s3_prefix}/transcript/{appointment_id}.txt")
        
        # 4. Process to chart
        return await self.create_from_transcript(transcript, appointment_id)

    async def process_audio_addendum_to_state(self, audio_path: str, appointment_id: str) -> EncounterState:
        """Helper: Addendum Audio -> Transcript -> Patch with S3 archival."""
        # 1. Upload addendum audio (unique timestamp to avoid overwriting)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._upload_to_s3(audio_path, f"{self.s3_prefix}/audio/{appointment_id}_add_{ts}.wav")
        
        # 2. Transcribe
        transcript = await self.transcriber.transcribe(audio_path)
        
        # 3. Store transcript in S3
        transcript_path = os.path.join(self.storage_dir, f"transcript_{appointment_id}_add_{ts}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        self._upload_to_s3(transcript_path, f"{self.s3_prefix}/transcript/{appointment_id}_add_{ts}.txt")
        
        # 4. Apply addendum
        return await self.apply_addendum(appointment_id, transcript)

    def export_docx(self, encounter_id: str) -> str:
        """Export encounter state to DOCX file."""
        state = self.load_state(encounter_id)
        output_path = os.path.join(self.storage_dir, f"{encounter_id}.docx")
        self.docx_gen.generate(state, output_path)
        return output_path

    def render_encounter(self, encounter_id: str, version: int = None) -> str:
        state = self.load_state(encounter_id)
        
        # Build version list from history — handle old entries without version/snapshot fields
        versions = []
        for idx, entry in enumerate(state.history):
            v_val = entry.get("version")
            if v_val is None:
                # Assign based on position if missing
                v_num = idx + 1
            else:
                v_num = int(v_val)
                
            ts = entry.get("timestamp", "")
            entry_type = entry.get("type", "unknown")
            type_label = "Initial" if entry_type == "initial_dictation" else "Addendum"
            time_str = ts[:16].replace("T", " ") if ts else "\u2014"
            label = f"v{v_num} \u2014 {type_label} ({time_str})"
            
            # Use 'state.version' for highlighting. 
            # If reconstruct_state was called, state.version already reflects requested version.
            versions.append({
                "version": v_num, 
                "label": label, 
                "timestamp": ts
            })
        
        # If no history at all (legacy Booked state), show current version
        if not versions:
            versions.append({"version": state.version, "label": f"v{state.version} \u2014 Current"})
            
        # Reverse so newest is on top
        versions.reverse()
        
        # If a specific version is requested, reconstruct that state
        if version is not None and version != state.version:
            for idx, entry in enumerate(state.history):
                entry_v = int(entry.get("version", idx + 1))
                if entry_v == version:
                    if "snapshot" in entry:
                        snap = entry["snapshot"]
                        if "patient_info" in snap:
                            state.patient_information = PatientInformation.model_validate(snap["patient_info"])
                        state.wounds = [WoundDetails(**w) for w in snap.get("wounds", [])]
                        state.provider_comments = snap.get("provider_comments", "")
                        state.treatment_plan = snap.get("treatment_plan", "")
                    state.version = version
                    break
        
        return self.renderer.render_html(state, versions=versions)

    def get_full_transcript(self, encounter_id: str) -> str:
        """Concatenate initial transcript and all addendums."""
        state = self.load_state(encounter_id)
        full_text = []
        for entry in state.history:
            timestamp = entry.get("timestamp", "")
            type_label = entry.get("type", "dictation").replace("_", " ").title()
            transcript = entry.get("transcript", "")
            if transcript:
                full_text.append(f"--- {type_label} [{timestamp}] ---\n{transcript}\n")
        return "\n".join(full_text)
