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

class EncounterManager:
    """Manages the lifecycle of an encounter: storage, updates, and rendering."""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.getenv("STORAGE_DIR", "data")
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.parser = ClinicalParser()
        self.renderer = NoteRenderer()
        self.transcriber = Transcriber()
        self.docx_gen = DocxGenerator()

    def _get_path(self, appointment_id: str) -> str:
        return os.path.join(self.storage_dir, f"{appointment_id}.json")

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
        with open(self._get_path(state.appointment_id), "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))

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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Appointment {appointment_id} not found")
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
        """Helper: Audio -> Transcript -> State."""
        transcript = await self.transcriber.transcribe(audio_path)
        return await self.create_from_transcript(transcript, appointment_id)

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
