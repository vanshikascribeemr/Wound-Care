"""
WoundCare AI - Encounter Manager (src/manager.py)
--------------------------------------------------
Core orchestrator of the WoundCare AI pipeline. Manages the full lifecycle
of a patient encounter from audio ingestion to HTML chart generation.

Responsibilities:
  - Load / save encounter state (JSON) locally and to S3
  - Fetch patient demographics from Scriberyte API (via ScriberyteClient)
  - Transcribe audio files using Whisper (via Transcriber)
  - Parse transcripts using Gemini LLM (via ClinicalParser)
  - Apply addendums as JSON patches to existing encounter states
  - Generate versioned HTML charts (via HtmlGenerator)
  - Upload all artifacts (audio, transcript, chart, state) to S3 in
    provider folders: {prefix}/{provider_id}/

S3 Output Structure:
  {prefix}/{id}/audio/                         - Archived audio files
  {prefix}/{id}/transcribed-speaker-label/      - Generated transcripts
  {prefix}/{id}/chatgpt_htmls/{appt_id}/       - HTML chart reports
  {prefix}/{id}/clinical_data_jsons/           - JSON state files
"""
import os
import json
import re
import jsonpatch
from typing import List
from datetime import datetime
import time
from .models import EncounterState, WoundDetails, PatientInformation, AppointmentStatus, EMJustification
from .parser import ClinicalParser
from .transcriber import Transcriber
from .html_generator import HtmlGenerator
import boto3
from botocore.exceptions import ClientError
from .scriberyte_client import ScriberyteClient
from .utils import get_output_basename

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
        self.s3_prefix = os.getenv("S3_OUTPUT_PREFIX", "woundcare")
        
        self.parser = ClinicalParser()
        self.transcriber = Transcriber()
        self.html_gen = HtmlGenerator()
        self.scriberyte = ScriberyteClient()

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

    def _delete_from_s3(self, s3_key: str):
        """Delete a file from S3 if configured."""
        if self.s3_client and self.s3_bucket:
            try:
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                print(f"Deleted s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                print(f"S3 Delete Error: {e}")

    def _copy_s3_object(self, source_key: str, dest_key: str):
        """Copy an object within the bucket."""
        if self.s3_client and self.s3_bucket:
            copy_source = {'Bucket': self.s3_bucket, 'Key': source_key}
            self.s3_client.copy_object(CopySource=copy_source, Bucket=self.s3_bucket, Key=dest_key)

    def move_s3_object(self, source_key: str, dest_key: str):
        """Move an object (Copy + Delete)."""
        print(f"Moving S3 object locally: {source_key} -> {dest_key}")
        self._copy_s3_object(source_key, dest_key)
        self._delete_from_s3(source_key)

    def create_appointment(self, req: PatientInformation) -> EncounterState:
        """Create a new booked appointment."""
        state = EncounterState(
            patient_information=req,
            status=AppointmentStatus.BOOKED
        )
        self.save_state(state)
        return state

    def list_appointments(self) -> List[EncounterState]:
        """List all appointments across all providers by scanning for clinical_data_jsons/*.json"""
        # 1. Sync list from S3 if available
        if self.s3_client and self.s3_bucket:
            try:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                # We scan the root prefix to find all provider folders
                for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=f"{self.s3_prefix}/"):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        # Look for JSON files in a 'clinical_data_jsons' folder
                        if key.endswith(".json") and "/clinical_data_jsons/" in key:
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

    def load_state(self, appointment_id: str, allow_uuid: bool = True, provider_id: str = None) -> EncounterState:
        """Load state from local disk or S3. Supports UUID lookup and Provider-specific paths."""
        local_path = self._get_path(appointment_id)
        
        # 1. Try exact local match
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return EncounterState.model_validate(data)

        # 2. Try UUID-based lookup locally
        if allow_uuid:
            uuid_match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', appointment_id, re.IGNORECASE)
            if uuid_match:
                uuid = uuid_match.group(1).lower()
                for f_name in os.listdir(self.storage_dir):
                    if f_name.endswith(".json") and uuid in f_name.lower():
                        with open(os.path.join(self.storage_dir, f_name), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            return EncounterState.model_validate(data)

        # 3. Try S3 lookup
        if self.s3_client and self.s3_bucket:
            if provider_id:
                # Direct lookup if provider is known
                s3_key = f"{self.s3_prefix}/{provider_id}/clinical_data_jsons/{appointment_id}.json"
                if self._download_from_s3(s3_key, local_path):
                     with open(local_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return EncounterState.model_validate(data)
            else:
                # SEARCH fallback: This happens if app.py calls without provider context
                print(f"   [Debug] Searching S3 for state: {appointment_id} across all providers...")
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=f"{self.s3_prefix}/"):
                    for obj in page.get('Contents', []):
                        if obj['Key'].endswith(f"/clinical_data_jsons/{appointment_id}.json"):
                            if self._download_from_s3(obj['Key'], local_path):
                                with open(local_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    return EncounterState.model_validate(data)

        raise FileNotFoundError(f"Appointment {appointment_id} (or UUID equivalent) not found.")

    async def create_from_transcript(self, transcript: str, appointment_id: str, provider_id: str = "default", original_filename: str = None, pre_patient_info: PatientInformation = None, pre_patient_context: str = "") -> EncounterState:
        """Parse transcript via LLM, create/update EncounterState, save and upload chart."""
        print(f"   -> Parsing transcript for {appointment_id}...")
        parsed = await self.parser.parse_transcript(transcript)
        
        if "error" in parsed:
            raise ValueError(f"LLM parsing failed: {parsed['error']}")
        
        # Try to load existing EXACT state, or create new one
        try:
            state = self.load_state(appointment_id, allow_uuid=False)
            print(f"   -> Loaded existing state (v{state.version}) for {appointment_id}")
            state.version += 1
            state.updated_at = datetime.now()
            # Force update provider_id so S3 uploads go to the right folder
            state.provider_id = provider_id
        except FileNotFoundError:
            state = EncounterState(appointment_id=appointment_id, provider_id=provider_id)
            print(f"   -> Creating new state for {appointment_id}")
        
        if original_filename:
            state.original_audio_filename = original_filename
            
        # Step 1: Pre-populate patient info from Scriberyte DB (authoritative source)
        scriberyte_info, patient_context = pre_patient_info, pre_patient_context
        
        if not scriberyte_info and self.scriberyte.is_configured():
            scriberyte_info, patient_context = self.scriberyte.fetch_patient_info(appointment_id)
        
        if scriberyte_info:
            state.patient_information = scriberyte_info
            print("   -> Patient info pre-populated from Scriberyte DB")
        elif state.patient_information and state.patient_information.patient_name:
            # Fallback: If DB fetch failed for this specific ID (common for addendums),
            # but we loaded an existing state via UUID, REUSE that info.
            print("   -> Reusing patient info from existing state (linked via UUID)")
        else:
            print("   -> No patient info found in DB or existing state.")
        
        # Prepend patient context sentence to transcript for LLM
        if patient_context:
            transcript = f"{patient_context}\n\n{transcript}"
        
        # Step 2: Overlay any additional info extracted from transcript (fills gaps only)
        patient_info = parsed.get("patient_information", {})
        if patient_info:
            for field, value in patient_info.items():
                if value and hasattr(state.patient_information, field):
                    # Only set if Scriberyte didn't already provide this field
                    existing = getattr(state.patient_information, field, None)
                    if not existing:
                        setattr(state.patient_information, field, value)
        
        # Update wounds
        wounds_data = parsed.get("wounds", [])
        if wounds_data:
            state.wounds = [WoundDetails(**w) for w in wounds_data]
        
        # Update other fields
        # Note: LLM prompt returns "comments" key, not "provider_comments"
        comments = parsed.get("comments") or parsed.get("provider_comments")
        if comments and comments != "-":
            state.provider_comments = comments
        treatment = parsed.get("treatment_plan")
        if treatment and treatment != "-":
            state.treatment_plan = treatment
        
        em_justif = parsed.get("em_justification")
        if em_justif:
            state.em_justification = EMJustification(**em_justif)
        
        # Append to history
        state.history.append({
            "version": state.version,
            "type": "initial_dictation",
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript,
            "snapshot": {
                "patient_info": state.patient_information.model_dump(),
                "wounds": [w.model_dump() for w in state.wounds],
                "provider_comments": state.provider_comments,
                "treatment_plan": state.treatment_plan
            }
        })
        
        state.status = AppointmentStatus.RECORDING_SAVED
        self.save_state(state)
        print(f"   -> Chart saved for {appointment_id} (v{state.version})")
        return state

    async def apply_addendum(self, appointment_id: str, transcript: str, provider_id: str = "default", original_filename: str = None) -> EncounterState:
        """Apply an addendum transcript as a patch to existing state."""
        print(f"   -> Applying addendum for {appointment_id}...")
        state = self.load_state(appointment_id, allow_uuid=True, provider_id=provider_id)
        
        # Generate patch using LLM
        patch_ops = await self.parser.generate_patch(
            state.model_dump(), transcript
        )
        
        if patch_ops:
            try:
                state_dict = json.loads(state.model_dump_json())
                patched = jsonpatch.apply_patch(state_dict, patch_ops)
                state = EncounterState.model_validate(patched)
                print(f"   -> Patch applied successfully ({len(patch_ops)} ops)")
            except Exception as e:
                print(f"   -> Patch failed ({e}), appending as comment instead")
                state.provider_comments = (state.provider_comments or "") + f"\n[Addendum]: {transcript}"
        else:
            print("   -> No patch ops generated, appending transcript as comment")
            state.provider_comments = (state.provider_comments or "") + f"\n[Addendum]: {transcript}"
        
        # Increment version
        state.version += 1
        if original_filename:
            state.original_audio_filename = original_filename
            
        # Append to history
        state.history.append({
            "version": state.version,
            "type": "addendum",
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript,
            "patch_ops": patch_ops,
            "snapshot": {
                "patient_info": state.patient_information.model_dump(),
                "wounds": [w.model_dump() for w in state.wounds],
                "provider_comments": state.provider_comments,
                "treatment_plan": state.treatment_plan
            }
        })
        
        self.save_state(state)
        print(f"   -> Addendum chart saved for {appointment_id} (v{state.version})")
        return state

    def delete_appointment(self, appointment_id: str):
        """Delete appointment from local storage."""
        local_path = self._get_path(appointment_id)
        if os.path.exists(local_path):
            os.remove(local_path)
        html_path = os.path.join(self.storage_dir, f"{appointment_id}.html")
        if os.path.exists(html_path):
            os.remove(html_path)

    def save_state(self, state: EncounterState):
        state.updated_at = datetime.now()
        
        # 1. Save JSON locally (Always keep for state management)
        json_path = self._get_path(state.appointment_id)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
        
        # 2. Generate HTML report locally (with retry for file locks)
        html_path = os.path.join(self.storage_dir, f"{state.appointment_id}.html")
        
        for i in range(3):
            try:
                self.html_gen.generate(state, html_path)
                break
            except Exception as e:
                if i == 2:
                    print(f"   [Error] Final save failed: {e}")
                else: 
                    print(f"   [Warning] Save attempt {i+1} failed ({e}), retrying...")
                    time.sleep(1)
        
        # 3. Upload artifacts to S3 with Provider Folder
        # Naming convention: same base name as original audio, only extension changes
        #   {base_name}.html  (chart)      — in chatgpt_htmls/{appointment_id}/
        #   {base_name}.json  (clinical data) — in clinical_data_jsons/
        #   {base_name}.txt   (transcript)  — uploaded in process_s3_audio_to_state
        #   {base_name}.mp3   (audio)       — original from Scriberyte
        # No history/ subfolder needed: version is embedded in filename (chart-1, addendum-1)
        
        provider_id = getattr(state, "provider_id", "default")
        base_chart_path = f"{self.s3_prefix}/{provider_id}/chatgpt_htmls"
        
        # Determine output name: strip original audio extension, replace with .html/.json
        if state.original_audio_filename:
            base_name = get_output_basename(state.original_audio_filename)
        else:
            base_name = state.appointment_id

        # Upload HTML chart
        s3_html_key = f"{base_chart_path}/{base_name}.html"
        self._upload_to_s3(html_path, s3_html_key)

        # Upload Clinical data JSON — same base name, .json extension
        s3_json_key = f"{self.s3_prefix}/{provider_id}/clinical_data_jsons/{base_name}.json"
        self._upload_to_s3(json_path, s3_json_key)


    async def process_audio_to_state(self, audio_path: str, appointment_id: str, provider_id: str = "default") -> EncounterState:
        """Helper: Audio -> Transcript -> State with S3 storage (Conditional on Success)."""
        # 1. Transcribe (Wait for success)
        transcript = await self.transcriber.transcribe(audio_path)
        
        # 2. Only if transcription succeeded, upload Audio and Transcript to S3
        #    Naming: same base name as audio, different extension
        fname = os.path.basename(audio_path)
        base_name = get_output_basename(fname)

        self._upload_to_s3(audio_path, f"{self.s3_prefix}/{provider_id}/audio/{fname}")
        transcript_path = os.path.join(self.storage_dir, f"{appointment_id}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        # Transcript: {base_name}.txt
        self._upload_to_s3(transcript_path, f"{self.s3_prefix}/{provider_id}/transcribed-speaker-label/{base_name}.txt")
        
        # 3. Process to chart
        return await self.create_from_transcript(transcript, appointment_id, provider_id=provider_id, original_filename=fname)

    async def process_audio_addendum_to_state(self, audio_path: str, appointment_id: str, provider_id: str = "default") -> EncounterState:
        """Helper: Addendum Audio -> Transcript -> Patch with S3 archival (Conditional)."""
        # 1. Transcribe
        transcript = await self.transcriber.transcribe(audio_path)
        
        # 2. Upload artifacts — same base name convention
        fname = os.path.basename(audio_path)
        base_name = get_output_basename(fname)

        self._upload_to_s3(audio_path, f"{self.s3_prefix}/{provider_id}/audio/{fname}")
        transcript_path = os.path.join(self.storage_dir, f"{base_name}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # Transcript: {base_name}.txt
        self._upload_to_s3(transcript_path, f"{self.s3_prefix}/{provider_id}/transcribed-speaker-label/{base_name}.txt")
        
        # 3. Apply addendum
        return await self.apply_addendum(appointment_id, transcript, provider_id=provider_id, original_filename=fname)

    async def process_s3_audio_to_state(self, s3_key: str, appointment_id: str, provider_id: str = "default") -> EncounterState:
        """Downloads audio from S3, processes it, and updates state + artifacts."""
        # 1. Download audio from S3 to temp local
        ext = os.path.splitext(s3_key)[1] or ".wav"
        temp_audio_path = os.path.join(self.storage_dir, f"temp_{appointment_id}{ext}")
        
        if not self._download_from_s3(s3_key, temp_audio_path):
            raise FileNotFoundError(f"S3 Key {s3_key} not found.")

        try:
            # 2. Transcribe
            transcript = await self.transcriber.transcribe(temp_audio_path)
            
            # 3. Save & Upload Transcript — same base name as audio, .txt extension
            fname = os.path.basename(s3_key)
            base_name = get_output_basename(fname)

            transcript_path = os.path.join(self.storage_dir, f"{base_name}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            # Transcript: {base_name}.txt
            self._upload_to_s3(transcript_path, f"{self.s3_prefix}/{provider_id}/transcribed-speaker-label/{base_name}.txt")
            
            # 4. Copy Audio to Provider Folder (preserves original name)
            dest_audio_key = f"{self.s3_prefix}/{provider_id}/audio/{fname}"
            if s3_key != dest_audio_key:
                 self._copy_s3_object(s3_key, dest_audio_key)
            
            # 5. Process (Create Chart) — save_state will upload .html and .json with same base name
            state = await self.create_from_transcript(transcript, appointment_id, provider_id=provider_id, original_filename=fname)
            
            # 6. ONLY delete from inbox if the chart was successfully generated
            if s3_key != dest_audio_key and self.s3_client:
                 try:
                     self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                     print(f"   -> Moved audio to {dest_audio_key}")
                 except Exception as e:
                     print(f"   [Warning] Failed to delete inbox audio {s3_key}: {e}")

            return state
            
        finally:
            # Cleanup temp audio
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    async def process_s3_addendum_to_state(self, s3_key: str, appointment_id: str, provider_id: str = "default") -> EncounterState:
        """Downloads addendum audio from S3, processes it, and patches state."""
        # 1. Download audio
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") # Unique timestamp
        original_ext = os.path.splitext(s3_key)[1] or ".wav"
        
        # Temp local file
        temp_audio_path = os.path.join(self.storage_dir, f"temp_{appointment_id}_add_{ts}{original_ext}")
        
        if not self._download_from_s3(s3_key, temp_audio_path):
            raise FileNotFoundError(f"S3 Key {s3_key} not found.")
            
        try:
            # 2. Transcribe
            transcript = await self.transcriber.transcribe(temp_audio_path)
            
            # 3. Save & Upload Transcript — same base name as audio, .txt extension
            fname = os.path.basename(s3_key)
            base_name = get_output_basename(fname)

            transcript_path = os.path.join(self.storage_dir, f"{base_name}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            # Transcript: {base_name}.txt
            self._upload_to_s3(transcript_path, f"{self.s3_prefix}/{provider_id}/transcribed-speaker-label/{base_name}.txt")
            
            # 4. Copy Audio to provider folder (preserves original name with -addendum- marker)
            dest_audio_key = f"{self.s3_prefix}/{provider_id}/audio/{fname}"
            self._upload_to_s3(temp_audio_path, dest_audio_key)

            # 5. Apply Addendum — save_state will upload .html and .json with same base name
            state = await self.apply_addendum(appointment_id, transcript, provider_id=provider_id, original_filename=fname)

            # 6. Delete original from inbox ONLY on success
            if self.s3_client:
                try:
                    self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                    print(f"   -> Moved addendum audio to {dest_audio_key}")
                except Exception as e:
                    print(f"   [Warning] Failed to delete inbox audio {s3_key}: {e}")
            
            return state
            
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
