"""
WoundCare AI - S3 Watcher Service (src/watcher.py)
----------------------------------------------------
Background polling service that monitors the S3 inbox folder for new audio
files and automatically triggers the processing pipeline.

How it works:
  1. Polls input S3 bucket prefix every N seconds
  2. Detects new .wav / .mp3 / .m4a / .ogg files
  3. Extracts appointment_id from the filename
     - "P1005_2026-02-18.wav"         → New visit for P1005_2026-02-18
     - "P1005_2026-02-18_add.wav"     → Addendum for P1005_2026-02-18
  4. Calls EncounterManager to transcribe, parse, and generate chart
  5. Deletes/Archives the source file from S3 after successful processing


Duplicate Protection:
  - In-memory cache (5-minute expiry) prevents re-processing the same file
    if the watcher polls before S3 deletion propagates.

Run with:
  python -m src.run_watcher
"""
import asyncio
import os
from datetime import datetime
from .manager import EncounterManager

class S3Watcher:
    """
    Background service that monitors an S3 'inbox' folder for new audio dictations.
    When a file is found:
      1. Determines if it's a new encounter or an addendum based on filename.
      2. Calls the appropriate processing pipeline.
      3. Moves the processed file to an 'archive' folder to prevent re-processing.
    """
    
    def __init__(self, input_prefix=None, loop_interval=15):
        self.manager = EncounterManager()
        # Watch the broad 'woundcare' prefix so we catch woundcare/{provider}/split-audio-files
        self.input_prefix = os.getenv("S3_INPUT_PREFIX", self.manager.s3_prefix)
        self.loop_interval = loop_interval
        self.is_running = False
        self.processed_cache = {} # Key -> Timestamp of processing
        from datetime import timedelta
        self.watcher_start_time = datetime.utcnow() - timedelta(minutes=15)
        print(f"Watcher will only process files uploaded after: {self.watcher_start_time} UTC")

    async def start(self):
        print(f"--> S3 Watcher Started (PID: {os.getpid()}). Monitoring s3://{self.manager.s3_bucket}/{self.input_prefix} every {self.loop_interval}s")
        self.is_running = True
        while self.is_running:
            await self.scan_and_process()
            await asyncio.sleep(self.loop_interval)

    async def scan_and_process(self):
        if not self.manager.s3_client:
            print("Warning: S3 Client not initialized. Check .env config.")
            return

        try:
            # Clean old cache entries (> 5 mins)
            now = datetime.now().timestamp()
            self.processed_cache = {k: v for k, v in self.processed_cache.items() if now - v < 300}

            pending_tasks = []

            # List objects in inbox
            paginator = self.manager.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.manager.s3_bucket, Prefix=self.input_prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Skip directory placeholders
                    if key.endswith("/"):
                        continue
                        
                    # 2. Process only audio files in 'split-audio-files' folders
                    if "split-audio-files" in key and key.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg', '.mp4')):
                         last_modified = obj['LastModified'].replace(tzinfo=None)
                         # Log everything found
                         print(f"   [Debug] Scanning file: {os.path.basename(key)} (Modified: {last_modified})")

                         # SKIP if older than watcher start time (only future audios)
                         if last_modified < self.watcher_start_time:
                             # print(f"   [Debug] Skipping existing file: {os.path.basename(key)}")
                             continue
                         
                         print(f"   [Debug] Found NEW file: {os.path.basename(key)} (Modified: {last_modified})")
                         # SKIP if recently processed
                         if key in self.processed_cache:
                             print(f"   [Debug] Skipping recently processed: {os.path.basename(key)}")
                             continue
                             
                         # Remove sequential await and build a task list instead
                         pending_tasks.append(self.process_and_cache(key))
            
            # Run all tasks found in this scan simultaneously in parallel!
            if pending_tasks:
                await asyncio.gather(*pending_tasks)
                         
        except Exception as e:
            print(f"Watcher Scan Error: {e}")

    async def process_and_cache(self, key):
        """Wrapper to handle caching after parallel processing."""
        success = await self.process_file(key)
        if success:
            self.processed_cache[key] = datetime.now().timestamp()
        else:
            print(f"   [Retry Queued] Will retry {os.path.basename(key)} on next loop.")

    async def process_file(self, s3_key: str):
        filename = os.path.basename(s3_key)
        print(f"\nProcessing detected file: {s3_key}")
        
        # --- 1. Parse filename components ---
        # Format: {timestamp}-{visitID}-{providerUUID}-{chart|addendum}-{version}.mp3
        from .utils import parse_audio_filename
        parsed = parse_audio_filename(filename)
        
        is_addendum = parsed['is_addendum']
        appointment_id = parsed['appointment_id']  # timestamp-visitID-providerUUID

        print(f"   [Debug] Extracted Appointment ID: {appointment_id}")

        # --- 2. Extract Provider ID ---
        # Primary: from S3 path (folder before 'split-audio-files')
        # Fallback: second UUID in filename (providerUUID)
        parts = s3_key.split('/')
        if 'split-audio-files' in parts:
            idx = parts.index('split-audio-files')
            provider_id = parts[idx-1] if idx > 0 else "default"
        elif parsed['provider_uuid']:
            provider_id = parsed['provider_uuid']
        else:
            provider_id = "default"

        visit_id = parsed['visit_id']
        
        print(f"   [Debug] Provider ID: {provider_id} | Visit ID: {visit_id}")
        
        # --- 4. Trigger Processing ---
        try:
            if is_addendum:
                print(f"   -> Detected Addendum for ID: {appointment_id} (Provider: {provider_id})")
                await self.manager.process_s3_addendum_to_state(s3_key, appointment_id, provider_id=provider_id)
            else:
                # Removed strict chart-exists check so users can re-process test files.
                # The in-memory cache still prevents infinite loops.
                print(f"   -> Detected New/Initial Dictation for ID: {appointment_id} (Provider: {provider_id})")
                await self.manager.process_s3_audio_to_state(s3_key, appointment_id, provider_id=provider_id)
                
            print(f"   -> Processing Complete for {filename}")
            return True
            
        except Exception as e:
            print(f"   [Error] Failed to process {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False
