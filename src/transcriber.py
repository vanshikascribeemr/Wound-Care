"""
WoundCare AI - Audio Transcriber (src/transcriber.py)
-------------------------------------------------------
Converts provider dictation audio files into text transcripts using
OpenAI's Whisper speech-to-text model.

Supports audio formats: .wav, .mp3, .m4a, .ogg, .mp4
Requires: OPENAI_API_KEY in .env
"""
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class Transcriber:
    """Handles Voice-to-Text conversion using OpenAI Whisper with Async support."""
    
    def __init__(self):
        # Lazy initialization to prevent startup crash if key creates issue
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None

    def _get_client(self):
        if not self.client:
            if not self.api_key:
                # Retrying fetch in case env var was loaded late
                self.api_key = os.getenv("OPENAI_API_KEY")
                
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set. Cannot transcribe audio.")
                
            self.client = AsyncOpenAI(api_key=self.api_key)
        return self.client

    async def transcribe(self, audio_file_path: str) -> str:
        """Converts audio file to text transcript (Async)."""
        client = self._get_client()
        
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        with open(audio_file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
            )
        return transcript
