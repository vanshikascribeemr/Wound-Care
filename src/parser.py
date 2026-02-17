import json
import os
from typing import Dict, Any, List
from .models import EncounterState, WoundDetails, PatientInformation
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

from .prompts import INTENT_EXTRACTION_PROMPT, ADDENDUM_PATCH_PROMPT
from .abbreviations import get_abbreviation_markdown

class ClinicalParser:
    """Uses LLM to extract structured clinical intent from transcripts."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_names = [model_name, "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro"]
        self.current_model_idx = 0
        self._init_model()

    def _init_model(self):
        name = self.model_names[self.current_model_idx]
        print(f"Initializing ClinicalParser with model: {name}")
        self.model = genai.GenerativeModel(name)

    async def _generate_with_retry(self, prompt: str):
        """Try multiple models if one fails."""
        while self.current_model_idx < len(self.model_names):
            try:
                return await self.model.generate_content_async(prompt)
            except Exception as e:
                print(f"Model {self.model_names[self.current_model_idx]} failed: {e}")
                self.current_model_idx += 1
                if self.current_model_idx < len(self.model_names):
                    self._init_model()
                else:
                    raise e
        return None

    async def parse_transcript(self, transcript: str) -> Dict[str, Any]:
        """Initial parsing of a full transcript."""
        abbrev_list = get_abbreviation_markdown()
        prompt = INTENT_EXTRACTION_PROMPT.format(
            transcript=transcript, 
            abbreviations_list=abbrev_list
        )
        response = await self._generate_with_retry(prompt)
        try:
            # Clean response text if it has markdown blocks
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {"error": "Failed to parse transcript"}

    async def generate_patch(self, existing_state: Dict[str, Any], addendum_transcript: str) -> List[Dict[str, Any]]:
        """Generate JSON Patch operations for an addendum."""
        # Strip history and other metadata to reduce payload size and speed up LLM reasoning
        minimized_state = {
            "patient_info": existing_state.get("patient_information", {}),
            "wounds": existing_state.get("wounds", []),
            "comments": existing_state.get("provider_comments", "")
        }
        
        abbrev_list = get_abbreviation_markdown()
        prompt = ADDENDUM_PATCH_PROMPT.format(
            existing_json=json.dumps(minimized_state, indent=2),
            addendum_transcript=addendum_transcript,
            abbreviations_list=abbrev_list
        )
        response = await self._generate_with_retry(prompt)
        try:
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            return json.loads(text)
        except Exception as e:
            print(f"Error parsing patch response: {e}")
            return []
