"""
WoundCare AI - Clinical LLM Parser (src/parser.py)
----------------------------------------------------
Interfaces with the Google Gemini API to extract structured clinical data
from provider dictation transcripts.

Key Functions:
  parse_transcript(transcript)
    - Sends transcript to Gemini with INTENT_EXTRACTION_PROMPT
    - Returns structured JSON: patient_information, wounds[], treatment_plan, comments

  generate_patch(existing_state, addendum_transcript)
    - Sends existing state + addendum transcript to Gemini with ADDENDUM_PATCH_PROMPT
    - Returns a JSON Patch (RFC 6902) list of operations to update the state

Model Strategy:
  Primary : gemini-3-pro-preview  (Gemini 3 Pro — confirmed available)
  Fallback1: gemini-2.5-pro
  Fallback2: gemini-2.0-flash
  Automatically retries with next model on failure.
"""
import json
import os
from typing import Dict, Any, List, Optional
from google import genai
from dotenv import load_dotenv

load_dotenv()

from .prompts import INTENT_EXTRACTION_PROMPT, ADDENDUM_PATCH_PROMPT  # noqa: E402
from .abbreviations import get_abbreviation_markdown  # noqa: E402
from .utils import clean_narrative_text  # noqa: E402

class ClinicalParser:
    """Uses LLM to extract structured clinical intent from transcripts using google-genai SDK."""
    
    def __init__(self, model_name: str = "gemini-3-pro-preview"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("WARNING: GOOGLE_API_KEY not found.")
            
        self.client = genai.Client(api_key=self.api_key)
        
        # Fallback chain: Gemini 3 Pro -> Gemini 2.5 Pro -> Gemini 2.0 Flash
        self.model_names = [model_name, "gemini-2.5-pro", "gemini-2.0-flash"]
        self.current_model_idx = 0

    async def _generate_with_retry(self, prompt: str):
        """Try multiple models if one fails."""
        while self.current_model_idx < len(self.model_names):
            model_id = self.model_names[self.current_model_idx]
            try:
                # New SDK Syntax: client.aio.models.generate_content
                response = await self.client.aio.models.generate_content(
                    model=model_id,
                    contents=prompt
                )
                return response
            except Exception as e:
                print(f"Model {model_id} failed: {e}")
                self.current_model_idx += 1
                if self.current_model_idx >= len(self.model_names):
                    raise e
        return None

    def _post_process_json(self, data: Any, key: Optional[str] = None) -> Any:
        """Deeply normalize units like 'centimeter' to 'cm' and clean separators."""
        if isinstance(data, dict):
            return {k: self._post_process_json(v, k) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._post_process_json(i, key) for i in data]
        elif isinstance(data, str):
            res = data
            
            # 1. Global Unit Normalization (Safe for all fields)
            global_norm = {
                "centimeters": "cm",
                "centimeter": "cm",
                "square centimeters": "sq cm",
                "cubic centimeters": "cm³",
                " %": "%",
                " .": ".",
            }
            
            for old, new in global_norm.items():
                if old in res.lower():
                    import re
                    res = re.sub(re.escape(old), new, res, flags=re.IGNORECASE)

            # 2. Field-Specific Normalization (ONLY for measurements/structured fields)
            measurement_fields = ["measurements", "tunnels", "max_depth", "undermining", "area_sq_cm", "volume_cu_cm"]
            if key in measurement_fields:
                struct_norm = {
                    ".point": ".",
                    " point ": ".",
                    " by ": " x ",
                    ";": ".",
                    # Only replace period with x if it looks like a measurement (digit period space digit)
                }
                for old, new in struct_norm.items():
                    if old in res.lower():
                        import re
                        res = re.sub(re.escape(old), new, res, flags=re.IGNORECASE)
                
                # Fix dimensions: "4. 3. 1" -> "4 x 3 x 1"
                import re
                res = re.sub(r'(\d+)\.\s+(\d+)', r'\1 x \2', res)
            
            # 3. Narrative Cleanup (Fix unintended "x" separators and common mis-hears)
            narrative_fields = ["clinical_summary", "treatment_plan", "comments"]
            
            if key in narrative_fields or key is None:
                # A. Apply Abbreviation Expansion (Safety Net)
                import re
                from .abbreviations import ABBREVIATION_STORE
                for category, items in ABBREVIATION_STORE.items():
                    for short_code, full_term in items.items():
                        # STRICT MATCHING: Word boundary only.
                        # This prevents "TEST3" matching "ST3"
                        # But allows "D" to match " D " (as a standalone letter/symbol)
                        pattern = r'\b' + re.escape(short_code) + r'\b'
                        
                        # Only replace if the full term isn't already immediately there.
                        # This crude check helps prevent "Stage 3 Pressure Injury Pressure Injury"
                        # but follows the user rule: "convert short form to long form"
                        if re.search(pattern, res, re.IGNORECASE):
                             if full_term.lower() not in res.lower():
                                res = re.sub(pattern, full_term, res, flags=re.IGNORECASE)

                res = clean_narrative_text(res)
            
            return res.strip()
        return data

    async def parse_transcript(self, transcript: str) -> Dict[str, Any]:
        """Initial parsing of a full transcript."""
        self.current_model_idx = 0  # Reset to primary model each call
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
            
            parsed = json.loads(text)
            return self._post_process_json(parsed)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {"error": "Failed to parse transcript"}

    async def generate_patch(self, existing_state: Dict[str, Any], addendum_transcript: str) -> List[Dict[str, Any]]:
        """Generate JSON Patch operations for an addendum."""
        self.current_model_idx = 0  # Reset to primary model each call
        # Strip history and other metadata to reduce payload size and speed up LLM reasoning
        minimized_state = {
            "patient_info": existing_state.get("patient_information", {}),
            "wounds": existing_state.get("wounds", []),
            "comments": existing_state.get("provider_comments", ""),
            "treatment_plan": existing_state.get("treatment_plan", ""),
            "em_justification": existing_state.get("em_justification", {})
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
            parsed = json.loads(text)
            return self._post_process_json(parsed)
        except Exception as e:
            print(f"Error parsing patch response: {e}")
            return []
