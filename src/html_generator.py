"""
WoundCare AI - HTML Chart Generator (src/html_generator.py)
-------------------------------------------------------------
Generates professional, responsive Wound Care visit report HTML files 
from a structured EncounterState object.

Output sections:
  1. Patient Information
  2. Wound Assessment Table
  3. Detailed Visit Summaries
  4. Provider Comments
  5. Clinical Plan
  6. Footer
"""
from .models import EncounterState
import json
from scriberyte.util import json_to_html_with_sections_for_wound_care

class HtmlGenerator:
    """Generates standardized Wound Care HTML reports using scriberyte.util."""

    def generate(self, state: EncounterState, output_path: str):
        # Convert state to dict for util function
        # We need to ensure the keys match what the LLM generates
        state_dict = json.loads(state.model_dump_json())
        
        # Map provider_comments internal key to 'comments' expected by util.py
        if 'provider_comments' in state_dict:
            state_dict['comments'] = state_dict.pop('provider_comments')
            
        # Generate HTML
        html, _ = json_to_html_with_sections_for_wound_care(state_dict)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        return output_path
