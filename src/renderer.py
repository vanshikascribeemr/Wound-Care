import json
from typing import Any, List, Dict, Optional
from jinja2 import Environment, FileSystemLoader
from .models import EncounterState, WoundDetails

class NoteRenderer:
    """Renders EncounterState into the standardized Wound Care format."""
    
    ATTRIBUTES = [
        ("mist_therapy", "MIST Therapy"),
        ("location", "Wound Location"),
        ("outcome", "Outcome"),
        ("type", "Wound Type"),
        ("status", "Wound Status"),
        ("measurements", "Measurements L x W x D"),
        ("area_sq_cm", "Area (sq cm)"),
        ("volume_cu_cm", "Volume (cm³)"),
        ("tunnels", "Tunnels"),
        ("max_depth", "Max depth of deepest tunnel (cm)"),
        ("undermining", "Undermining (cm)"),
        ("stage_grade", "Stage or grade if applicable"),
        ("drainage", "Exudate Amount"),
        ("exudate_type", "Exudate Type"),
        ("odor", "Odor"),
        ("wound_margin", "Wound Margin"),
        ("periwound", "Periwound"),
        ("necrotic_material", "Necrotic Material"),
        ("granulation", "Granulation"),
        ("tissue_exposed", "Tissue Exposed"),
        ("procedure", "Debridement"),
    ]

    def __init__(self, template_dir: str = "templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def render_html(self, state: EncounterState, versions: list = None) -> str:
        """Render the encounter state to HTML."""
        # 1. Prepare Assessment Table Data (Transpose wounds to attribute rows)
        wound_nums = [w.number for w in state.wounds]
        rows = []
        for attr_id, label in self.ATTRIBUTES:
            row_values = []
            for w in state.wounds:
                val = getattr(w, attr_id, None)
                if val is None:
                    val = w.attributes.get(attr_id, "-")
                row_values.append(val if val is not None else "-")
            rows.append({"label": label, "vals": row_values})

        # 2. Prepare Detailed Wound Summaries
        wound_summaries = []
        for w in state.wounds:
            # Use the LLM-generated clinical_summary if it exists
            if w.clinical_summary and w.clinical_summary != "-":
                summary_text = w.clinical_summary
            else:
                # Fallback: Generate one from attributes if narrative is missing
                attr_details = []
                for attr_id, label in self.ATTRIBUTES:
                    if attr_id in ["number", "type", "location", "clinical_summary"]: continue 
                    val = getattr(w, attr_id, None)
                    if val is None:
                        val = w.attributes.get(attr_id)
                    
                    if val and val != "-" and val != "":
                        attr_details.append(f"<strong>{label}:</strong> {val}")
                
                summary_text = "; ".join(attr_details) if attr_details else "-"
            
            wound_summaries.append({
                "number": w.number,
                "type": w.type or "Unspecified Type",
                "location": w.location or "Unspecified Location",
                "summary": summary_text,
                "treatment_plan": w.treatment_plan if w.treatment_plan and w.treatment_plan != "-" else None
            })

        template = self.env.get_template("visit_report.html")
        return template.render(
            state=state, 
            rows=rows, 
            wound_nums=wound_nums, 
            wound_summaries=wound_summaries,
            versions=versions or []
        )
