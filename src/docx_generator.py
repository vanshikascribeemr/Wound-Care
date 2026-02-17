from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from .models import EncounterState
import os

class DocxGenerator:
    """Generates standardized Wound Care DOCX reports."""

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

    def generate(self, state: EncounterState, output_path: str):
        doc = Document()
        
        # Title
        title = doc.add_heading('WOUND CARE VISIT REPORT', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Patient Info Section
        doc.add_heading('Patient Information', level=1)
        pi = state.patient_information
        table = doc.add_table(rows=3, cols=2)
        table.style = 'Table Grid'
        
        cells = table.rows[0].cells
        cells[0].text = f"Patient Name: {pi.patient_name or '—'}"
        cells[1].text = f"DOB: {pi.dob or '—'}"
        
        cells = table.rows[1].cells
        cells[0].text = f"Date of Service: {pi.date_of_service or '—'}"
        cells[1].text = f"Physician: {pi.physician or '—'}"
        
        cells = table.rows[2].cells
        cells[0].text = f"Facility: {pi.facility or '—'}"
        cells[1].text = f"Version: {state.version}"

        doc.add_paragraph() # Spacer

        # Wound Table
        doc.add_heading('Wound Assessment Table', level=1)
        num_wounds = len(state.wounds)
        if num_wounds > 0:
            w_table = doc.add_table(rows=len(self.ATTRIBUTES) + 1, cols=num_wounds + 1)
            w_table.style = 'Table Grid'
            
            # Header Row
            hdr_cells = w_table.rows[0].cells
            hdr_cells[0].text = 'Attribute'
            for i, w in enumerate(state.wounds):
                hdr_cells[i+1].text = f"Wound #{w.number}"
                hdr_cells[i+1].paragraphs[0].runs[0].bold = True

            # Data Rows
            for row_idx, (attr_id, label) in enumerate(self.ATTRIBUTES):
                row = w_table.rows[row_idx + 1].cells
                row[0].text = label
                row[0].paragraphs[0].runs[0].bold = True
                for col_idx, w in enumerate(state.wounds):
                    val = getattr(w, attr_id, None) or w.attributes.get(attr_id, "-")
                    row[col_idx + 1].text = str(val if val is not None else "-")
        else:
            doc.add_paragraph("No wound data recorded.")

        doc.add_paragraph()

        # Summaries
        doc.add_heading('Detailed Visit Summaries', level=1)
        for w in state.wounds:
            p = doc.add_paragraph()
            # Professional Header: Wound #1 - Type - Location:
            header = f"Wound #{w.number} - {w.type or 'Wound'} - {w.location or 'Unspecified Location'}:"
            run = p.add_run(header)
            run.bold = True
            
            # Narrative content
            narrative = w.clinical_summary if w.clinical_summary and w.clinical_summary != "-" else "No detailed clinical summary recorded."
            p.add_run(f"\nSummary: {narrative}")
            
            if w.treatment_plan and w.treatment_plan != "-":
                run = p.add_run(f"\nTreatment Plan: ")
                run.bold = True
                p.add_run(f"{w.treatment_plan}")
            
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

        doc.add_heading('Provider Comments', level=2)
        doc.add_paragraph(state.provider_comments or "No comments.")

        doc.add_heading('Clinical Plan', level=2)
        doc.add_paragraph(state.treatment_plan or "No plan.")

        # Footer
        footer = doc.sections[0].footer
        footer.paragraphs[0].text = f"Generated: {state.updated_at.strftime('%Y-%m-%d %H:%M:%S')} | ID: {state.encounter_id}"

        doc.save(output_path)
        return output_path
