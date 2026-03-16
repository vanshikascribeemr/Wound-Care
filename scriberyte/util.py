import re
import json
import copy

def json_to_html_with_sections_for_wound_care(json_data, schema_name=None):
    """
    Convert ChatGPT JSON output to:
    1. HTML string
    2. Section-wise dictionary
    """
    def is_numbered(item):
        return bool(re.match(r"^\s*\d+[\.\)]?\s", str(item).strip()))

    # Parse JSON if string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = copy.deepcopy(json_data)
    
    # Unwrap top-level key if it's the only key (sometimes LLM wraps output)
    top_level_keys = ["Multi Wound Chart Details", "Follow-Up - Wound Chart Details", "wound_care_visit_schema", "mist_documentation"]
    if isinstance(data, dict) and len(data) == 1:
        key = list(data.keys())[0]
        if key in top_level_keys:
            data = data[key]

    # Use the (potentially unwrapped) data for section-wise output as well
    orig_data_for_sections = data

    # Auto-detect schema if not provided
    if not schema_name:
        if any(k in data for k in ["Wound Entries", "Patient Wound Entries", "Multi Wound Chart Details"]):
            schema_name = "mist_documentation"
        elif any(k in data for k in ["Wound Assessment Table", "wounds", "Follow-Up - Wound Chart Details"]):
            schema_name = "wound_care_visit_schema"


    def render_special_chart(d):
        html = ""
        # 1. Patient Info Table
        if 'patient_information' in d or 'Patient Information' in d:
            pi = d.pop('patient_information', d.pop('Patient Information', {}))
            if not pi: pi = {} # Ensure pi is a dict not None
            p_name = pi.get("patient_name", pi.get("Patient Name", "______________________________"))
            if p_name == "-" or not p_name: p_name = "______________________________"
            
            html += f"<h1>Follow-Up - Wound Chart Details &ndash; {p_name}</h1>\n"
            formatted_pi = {
                "Patient Name": pi.get("patient_name", pi.get("Patient Name", "-")),
                "DOB": pi.get("dob", pi.get("DOB", "-")),
                "Date of Service": pi.get("date_of_service", pi.get("Date of Service", "-")),
                "Physician": pi.get("physician", pi.get("Physician", "-")),
                "Scribe": pi.get("scribe", pi.get("Transcriptionist", "-")),
                "Facility": pi.get("facility", pi.get("Facility", "-"))
            }
            cols = list(formatted_pi.items())
            html += "<h2 class='wc-h2'>Patient Information</h2>\n<table class='patient-info'>\n"
            for i in range(0, len(cols), 2):
                html += "  <tr>\n"
                html += f"    <th>{cols[i][0]}</th><td>{cols[i][1]}</td>\n"
                if i + 1 < len(cols): html += f"    <th>{cols[i+1][0]}</th><td>{cols[i+1][1]}</td>\n"
                else: html += "    <th></th><td></td>\n"
                html += "  </tr>\n"
            html += "</table>\n"
 
        # 2. Wound Assessment Table
        wounds_data = d.pop('wounds', [])
        wat = d.pop('Wound Assessment Table', {})
        if wat and 'headers' in wat and 'rows' in wat:
            html += "<h2 class='wc-h2'>Wound Assessment Table</h2>\n<table class='wound-table'>\n"
            html += "  <thead><tr><th>Field</th>" + "".join(f"<th>{h}</th>" for h in wat['headers'][1:]) + "</tr></thead>\n<tbody>\n"
            for row in wat['rows']:
                if row: html += f"  <tr><th>{row[0]}</th>" + "".join(f"<td>{v}</td>" for v in row[1:]) + "</tr>\n"
            html += "</tbody></table>\n"
        elif wounds_data:
            ATTRS = [
                ("MIST Therapy", "mist_therapy"),
                ("Wound Location", "location"),
                ("Outcome", "outcome"),
                ("Wound Type", "type"),
                ("Wound Status", "status"),
                ("Measurements (L x W x D)", "measurements"),
                ("Area (sq cm)", "area_sq_cm"),
                ("Volume (cm³)", "volume_cu_cm"),
                ("Tunnels", "tunnels"),
                ("Max Depth (cm)", "max_depth"),
                ("Undermining (cm)", "undermining"),
                ("Stage / Grade", "stage_grade"),
                ("Exudate Amount", "exudate_amount"),
                ("Exudate Type", "exudate_type"),
                ("Odor", "odor"),
                ("Wound Margin", "wound_margin"),
                ("Periwound", "periwound"),
                ("Necrotic Material (%)", "necrotic_material"),
                ("Granulation (%)", "granulation"),
                ("Tissue Exposed", "tissue_exposed"),
                ("Debridement", "debridement"),
                ("Primary Dressing", "primary_dressing"),
                ("Secondary Dressing", "secondary_dressing"),
                ("Frequency", "frequency"),
                ("Special Equipment", "special_equipment")
            ]
            wound_headers = [f"<th>Wound {w.get('number', i+1)}</th>" for i, w in enumerate(wounds_data)]
            html += "<h2 class='wc-h2'>Wound Assessment Table</h2>\n<table class='wound-table'>\n"
            html += "  <thead><tr><th>Field</th>" + "".join(wound_headers) + "</tr></thead>\n<tbody>\n"
            for label, key in ATTRS:
                html += f"  <tr><th>{label}</th>" + "".join(f"<td>{w.get(key, '-')}</td>" for w in wounds_data) + "</tr>\n"
            html += "</tbody></table>\n"

        # 3. Detailed Summaries
        summaries = d.pop('Detailed Visit Summaries', d.pop('summaries', []))
        if not summaries and wounds_data:
            summaries = []
            for i, w in enumerate(wounds_data, start=1):
                num = w.get('number', str(i))
                summaries.append({
                    "header": f"🩹 Wound {num}: {w.get('type', '-')}",
                    "narrative": w.get('clinical_summary', w.get('clinical_summary_notes', '-')),
                    "location": w.get('location', '-'),
                    "stage_grade": w.get('stage_grade', '-'),
                    "debridement_sharp": w.get('debridement_sharp'),
                    "debridement_none": w.get('debridement_none'),
                    "debridement_details": w.get('debridement_details', '-'),
                    "primary_dressing": w.get('primary_dressing', '-'),
                    "secondary_dressing": w.get('secondary_dressing', '-'),
                    "offloading_equipment": w.get('offloading_equipment', '-')
                })

        if summaries:
            html += "<h2 class='wc-h2'>Detailed Visit Summaries</h2>\n"
            for s in summaries:
                header = s.get('header', 'Wound Summary')
                narrative = s.get('narrative', s.get('content', '-'))
                html += f"<div class='summary-block'><div class='summary-header'>{header}</div>"
                html += "<div>"
                if 'location' in s: html += f"  <p><strong>Location:</strong> {s['location']}</p>"
                if 'stage_grade' in s: html += f"  <p><strong>Stage:</strong> {s['stage_grade']}</p>"
                if 'primary_dressing' in s and s['primary_dressing'] != "-": html += f"  <p><strong>Primary Dressing:</strong> {s['primary_dressing']}</p>"
                if 'secondary_dressing' in s and s['secondary_dressing'] != "-": html += f"  <p><strong>Secondary Dressing:</strong> {s['secondary_dressing']}</p>"
                if 'offloading_equipment' in s and s['offloading_equipment'] != "-": html += f"  <p><strong>Offloading Equipment:</strong> {s['offloading_equipment']}</p>"
                
                if any(k in s for k in ["debridement_sharp", "debridement_none"]):
                    d_sharp_checked = 'checked' if s.get("debridement_sharp") else ''
                    d_none_checked = 'checked' if s.get("debridement_none") else ''
                    html += f"""
                    <ul style='list-style-type:none; padding-left:15px; margin-top:5px;'>
                        <li><strong>Debridement:</strong>
                            <ul style='list-style-type:none; padding-left:15px; margin-top:5px;'>
                                <li><input type='checkbox' {d_sharp_checked} class='wc-check' onclick='return false;'> Sharp debridement</li>
                                <li><input type='checkbox' {d_none_checked} class='wc-check' onclick='return false;'> No debridement</li>
                                <li><strong>Details:</strong> {s.get('debridement_details', '-')}</li>
                            </ul>
                        </li>
                    </ul>"""

                if narrative and narrative != "-":
                    html += "<div style='margin-top: 10px;'><strong>Clinical Narrative:</strong>"
                    clean_narrative = str(narrative).replace('\\n', '\n')
                    clean_narrative = re.sub(r'(?<=[.\s])\s*(Procedure:)', r'\n\1', clean_narrative)
                    clean_narrative = re.sub(r'(?<=[.\s])\s*(Plan:)', r'\n\1', clean_narrative)
                    for line in clean_narrative.split('\n'):
                        if line.strip(): html += f"<p style='margin: 3px 0; padding-left: 10px;'>{line.strip()}</p>"
                    html += "</div>"
                html += "</div></div>\n"
        
        # 4. Comments & EM
        if 'Provider Comment' in d or 'comments' in d:
            comm = d.pop('Provider Comment', d.pop('comments', "-"))
            if comm and comm != "-":
                clean_val = str(comm).replace('\\n', '<br>')
                html += f"<h2 class='wc-h2'>Provider Comments</h2><div style='padding-left:10px;'>{clean_val}</div>"
        
        if 'EM Justification' in d or 'em_justification' in d:
            em = d.pop('EM Justification', d.pop('em_justification', {}))
            html += "<h2 class='wc-h2'>E/M Justification</h2>\n<div style='padding-left: 10px;'>"
            em_fields = [
                ('Time Spent Examining/Evaluating', 'time_spent_examining'),
                ('Time Spent Documenting', 'time_spent_documenting'),
                ('Time Spent Coordinating', 'time_spent_coordinating'),
                ('Resolved Wound Sign Off', 'resolved_wound_sign_off'),
                ('Total Time', 'total_time')
            ]
            for label, key in em_fields:
                val = em.get(label, em.get(key, '______'))
                html += f"<p>{label}: {val}</p>"
            html += "</div>"

        if d:
            html += "<h2 class='wc-h2'>Additional Information</h2>\n"
            html += format_content(d)
        return html

    def render_mist_chart(data, level=1, parent_key=None):
        html = ""
        # Handle parsed JSON layout - use pop to track rendered keys
        pi = data.pop("Patient Information", {}) or {}
        entries = data.pop("Wound Entries", data.pop("Patient Wound Entries", []))
    
        if not entries and not pi:
            # Fallback
            return "<p>No MIST wound entries found.</p>"
    
        p_name = pi.get("Patient Name", pi.get("patient_name", "______________________________"))
        if p_name == "-": p_name = "______________________________"
        
        html += "<h1 style='text-align:center;'>Multi Wound Chart Details</h1>\n"
        
        # Static Patient Information block matching PDF text layout exactly
        html += "<table class='patient-info-mist' style='width:100%; margin-bottom: 25px;'>\n"
        html += f"  <tr><th>Patient Name:</th><td>{pi.get('Patient Name', '-')}</td><th>Date:</th><td>{pi.get('Date', '-')}</td></tr>\n"
        html += f"  <tr><th>Patient Date of Birth:</th><td>{pi.get('Patient Date of Birth', '-')}</td><th>Physician/Extender:</th><td>{pi.get('Physician/Extender', '-')}</td></tr>\n"
        html += f"  <tr><th>Transcriptionist:</th><td>{pi.get('Transcriptionist', '-')}</td><th>Facility:</th><td>{pi.get('Facility', '-')}</td></tr>\n"
        html += "</table>\n"
    
        # MIST specific attributes requested in PDF template that map to our JSON Schema
        ATTRS = [
            ("Wound Number", "Wound Number"),
            ("MIST Therapy", "MIST Therapy"),
            ("Wound Location", "Wound Location"),
            ("Outcome", "Outcome"),
            ("Wound Type", "Wound Type"),
            ("Wound Status", "Wound Status"),
            ("Measurements L x W x D", "Measurements L x W x D"),
            ("Area (sq cm)", "Area (sq cm)"),
            ("Volume (cm3)", "Volume (cm3)"),
            ("Treatment No.", "Treatment No."),
            ("Time", "Time"),
            ("Tunnels", "Tunnels"),
            ("Max depth of deepest tunnel (cm)", "Max depth of deepest tunnel (cm)"),
            ("Undermining (cm)", "Undermining (cm)"),
            ("Stage or grade if applicable", "Stage or grade if applicable"),
            ("Exudate Amount", "Exudate Amount"),
            ("Exudate Type", "Exudate Type"),
            ("Odor", "Odor"),
            ("Wound Margin", "Wound Margin"),
            ("Periwound", "Periwound"),
            ("Necrotic Material", "Necrotic Material"),
            ("Granulation", "Granulation"),
            ("Tissue Exposed", "Tissue Exposed"),
            ("Debridement", "Debridement"),
            ("MIST indication", "MIST indication"),
            ("Benchmark Justification", "Benchmark Justification"),
            ("NCF", "NCF"),
            ("TO Pre", "TO Pre"),
            ("TO post", "TO post"),
            ("Treatment performed", "Treatment performed"),
            ("PT specific comments/documentation", "PT specific comments/documentation"),
        ]
        
        wound_headers = [f"<th>Wound {w.get('Wound Number', i+1) if isinstance(w, dict) else i+1}</th>" for i, w in enumerate(entries)]
        html += "<table class='wound-table'>\n"
        html += "  <thead><tr><th>Field</th>" + "".join(wound_headers) + "</tr></thead>\n<tbody>\n"
        
        for label, key in ATTRS:
            html += f"  <tr><th>{label}</th>"
            for w in entries:
                val = w.get(key, "-") if isinstance(w, dict) else "-"
                if val is None: val = "-"
                html += f"<td>{val}</td>"
            html += "</tr>\n"
            
        html += "</tbody></table>\n"
        
        physician_rec_raw = data.pop("Physician Recommendation Details", "-")
        if isinstance(physician_rec_raw, list):
            physician_rec = "\n\n".join(str(i) for i in physician_rec_raw)
        else:
            physician_rec = str(physician_rec_raw) if physician_rec_raw else "-"
        
        if physician_rec != "-":
            html += "<h2 class='wc-h2'>Physician Recommendation Details</h2>\n"
            # Split by double newline to handle different wounds
            blocks = physician_rec.split('\n\n')
            for block in blocks:
                if not block.strip(): continue
                block_lines = block.strip().split('\n')
                if not block_lines: continue
                
                # Header: Wound #...
                html += f"<div style='margin-bottom: 20px;'>\n"
                html += f"  <p><strong>{block_lines[0]}</strong></p>\n"
                
                # Sub-headers and content
                for i in range(1, len(block_lines)):
                    line = block_lines[i]
                    if "Procedure:" in line:
                        html += f"  <p style='margin-bottom: 5px;'><strong>Procedure</strong></p>\n"
                        # Use line content if it contains more than just the label
                        line_parts = line.split("Procedure:", 1)
                        line_content = line_parts[1].strip() if len(line_parts) > 1 else ""
                        if line_content and line_content not in ["-", "not mentioned"]:
                            proc_val = line_content
                        else:
                            # Explicitly handle block as string for type inference
                            block_str = str(block)
                            proc_val = "• MIST Therapy" if "mist" in block_str.lower() else "• procedure not mentioned"
                        html += f"  <p style='margin-top: 0; padding-left: 20px;'>{proc_val}</p>\n"
                    elif "mist" in line.lower() or "• procedure not mentioned" in line.lower():
                        continue
                    else:
                        if line.strip():
                            html += f"  <p style='margin-top: 0; padding-left: 20px;'>{line.strip()}</p>\n"
                html += f"</div>\n"
        
        # Append Provider Comments
        provider_comments = data.pop("Provider Comments", "-")
        if provider_comments == "-":
            # Fallback: check if any wound entries have PT specific comments
            all_pt_comments: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict): continue
                cmt = entry.get("PT specific comments/documentation", "-")
                if cmt != "-":
                    all_pt_comments.append(f"Wound #{entry.get('Wound Number', '?')}: {cmt}")
            if all_pt_comments:
                provider_comments = "\n".join(all_pt_comments)

        if provider_comments != "-":
            html += "<h2 class='wc-h2'>Provider Comments</h2>\n"
            clean_val = str(provider_comments).replace('\\n', '<br>').replace('\n', '<br>')
            html += f"<div style='padding-left:10px; font-size: 14px;'>{clean_val}</div>\n"
        
        if data:
            html += "<h2 class='wc-h2'>Additional Information</h2>\n"
            html += format_content(data)
        return html

    def format_content(content, level=1, parent_key=None):
        html = ""
        if isinstance(content, dict):
            for key, value in content.items():
                header_level = min(level + 1, 6)
                html += f"<h{header_level}>{key}</h{header_level}>\n"
                if not value: html += "<p>Not mentioned.</p>\n"
                else: html += format_content(value, level + 1, key)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, list):
                    for idx, line in enumerate(item):
                        margin = "0em" if idx == 0 else "2em"
                        bullet = "• " if idx == 0 and not is_numbered(line) else ("○ " if idx > 0 else "")
                        html += f"<p style='margin-left: {margin};'>{bullet}{line}</p>\n"
                elif isinstance(item, dict):
                    html += "<div class='summary-block'>\n" + format_content(item, level) + "</div>\n"
                else: 
                     # For a standard list of strings, add bullets
                     bullet = "• " if not is_numbered(item) else ""
                     html += f"<p style='margin-left: 0em;'>{bullet}{item}</p>\n"
        else:
            for text in str(content).strip().split('\n'):
                if text.strip(): html += f"<p>{text.strip()}</p>\n"
        return html
 
    # Execute conversion
    if schema_name == "mist_documentation" or "Patient Wound Entries" in data or "Multi Wound Chart Details" in data:
        body_content = render_mist_chart(data)
        title_html = ""
    elif schema_name == "wound_care_visit_schema" or "Wound Assessment Table" in data or "wounds" in data or "Follow-Up - Wound Chart Details" in data:
        body_content = render_special_chart(data)
        title_html = ""
    else:
        body_content = format_content(data)
        title_html = "<h1>MEDICAL NOTE</h1>"
 
    html_output = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }} h1, h2, h3, h4, h5, h6 {{ color: #000; margin-top: 25px; }} p {{ margin-bottom: 10px; }} table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; table-layout: fixed; }} th, td {{ border: 1px solid #999; padding: 8px; text-align: left; word-wrap: break-word; font-size: 13px; }} .patient-info th {{ width: 25%; background-color: #f8f9fa; }} .patient-info-mist th {{ width: 22%; text-align: left; border: 1px solid #999; background-color: #fff; font-weight: bold; color: #000; }} .patient-info-mist td {{ width: 28%; border: 1px solid #999; }} .wound-table th {{ background-color: #f8f9fa; width: 30%; }} .summary-block {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 4px; }} .summary-header {{ font-weight: bold; margin-bottom: 8px; font-size: 1.1em; border-bottom: 1px solid #ccc; }} .wc-h2 {{ border-bottom: 1px solid #000; margin-bottom: 15px; font-size: 1.25em; padding-bottom: 5px; }} .wc-check {{ accent-color: #333333; width: 14px; height: 14px; vertical-align: middle; cursor: default; }}</style></head><body>{title_html}{body_content}</body></html>"
 
    def format_value(value):
        if isinstance(value, dict): return "\n".join(f"{k}: \n{format_value(v)}" for k, v in value.items())
        elif isinstance(value, list): return "\n".join(format_value(i) for i in value)
        else: return str(value)
 
    if isinstance(orig_data_for_sections, dict):
        sectionwise_output = [{'category': f"{k}", 'content': f"{format_value(v)}"} for k, v in orig_data_for_sections.items()]
    else:
        sectionwise_output = [{'category': 'Wound Care Report', 'content': f"{format_value(orig_data_for_sections)}"}]
    return html_output, sectionwise_output
