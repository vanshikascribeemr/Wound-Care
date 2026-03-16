import json
import re
import copy

def json_to_html_with_sections_for_wound_care(json_data, schema_name=None):
    """
    Convert Wound Care JSON data to a premium HTML chart.
    Handles both MIST Documentation and Follow-Up visits dynamically.
    """
    # 1. Parse JSON
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
            orig_data = copy.deepcopy(data)
        except Exception as e:
            return f"<pre>Error parsing JSON: {str(e)}\n{json_data[:200]}</pre>", []
    else:
        data = copy.deepcopy(json_data)
        orig_data = copy.deepcopy(json_data)

    # 2. Identify Visit Type
    is_mist = (schema_name == "mist_documentation") or ("Wound Entries" in data) or ("Patient Wound Entries" in data)
    
    # 3. Helper: Format Generic Content
    def format_content(content, level=1):
        html_str = ""
        if isinstance(content, dict):
            for k, v in content.items():
                if v is None or v == "-" or v == "": continue
                if level > 1:
                    html_str += f"<div style='margin-top: 5px;'><strong>{k}:</strong> "
                    if isinstance(v, (dict, list)):
                        html_str += "</div>" + format_content(v, level + 1)
                    else:
                        html_str += f"{v}</div>\n"
                else:
                    h_level = min(level + 1, 6)
                    html_str += f"<h{h_level} class='wc-section-header'>{k}</h{h_level}>\n"
                    html_str += format_content(v, level + 1)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, (dict, list)):
                    html_str += "<div class='summary-block'>\n" + format_content(item, level + 1) + "</div>\n"
                else:
                    text_val = str(item).strip()
                    if text_val and text_val != "-":
                        html_str += f"<p style='margin: 2px 0; padding-left: 20px;'>• {text_val}</p>\n"
        else:
            text_val = str(content).strip()
            if text_val and text_val != "-":
                lines = text_val.split('\n')
                if len(lines) > 1:
                    for line in lines:
                        if not line.strip(): continue
                        p_pad = "30px" if line.strip().startswith(('•', '○', '-')) else "20px"
                        html_str += f"<p style='margin: 2px 0; padding-left: {p_pad};'>{line.strip()}</p>\n"
                else:
                    html_str += f"<p style='margin: 2px 0; padding-left: 20px;'>{text_val}</p>\n"
        return html_str

    # 4. Render Chart Sections
    html = ""
    
    # A. Patient Information
    pi_dict = data.pop('patient_information', data.pop('Patient Information', {}))
    if pi_dict:
        p_name = pi_dict.get("patient_name", pi_dict.get("Patient Name", pi_dict.get("Patient name", "______________________________")))
        if p_name == "-": p_name = "______________________________"
        dos = pi_dict.get("date_of_service", pi_dict.get("Date", pi_dict.get("Date of Service", "-")))
        dob = pi_dict.get("dob", pi_dict.get("Patient Date of Birth", pi_dict.get("DOB", "-")))
        phys = pi_dict.get("physician", pi_dict.get("Physician/Extender", pi_dict.get("Physician", "-")))
        scribe = pi_dict.get("scribe", pi_dict.get("Transcriptionist", "-"))
        fac = pi_dict.get("facility", pi_dict.get("Facility", "-"))
        
        visit_title = "MIST Documentation" if is_mist else "Wound Care Follow-Up"
        html += f"<h1 style='text-align:center;'>{visit_title} Chart Details</h1>\n"
        
        html += "<table class='patient-info-table'>\n"
        html += f"  <tr><th>Patient Name:</th><td>{p_name}</td><th>Date:</th><td>{dos}</td></tr>\n"
        html += f"  <tr><th>Patient Date of Birth:</th><td>{dob}</td><th>Physician/Extender:</th><td>{phys}</td></tr>\n"
        html += f"  <tr><th>Transcriptionist:</th><td>{scribe}</td><th>Facility:</th><td>{fac}</td></tr>\n"
        html += "</table>\n"

    # B. Wound Assessment Table
    w_list = data.pop('wounds', data.pop('Wound Entries', data.pop('Patient Wound Entries', [])))
    wat_data = data.pop('Wound Assessment Table', {})
    
    if (w_list and isinstance(w_list, list)) or (wat_data and 'rows' in wat_data):
        html += "<h2 class='wc-h2'>Wound Assessment</h2>\n"
        
        if w_list:
            if is_mist:
                ATTRS = [
                    ("Wound Number", "Wound Number"), ("MIST Therapy", "MIST Therapy"), ("Wound Location", "Wound Location"),
                    ("Outcome", "Outcome"), ("Wound Type", "Wound Type"), ("Wound Status", "Wound Status"),
                    ("Measurements L x W x D", "Measurements L x W x D"), ("Area (sq cm)", "Area (sq cm)"), ("Volume (cm3)", "Volume (cm3)"),
                    ("Treatment No.", "Treatment No."), ("Time", "Time"), ("Tunnels", "Tunnels"),
                    ("Max depth of deepest tunnel (cm)", "Max depth of deepest tunnel (cm)"), ("Undermining (cm)", "Undermining (cm)"),
                    ("Stage or grade if applicable", "Stage or grade if applicable"), ("Exudate Amount", "Exudate Amount"),
                    ("Exudate Type", "Exudate Type"), ("Odor", "Odor"), ("Wound Margin", "Wound Margin"),
                    ("Periwound", "Periwound"), ("Necrotic Material", "Necrotic Material"), ("Granulation", "Granulation"),
                    ("Tissue Exposed", "Tissue Exposed"), ("Debridement", "Debridement"), ("MIST indication", "MIST indication"),
                    ("Benchmark Justification", "Benchmark Justification"), ("NCF", "NCF"), ("TO Pre", "TO Pre"),
                    ("TO post", "TO post"), ("Treatment performed", "Treatment performed"), ("PT specific comments/documentation", "PT specific comments/documentation")
                ]
            else:
                ATTRS = [
                    ("MIST Therapy", "mist_therapy"), ("Wound Location", "location"), ("Outcome", "outcome"),
                    ("Wound Type", "type"), ("Wound Status", "status"), ("Measurements L x W x D", "measurements"),
                    ("Area (sq cm)", "area_sq_cm"), ("Volume (cm3)", "volume_cu_cm"), ("Tunnels", "tunnels"),
                    ("Max Depth (cm)", "max_depth"), ("Undermining (cm)", "undermining"), ("Stage / Grade", "stage_grade"),
                    ("Exudate Amount", "exudate_amount"), ("Exudate Type", "exudate_type"), ("Odor", "odor"),
                    ("Wound Margin", "wound_margin"), ("Periwound", "periwound"), ("Necrotic Material (%)", "necrotic_material"),
                    ("Granulation (%)", "granulation"), ("Tissue Exposed", "tissue_exposed"), ("Debridement", "debridement"),
                    ("Primary Dressing", "primary_dressing"), ("Secondary Dressing", "secondary_dressing"),
                    ("Frequency", "frequency"), ("Special Equipment", "special_equipment")
                ]

            headers = [f"<th>Wound {w.get('number', w.get('Wound Number', i+1))}</th>" for i, w in enumerate(w_list)]
            html += "<table class='wound-table'>\n"
            html += "  <thead><tr><th>Field</th>" + "".join(headers) + "</tr></thead>\n<tbody>\n"
            for label, key in ATTRS:
                row_vals = "".join(f"<td>{w.get(key, '-')}</td>" for w in w_list)
                html += f"  <tr><th>{label}</th>{row_vals}</tr>\n"
            html += "</tbody></table>\n"
        elif wat_data:
            html += "<table class='wound-table'>\n"
            if 'headers' in wat_data:
                html += "  <thead><tr><th>Field</th>" + "".join(f"<th>{h}</th>" for h in wat_data['headers'][1:]) + "</tr></thead>\n"
            html += "<tbody>\n"
            for row in wat_data.get('rows', []):
                if row: html += f"  <tr><th>{row[0]}</th>" + "".join(f"<td>{v}</td>" for v in row[1:]) + "</tr>\n"
            html += "</tbody></table>\n"

    # C. Detailed Visit Summaries (Follow-Up specific layout)
    if not is_mist and w_list:
        html += "<h2 class='wc-h2'>Detailed Visit Summaries</h2>\n"
        for w in w_list:
            num = w.get("number", w.get("Wound Number", "-"))
            w_type = w.get("type", w.get("Wound Type", "-"))
            loc = w.get("location", w.get("Wound Location", "-"))
            stage = w.get("stage_grade", w.get("Stage or grade if applicable", "-"))
            
            html += "<div class='summary-block'>\n"
            html += f"  <div class='summary-header'>🩹 Wound {num}: {w_type}</div>\n"
            html += f"  <p><strong>Wound Location:</strong> {loc}</p>\n"
            html += f"  <p><strong>Stage:</strong> {stage}</p>\n"
            
            # Treatment Sub-block
            html += "  <p><strong>Treatment:</strong></p>\n"
            html += f"  <p style='margin-left:20px;'><strong>Primary Dressing:</strong> {w.get('primary_dressing', '-')}</p>\n"
            html += f"  <p style='margin-left:20px;'><strong>Secondary Dressing:</strong> {w.get('secondary_dressing', '-')}</p>\n"
            
            # Debridement Sub-block
            html += "  <p><strong>Debridement:</strong></p>\n"
            d_sharp_checked = 'checked' if w.get("debridement_sharp") else ''
            d_none_checked = 'checked' if w.get("debridement_none") else ''
            html += f"  <p style='margin-left:20px;'><input type='checkbox' {d_sharp_checked} disabled> Sharp debridement</p>\n"
            html += f"  <p style='margin-left:20px;'><input type='checkbox' {d_none_checked} disabled> No debridement</p>\n"
            html += f"  <p style='margin-left:20px;'><strong>Details:</strong> {w.get('debridement_details', '-')}</p>\n"
            
            html += f"  <p><strong>Offloading / Equipment:</strong> {w.get('offloading_equipment', '-')}</p>\n"
            
            # Narrative Summary Sub-block
            html += "  <p><strong>Summary:</strong></p>\n"
            s_content = w.get("clinical_summary", w.get("clinical_summary_notes", "-"))
            if s_content and s_content != "-":
                for line in str(s_content).split('\n'):
                    line = line.strip()
                    if not line: continue
                    if ":" in line:
                        p_parts = line.split(":", 1)
                        html += f"  <p style='margin-left:20px;'><strong>{p_parts[0].strip()}:</strong> {p_parts[1].strip()}</p>\n"
                    else:
                        html += f"  <p style='margin-left:20px;'>{line}</p>\n"
            html += "</div>\n"

    # D. E/M Justification (Follow-Up)
    em_data_dict = data.pop('em_justification', data.pop('EM Justification', {}))
    if em_data_dict:
        html += "<h2 class='wc-h2'>E/M Justification</h2>\n"
        html += "<div style='padding-left:15px;'>\n"
        em_mapping = [
            ('Time Spent Examining/Evaluating', 'time_spent_examining'),
            ('Time Spent Documenting', 'time_spent_documenting'),
            ('Time Spent Coordinating', 'time_spent_coordinating'),
            ('Resolved Wound Sign Off', 'resolved_wound_sign_off'),
            ('Total Time', 'total_time')
        ]
        has_em_entries = False
        for label, key in em_mapping:
            val = em_data_dict.get(label, em_data_dict.get(key, "-"))
            if val != "-":
               html += f"  <p><strong>{label}:</strong> {val}</p>\n"
               has_em_entries = True
        if not has_em_entries:
            html += format_content(em_data_dict, level=2)
        html += "</div>\n"

    # E. Physician Recommendation Details (MIST)
    prd_val = data.pop('Physician Recommendation Details', "-")
    if prd_val and prd_val != "-":
        html += "<h2 class='wc-h2'>Physician Recommendation Details</h2>\n"
        html += format_content(prd_val, level=1)

    # F. Provider Comments (Root level, bottom)
    comm_val = data.pop('comments', data.pop('Provider Comments', data.pop('Provider Comment', "-")))
    if comm_val and comm_val != "-":
        html += "<h2 class='wc-h2'>Provider Comments</h2>\n"
        html += format_content(comm_val, level=1)

    # G. Remaining data (if any)
    for k_rem, v_rem in data.items():
        if v_rem and v_rem != "-":
            html += f"<h2 class='wc-h2'>{k_rem}</h2>\n"
            html += format_content(v_rem, level=1)

    # 5. Final Assembly with CSS
    CSS_STYLES = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    body { font-family: 'Inter', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }
    h1, h2, h3, h4, h5, h6 { color: #000; margin-top: 25px; font-weight: 700; }
    p { margin: 8px 0; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 25px; table-layout: fixed; }
    th, td { border: 1px solid #999; padding: 10px; text-align: left; word-wrap: break-word; font-size: 13px; }
    .patient-info-table th { width: 22%; text-align: left; border: 1px solid #999; background-color: #f8f9fa; font-weight: bold; }
    .patient-info-table td { width: 28%; border: 1px solid #999; }
    .wound-table th { background-color: #f8f9fa; width: 25%; font-weight: bold; color: #000; }
    .wc-h2 { border-bottom: 2px solid #000; margin-bottom: 15px; font-size: 1.3em; padding-bottom: 5px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    .summary-block { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-left: 5px solid #000; background-color: #fafafa; }
    .summary-header { font-weight: bold; margin-bottom: 10px; font-size: 1.15em; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    .wc-section-header { border-bottom: 1px solid #ccc; font-size: 1.1em; margin-bottom: 10px; padding-bottom: 3px; font-weight: bold; }
    """
    
    html_out = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><style>{CSS_STYLES}</style></head><body>{html}</body></html>"

    # Create Section-wise list (using the original un-mutated data)
    def format_val_recursive(v):
        if isinstance(v, dict): return "\n".join(f"{k}: {format_val_recursive(val)}" for k, val in v.items())
        elif isinstance(v, list): return "\n".join(format_val_recursive(i) for i in v)
        else: return str(v)

    sectionwise_out = [{'category': f"{k}", 'content': f"{format_val_recursive(v)}"} for k, v in orig_data.items() if v and v != "-"]

    return html_out, sectionwise_out