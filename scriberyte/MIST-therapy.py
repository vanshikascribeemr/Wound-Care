def render_mist_chart(data, level=1, parent_key=None):
    html = ""
    # Fallback to dictionary format if standard parsing sends something generic.
    if isinstance(data, list):
        entries = data
    else:
        entries = data.get("Patient Wound Entries", [])

    if not entries:
        # If the expected arrays aren't there, fall back to simple recursive html so nothing breaks.
        return "<p>No MIST wound entries found.</p>"

    first_entry = entries[0] if entries else {}
    p_name = first_entry.get("Patient name", first_entry.get("patient_name", "______________________________"))
    if p_name == "-": p_name = "______________________________"
    
    html += f"<h1>MIST - Multi Wound Chart Details &ndash; {p_name}</h1>\n"
    
    # Static Patient Information block matching PDF text layout exactly
    html += "<table style='width:100%; border:none; margin-bottom: 25px;'>\n"
    html += f"  <tr><td style='border:none;'><strong>Patient Name:</strong> {p_name}</td><td style='border:none;'><strong>Date:</strong> {pi.get('Date', '-')}</td></tr>\n"
    html += f"  <tr><td style='border:none;'><strong>Patient Date of Birth:</strong> {pi.get('Patient Date of Birth', '-')}</td><td style='border:none;'><strong>Physician/Extender:</strong> {pi.get('Physician/Extender', '-')}</td></tr>\n"
    html += f"  <tr><td style='border:none;'><strong>Transcriptionist:</strong> {pi.get('Transcriptionist', '-')}</td><td style='border:none;'></td></tr>\n"
    html += f"  <tr><td style='border:none;'><strong>Facility:</strong> {pi.get('Facility', '-')}</td><td style='border:none;'></td></tr>\n"
    html += "</table>\n"

    # MIST specific attributes requested in PDF template that map to our JSON Schema
    ATTRS = [
        ("Wound Number", "_num"),
        ("MIST Therapy", "MIST"),
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
    
    wound_headers = [f"<th>Wound {i+1}</th>" for i in range(len(entries))]
    html += "<table class='wound-table'>\n"
    html += "  <thead><tr><th>Field</th>" + "".join(wound_headers) + "</tr></thead>\n<tbody>\n"
    
    for label, key in ATTRS:
        html += f"  <tr><th>{label}</th>"
        for i, w in enumerate(entries, start=1):
            if key == "MIST":
                val = "MIST"
            elif key == "_num":
                val = str(i)
            else:
                val = w.get(key, "-")
            html += f"<td>{val}</td>"
        html += "</tr>\n"
        
    html += "</tbody></table>\n"
    
    # Under the unified table, append provider comments dynamically for each wound
    has_comments = any(w.get("Provider Comments", "-") != "-" for w in entries)
    if has_comments:
        html += "<h2 class='wc-h2'>Physician Recommendation Details</h2>\n"
        
    for i, w in enumerate(entries, start=1):
        comments = w.get("Provider Comments", "-")
        if comments != "-":
            wtype = w.get("Wound Type", "-")
            loc = w.get("Wound Location", "-")
            
            html += f"<h3 style='margin-top:20px;'>Wound #{i} - {wtype} - {loc}</h3>\n"
            html += "<div style='padding-left:10px;'>\n"
            html += "  <h4 style='margin-bottom:5px;'>Procedure</h4>\n"
            html += "  <p style='margin-top:0;'>&bull; MIST Therapy</p>\n"
            html += "  <h4 style='margin-bottom:5px;'>Provider Comments</h4>\n"
            clean_val = str(comments).replace('\\n', '<br>')
            html += f"  <div style='margin-top:0;'>{clean_val}</div>\n"
            html += "</div>\n"
            
    return html