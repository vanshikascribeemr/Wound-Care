import json
import os
import glob
from doctor_patient_conversation_utils_code import json_to_html_with_sections_for_wound_care

def main():
    directory = r"c:\Users\uttarwar.vanshika\Documents\WC-Dectation\scriberyte\audio and  chart"
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    if not json_files:
        print("No JSON files found.")
        return

    for json_file in json_files:
        print(f"Processing {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = f.read()
            
        html, sections = json_to_html_with_sections_for_wound_care(data)
        
        out_html = json_file.replace(".json", "_new_generated_chart.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Generated {out_html}")

if __name__ == "__main__":
    main()
