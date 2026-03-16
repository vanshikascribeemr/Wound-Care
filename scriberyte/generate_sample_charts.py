
import json
import os
import copy
import re
import sys
from types import ModuleType

# 1. MOCK 'utils.utils' before importing any code that depends on it
m = ModuleType("utils")
sys.modules["utils"] = m
m_utils = ModuleType("utils.utils")
sys.modules["utils.utils"] = m_utils
m_utils.send_email = lambda *args, **kwargs: print(f"Mock email sent: {args[0]}")

# 2. MOCK environment variables required by the main script
os.environ["ORG"] = "test_org"
os.environ["ENV"] = "test_env"
os.environ["GEMINI_MODEL"] = "gemini-1.5-pro"
os.environ["LLM"] = "google"
os.environ["CHATGPT_MODEL"] = "gpt-4o"

# 3. Import the target function
from NEW_doctor_patient_conversation_utils_code import json_to_html_with_sections_for_wound_care

def generate_charts():
    base_path = r"c:\Users\uttarwar.vanshika\Documents\WC-Dectation\scriberyte\audio and  chart"
    
    # 1. MIST Chart (using JSON found in the directory)
    mist_json_path = os.path.join(base_path, "2026-03-04-05-28-18-431879-96c4024d-0e4a-4e12-9b3e-6c944eb10d81-chart-vt1_s60DB_MIST.json")
    with open(mist_json_path, 'r', encoding='utf-8') as f:
        mist_data = f.read()
    
    print(f"Generating MIST Chart from {os.path.basename(mist_json_path)}...")
    mist_html, _ = json_to_html_with_sections_for_wound_care(mist_data, schema_name="mist_documentation")
    mist_out = os.path.join(base_path, "MIST_Sample_Output.html")
    with open(mist_out, 'w', encoding='utf-8') as f:
        f.write(mist_html)
    print(f"Generated MIST Chart: {mist_out}")

    # 2. FOLLOWUP Chart (using JSON found in the directory)
    followup_json_path = os.path.join(base_path, "2026-03-04-05-58-47-431882-96c4024d-0e4a-4e12-9b3e-6c944eb10d81-chart-vt1_s60DB_FOLLOWUP.json")
    with open(followup_json_path, 'r', encoding='utf-8') as f:
        followup_data = f.read()
    
    print(f"Generating Followup Chart from {os.path.basename(followup_json_path)}...")
    followup_html, _ = json_to_html_with_sections_for_wound_care(followup_data, schema_name="wound_care_visit_schema")
    followup_out = os.path.join(base_path, "FOLLOWUP_Sample_Output.html")
    with open(followup_out, 'w', encoding='utf-8') as f:
        f.write(followup_html)
    print(f"Generated Followup Chart: {followup_out}")

if __name__ == "__main__":
    generate_charts()
