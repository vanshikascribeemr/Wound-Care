import asyncio
import os
from src.manager import EncounterManager
from dotenv import load_dotenv

load_dotenv()

async def run_demo():
    # Initialize manager
    manager = EncounterManager()
    
    # 1. Initial Dictation
    print("--- Phase 1: Initial Dictation ---")
    initial_transcript = """
    Patient John Doe, DOB 05/12/1965. Facility Grace Wound Center.
    Wound one is on the left heel, measures 2.5 by 3 cm with 0.5 depth. 
    Moderate serosanguinous drainage present. 
    Performed sharp debridement to healthy tissue. 
    Applied collagen dressing and foam cover.
    Overall patient is healing well, continue currently plan.
    """
    
    print("Processing initial dictation...")
    encounter = await manager.create_from_transcript(initial_transcript, patient_id="P123")
    print(f"Created Encounter: {encounter.encounter_id} (Version: {encounter.version})")
    
    # Render and save
    html = manager.render_encounter(encounter.encounter_id)
    with open("report_v1.html", "w") as f:
        f.write(html)
    print("Report v1 saved to report_v1.html")

    # 2. Addendum (Patching)
    print("\n--- Phase 2: Addendum Dictation ---")
    addendum_transcript = """
    Addendum for wound one on John Doe. 
    Actually the depth is now 0.8 cm not 0.5. 
    Also add wound two on right lateral malleolus, 1 by 1 cm, superficial.
    """
    
    print("Applying addendum patch...")
    updated_encounter = await manager.apply_addendum(encounter.encounter_id, addendum_transcript)
    print(f"Updated Encounter (Version: {updated_encounter.version})")
    
    # Render and save
    updated_html = manager.render_encounter(updated_encounter.encounter_id)
    with open("report_v2.html", "w") as f:
        f.write(updated_html)
    print("Report v2 saved to report_v2.html")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set GOOGLE_API_KEY in .env file to run this demo.")
    else:
        asyncio.run(run_demo())
