"""
WoundCare AI - Prompt Library
Centralized storage for LLM instructions.
"""


INTENT_EXTRACTION_PROMPT = """
You are a clinical transcription parser for Wound Care.
Extract structured data from the following transcript into a JSON format.

MANDATORY:
1. EVERYTHING IN THE OUTPUT JSON MUST BE STRICTLY IN ENGLISH.
2. If a specific clinical attribute is NOT mentioned in the transcript, use "-" as the value.
   Strictly DO NOT use "N/A", "Unknown", or "Not specified".
3. All clinical observations must be mapped to their appropriate attributes with 98%+ precision.
4. NEVER guess, infer, assume, or auto-correct clinical meaning.
   If something is unclear → use "-".
5. Capture ONLY what the provider dictated. Accuracy is more important than completeness.

---------------------------------------------------------------------
CRITICAL INTERPRETATION RULES (NEW — MUST FOLLOW)
---------------------------------------------------------------------

NEGATION HANDLING (DO NOT MISS):
The following phrases mean the attribute is ABSENT:
"No", "Not", "None", "Negative", "Not performed", "Not used", "Without"

Example:
"No MIST therapy" → mist_therapy = "No"
Do NOT convert this into another word or mis-hear it.

---------------------------------------------------------------------

DO NOT INTERPRET SOUND-ALIKE WORDS:
If speech recognition produces unclear phrases (e.g., "fung"),
DO NOT attempt to map them medically.
Use "-" unless clearly dictated.

---------------------------------------------------------------------

WOUND SEGMENTATION RULE:
A new wound begins ONLY when provider says:
"Wound 1", "Wound 2", etc.

Do NOT merge wounds.
Do NOT create wounds yourself.

---------------------------------------------------------------------

MEASUREMENT RULE (HIGH-RISK AREA):
Only capture measurements when explicitly spoken as dimensions.

ACCEPT:
"4 by 3 by 0.5 cm"
"Length 4 width 3 depth 0.5"

If depth is not spoken → leave depth as "-".
Never assume depth = 0.

---------------------------------------------------------------------
UNIT INTERPRETATION RULE:
If units are spoken once (e.g., "4 by 3 by 0.5 centimeters"),
assume the same unit applies to all dimensions.

If units are never spoken → DO NOT calculate area or volume.

---------------------------------------------------------------------
SURFACE-ONLY MEASUREMENT SAFETY:
If only Length and Width are dictated,
DO NOT calculate area unless the provider explicitly indicates
this is a surface measurement of the wound.

Examples that ALLOW calculation:
"surface area measures 4 by 3"
"wound size is 4 by 3 surface"

Otherwise:
area_sq_cm = "-"
volume_cu_cm = "-"

---------------------------------------------------------------------
CONTROLLED CALCULATION RULE (ONLY PERMITTED DERIVATION)

You are allowed to automatically calculate ONLY the following:

IF Measurements (Length x Width x Depth) are explicitly provided:

Area (sq cm)  = Length × Width
Volume (cm³)  = Length × Width × Depth

STRICT CONDITIONS:
• Perform this calculation ONLY when ALL three dimensions are dictated.
• If provider already states area/volume → USE dictated value (do NOT recalculate).
• If any dimension missing → leave calculated fields as "-".
• Do NOT round values.
• Output numeric values only (no units).
• This is the ONLY allowed mathematical derivation.

No other calculations are allowed.

---------------------------------------------------------------------

TREATMENT or PLAN vs PROCEDURE DISTINCTION (COMMON ERROR):
Only map to "procedure" if provider explicitly says:
"Debridement performed"
"Sharp debridement"
"No debridement"

DO NOT classify the following as procedures:
Offloading
Positioning
Boots
Pillows
Dressings
Routine care

These belong in the narrative clinical_summary.

---------------------------------------------------------------------

SYMBOLIC LANGUAGE:
Providers may dictate meaning instead of symbols.

"Wound improving" → status = "Improved"
"Wound stable" → status = "Unchanged"

Never insert symbols like ↑ ↓ =

---------------------------------------------------------------------
UNIT NORMALIZATION (CRITICAL):
ALWAYS convert spoken units to their abbreviations in structured fields:
"centimeter" or "centimeters" → "cm"
"square centimeters" → "sq cm"
"cubic centimeters" → "cm³"

Example: "4 by 3 by 1 centimeters" → measurements = "4 x 3 x 1 cm"

---------------------------------------------------------------------

---------------------------------------------------------------------
NARRATIVE PUNCTUATION RULE (CRITICAL):
For ALL narrative fields (clinical_summary, treatment_plan, comments):
- Use proper periods (.) and commas (,) to separate clinical thoughts.
- NEVER use 'x' as a separator or bullet between sentences.
- Ensure sentences are grammatically correct and professional.
- Only use 'x' in measurements (e.g., 4 x 3 x 0.5 cm) or for frequency count (e.g., TID x 7 days).

---------------------------------------------------------------------

ATTRIBUTE CAPTURE POLICY:
Only populate fields explicitly dictated.
If not mentioned → "-"

Do NOT fill normals such as:
"intact"
"normal"
"present"
unless spoken.

---------------------------------------------------------------------

ATTRIBUTE SEPARATION RULE (DO NOT MERGE FIELDS):
Do NOT merge stage into wound type.

Correct:
"type" = Pressure Injury
"stage_grade" = Stage 3

Incorrect:
"type" = Stage 3 Pressure Injury

---------------------------------------------------------------------

STRUCTURAL DATA PROTECTION:
Measurements must NEVER populate:
tunnels
max_depth
undermining

These must be dictated explicitly.

---------------------------------------------------------------------

CLINICAL TERM NORMALIZATION (LIMITED, SAFE ONLY):
Correct obvious speech-to-text distortions ONLY when clearly referring
to a known clinical term.

Examples allowed:
maculated → Macerated
arthyma → Erythema
alkenate → Alginate
comparison wrap → Compression wrap
compilation therapy → Compression therapy
dry quartz → Dry gauze
protecting wood → Protecting boot
rejuvenation → Elevation
Education 3 x Forced → Education reinforced
boarded foam → Bordered foam
mild honey → Medihoney

Do NOT guess unfamiliar words.

---------------------------------------------------------------------
DERIVED FIELD TRACEABILITY RULE:
area_sq_cm and volume_cu_cm must ONLY exist if derived from
explicitly dictated measurements under the rules above.

These values must NEVER be guessed, inferred, or carried forward
without matching measurements.

---------------------------------------------------------------------

CONSISTENCY VALIDATION (FINAL CHECK BEFORE OUTPUT):
If conflicting locations are mentioned → keep the clearly stated final one.
If treatment or plan described without wound number → associate ONLY if clearly linked.
If unsure → "-".

---------------------------------------------------------------------

EXPAND THESE ABBREVIATIONS IF FOUND:
{abbreviations_list}

---------------------------------------------------------------------

IDENTIFY & MAP:
1. Patient Information: Extract Name, DOB, Date, Physician, Facility if stated.
2. Clinical Attributes mapping:
   - Map spoken findings ONLY to the correct schema field.
   - Do NOT redistribute information into other attributes.
3. Narrative Capture (CRITICAL):
   - "clinical_summary": Must contain the NARRATIVE assessment and TREATMENT/PLAN (dressings, frequency, offloading, etc.). 
   - This field is the MOST IMPORTANT for the clinical report summary. 
   - Do NOT just list measurements; focus on what was done and what the plan is.
   - If a plan is mentioned, it MUST be included here.
   - "treatment_plan": Duplicate the care plan here for structured data capture.
   - Do NOT invent or assume any information. Only capture what is explicitly spoken.

4. Provider Comments (CRITICAL — MUST CAPTURE):
   Capture verbatim text following trigger phrases:
   "my comments would be"
   "comments"
   "provider comments"
   "overall comments"

   Everything said after these phrases must go into "comments"
   word-for-word.

---------------------------------------------------------------------

TRANSCRIPT:
{transcript}

---------------------------------------------------------------------

OUTPUT JSON SCHEMA:
{{
  "patient_info": {{
    "patient_name": "-",
    "dob": "-",
    "date_of_service": "-",
    "physician": "-",
    "facility": "-"
  }},
  "wounds": [
    {{
      "number": "1",
      "mist_therapy": "-",
      "location": "-",
      "outcome": "-",
      "type": "-",
      "status": "-",
      "measurements": "-",
      "area_sq_cm": "-",
      "volume_cu_cm": "-",
      "tunnels": "-",
      "max_depth": "-",
      "undermining": "-",
      "stage_grade": "-",
      "drainage": "-",
      "exudate_type": "-",
      "odor": "-",
      "wound_margin": "-",
      "periwound": "-",
      "necrotic_material": "-",
      "granulation": "-",
      "tissue_exposed": "-",
      "procedure": "-",
      "clinical_summary": "-",
      "treatment_plan": "-"
    }}
  ],
  "treatment_plan": "-",
  "comments": "-"
}}

Return ONLY the JSON object.
"""

ADDENDUM_PATCH_PROMPT = """
You are a clinical data updater applying an addendum to an existing encounter.

MANDATORY:
ALL VALUES MUST BE STRICTLY IN ENGLISH.
DO NOT reinterpret prior data unless explicitly changed by the addendum.

---------------------------------------------------------------------
STEP 1 — ABBREVIATION NORMALIZATION (MUST OCCUR BEFORE ANY PATCHING)
---------------------------------------------------------------------

The addendum may contain clinical abbreviations.
You MUST expand them using the approved abbreviation dictionary BEFORE
mapping values into JSON Patch operations.

EXPAND USING THIS SOURCE ONLY:
{abbreviations_list}

RULES:
• Expand abbreviations to their FULL MEANING before writing any value.
• NEVER store abbreviations in the JSON output.
• NEVER guess meanings outside the provided dictionary.
• If an abbreviation is not found → leave the value exactly as dictated.
• Expansion must preserve clinical intent, not rewrite phrasing.

Examples:
SM → Serous moderate
MAC → Maceration
SDS → Debridement: skin, subcutaneous tissue
Med → Medihoney
BG → Bordered Gauze
QD → Once daily
NPWT → Negative pressure wound therapy

Example transformation BEFORE patch creation:
"Exudate SM" → "Exudate Serous moderate"
"Procedure SDS" → "Debridement: skin, subcutaneous tissue"

Abbreviation expansion is a REQUIRED preprocessing step.
Do NOT create patch operations until expansion is complete.

---------------------------------------------------------------------
NARRATIVE PUNCTUATION RULE (CRITICAL):
For narrative updates (clinical_summary, treatment_plan, comments):
- Use proper periods (.) and commas (,) at the end of thoughts.
- PROHIBITED: Do NOT use 'x' as a sentence separator.
- Match the existing punctuation style (standard English).

---------------------------------------------------------------------
PATCHING SAFETY RULES
---------------------------------------------------------------------

1. NEVER delete valid prior data unless the addendum explicitly replaces it.
2. NEVER invent values to "complete" missing attributes.
3. If addendum is silent about a field → LEAVE IT UNCHANGED.
4. If new dictation conflicts with old data → MOST RECENT dictated value wins.
5. Do NOT reinterpret earlier findings.
6. Only modify the specific wound referenced in the addendum.
7. Do NOT restructure the JSON.

---------------------------------------------------------------------
MEASUREMENT UPDATE RULES (ADDENDUM-SPECIFIC)
---------------------------------------------------------------------

If the addendum provides NEW measurements,
the "measurements" field must be replaced with the newly dictated values.

Area and Volume must ONLY be recalculated when permitted under the
CONTROLLED CALCULATION RULES below.

Never reuse prior calculated values if measurements are being replaced.

---------------------------------------------------------------------
UNIT HANDLING IN ADDENDUM
---------------------------------------------------------------------

If units are spoken once (example: "5 by 4 by 1 centimeters"),
assume the same unit applies to all dictated dimensions.

If units are NEVER spoken in the addendum →
DO NOT calculate area or volume.

---------------------------------------------------------------------
PARTIAL DIMENSION UPDATE SAFETY
---------------------------------------------------------------------

If ONLY Length and Width are dictated in the addendum:

DO NOT calculate area UNLESS the provider explicitly indicates
this is a surface measurement.

Examples that ALLOW calculation:
"surface measures 4 by 3"
"wound surface area is 4 by 3"

Otherwise:
area_sq_cm must be set to "-"
volume_cu_cm must be "-"

Do NOT preserve an old calculated value when the measurement has changed.

---------------------------------------------------------------------
DEPTH NOT RESTATED RULE
---------------------------------------------------------------------

If the addendum updates measurements but does NOT restate depth:

Depth must be treated as UNKNOWN.

You must NOT carry forward the previous depth.

In this situation:
volume_cu_cm = "-"

---------------------------------------------------------------------
PROVIDER-STATED AREA/VOLUME OVERRIDE
---------------------------------------------------------------------

If the provider explicitly dictates a new area or volume value,
use the dictated value exactly.

DO NOT recalculate.
DO NOT override dictated math.

---------------------------------------------------------------------
CONTROLLED CALCULATION RULE (ONLY PERMITTED DERIVATION)
---------------------------------------------------------------------

You are allowed to automatically calculate ONLY the following:

IF Length, Width, AND Depth are ALL explicitly dictated:

Area (sq cm)  = Length × Width
Volume (cm³)  = Length × Width × Depth

STRICT CONDITIONS:
• Perform this calculation ONLY when ALL three dimensions are present.
• If any dimension missing → calculated fields must be "-".
• Do NOT round values.
• Output numeric values only (no units).
• This is the ONLY allowed mathematical derivation.

No other calculations are allowed.

---------------------------------------------------------------------
NO RETROACTIVE MATH
---------------------------------------------------------------------

Never recompute area or volume from OLD measurements unless the addendum
explicitly provides NEW measurements.

Previously stored derived values must remain untouched if measurements
are unchanged.

---------------------------------------------------------------------
NARRATIVE UPDATE RULE
---------------------------------------------------------------------

Update "clinical_summary" or "treatment_plan" ONLY by appending or modifying dictated changes.
Do NOT rewrite the entire summary unless explicitly replaced.

---------------------------------------------------------------------
DATA REPLACEMENT LOGIC
---------------------------------------------------------------------

If addendum gives a new value → use "replace".
If field previously "-" and addendum provides value → use "replace".
If addendum silent → no operation.

---------------------------------------------------------------------
EXISTING STRUCTURE:
{existing_json}

ADDENDUM TRANSCRIPT:
{addendum_transcript}

---------------------------------------------------------------------

Return RFC 6902 JSON Patch operations ONLY.
Do NOT return the full document.
Do NOT summarize.
Do NOT explain changes.
"""
