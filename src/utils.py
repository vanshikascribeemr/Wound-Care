import re
import os

# ─────────────────────────────────────────────────────────────
# Scriberyte Filename Convention
# ─────────────────────────────────────────────────────────────
# All files follow:
#   {timestamp}-{visitID}-{providerUUID}-{chart|addendum}-{version}.ext
#
# Examples:
#   20260218143000-550e8400-e29b-41d4-a716-446655440000-660e8400-f30c-52e5-b827-557766551111-chart-1.mp3
#   20260218143000-550e8400-e29b-41d4-a716-446655440000-660e8400-f30c-52e5-b827-557766551111-addendum-1.mp3
#
# Output files share the same base name, only extension differs:
#   ...chart-1.html   (chart)
#   ...chart-1.json   (clinical data)
#   ...chart-1.txt    (transcript)
# ─────────────────────────────────────────────────────────────

UUID_PATTERN = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'


def parse_audio_filename(filename: str) -> dict:
    """
    Parse a Scriberyte audio filename into its components.

    Format: {timestamp}-{visitID}-{providerUUID}-{chart|addendum}-{version}.ext

    Returns dict with:
        base_name       : full name without extension (e.g. "20260218-VID-UUID-chart-1")
        timestamp       : timestamp portion
        visit_id        : visit ID (UUID)
        provider_uuid   : provider UUID
        file_type       : "chart" or "addendum"
        version         : version string (e.g. "1")
        appointment_id  : timestamp-visitID-providerUUID (unique per visit, used as state key)
        is_addendum     : bool
        extension       : original file extension (e.g. ".mp3")
    """
    name = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]

    # 1. Detect type marker
    is_addendum = bool(re.search(r'(-addendum-|_add)', name, re.IGNORECASE))

    if '-chart-' in name.lower():
        marker = '-chart-'
        file_type = 'chart'
    elif '-addendum-' in name.lower():
        marker = '-addendum-'
        file_type = 'addendum'
    elif '_add' in name.lower():
        marker = '_add'
        file_type = 'addendum'
    else:
        marker = None
        file_type = 'chart'

    # 2. Split into appointment_id (base_id) and version
    if marker and marker in ['-chart-', '-addendum-']:
        parts = re.split(marker, name, flags=re.IGNORECASE, maxsplit=1)
        base_id = parts[0]
        version = parts[1] if len(parts) > 1 else '1'
    elif marker == '_add':
        parts = re.split(r'_add', name, flags=re.IGNORECASE, maxsplit=1)
        base_id = parts[0].strip('-_ ')
        version = parts[1].strip('-_ ') if len(parts) > 1 and parts[1].strip('-_ ') else '1'
    else:
        base_id = name
        version = '1'

    # 3. Extract UUIDs from base_id
    #    First UUID = visitID, Second UUID = providerUUID
    uuids = re.findall(UUID_PATTERN, base_id, re.IGNORECASE)

    visit_id = uuids[0] if len(uuids) >= 1 else base_id
    provider_uuid = uuids[1] if len(uuids) >= 2 else None

    # 4. Extract timestamp (everything before the first UUID)
    timestamp = ''
    if uuids:
        first_uuid_pos = base_id.lower().index(uuids[0].lower())
        timestamp = base_id[:first_uuid_pos].rstrip('-')

    return {
        'base_name': name,
        'timestamp': timestamp,
        'visit_id': visit_id,
        'provider_uuid': provider_uuid,
        'file_type': file_type,
        'version': version,
        'appointment_id': base_id,  # timestamp-visitID-providerUUID
        'is_addendum': is_addendum,
        'extension': ext,
    }


def get_output_basename(original_audio_filename: str) -> str:
    """
    Get the base name for output files from the original audio filename.
    Simply strips the extension — all outputs use the same base name.

    Input:  "20260218-VID-UUID-chart-1.mp3"
    Output: "20260218-VID-UUID-chart-1"
    """
    return os.path.splitext(original_audio_filename)[0]


def clean_narrative_text(text: str) -> str:
    """
    Cleans up clinical narrative text:
    1. Replaces unintended ' x ' separators with proper punctuation.
    2. Fixes common clinical transcription errors.
    3. Ensures proper sentence casing and spacing.
    """
    if not text or text == "-":
        return text
    
    res = text.strip()
    
    # 1. Clinical Term Normalization (Run BEFORE punctuation fix to catch things like "3 x Forced")
    norm_map = {
        "alkenate": "alginate",
        "calcium alkenate": "calcium alginate",
        "protecting wood": "protecting boot",
        "protecting wedge": "positioning wedge",
        "comparison wrap": "compression wrap",
        "compilation therapy": "compression therapy",
        "dry quartz": "dry gauze", 
        "boarded foam": "bordered foam",
        "boarder foam": "bordered foam",
        "mild honey": "Medihoney",
        "normal saline": "Normal Saline",
        "rejuvenation": "elevation",
        "Education 3 x Forced": "Education reinforced",
        "Education 3 x reinforced": "Education reinforced",
        "Education reinforced": "Education was reinforced"
    }
    
    for old, new in norm_map.items():
        res = re.sub(re.escape(old), new, res, flags=re.IGNORECASE)

    # 2. Fix unintentional " x " separator glitch
    
    # pattern: optional period + space + x + space followed by a sentence starter
    sentence_starters = "Apply|Cleaned|Continue|Change|Heal|To|No|Education|Patient|Observed|Wound|Dressings|Initiate|Encourage"
    # Added [a-z] to catching start of sentence if it was lowercase
    res = re.sub(r'[\.\s]*\s+x\s+([A-Z]|' + sentence_starters + r')', r'. \1', res)
    
    # Handle lowercase joining: "word x word" -> "word. Word"
    # Ensure it's not a measurement like "4 x 5"
    # We use a lambda to capitalize the second word
    def repl_func(match):
        return f"{match.group(1)}. {match.group(2).capitalize()}"
    
    res = re.sub(r'([a-zA-Z]{2,})\s+x\s+([a-zA-Z]{2,})', repl_func, res)

    # Handle trailing x
    res = re.sub(r'\s+x\s*$', r'.', res)

    # 3. Final Punctuation Check
    # Ensure double periods don't happen
    res = res.replace("..", ".")
    res = res.replace(". .", ".")
    
    # Ensure start of sentence is capitalized (optional but nice)
    if res and res[0].islower():
        res = res[0].upper() + res[1:]

    return res.strip()
