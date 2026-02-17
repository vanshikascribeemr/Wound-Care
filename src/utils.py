import re

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
    
    # pattern: optional period + space + x + space followed by a sentence starter
    sentence_starters = "Apply|Cleaned|Continue|Change|Heal|To|No|Education|Patient|Observed|Wound|Dressings|Initiate|Encourage"
    res = re.sub(r'[\.\s]*\s+x\s+([A-Z]|' + sentence_starters + r')', r'. \1', res)
    
    # Handle lowercase joining: "word x word" -> "word. word"
    # Ensure it's not a measurement like "4 x 5"
    res = re.sub(r'([a-zA-Z]{3,})\s+x\s+([a-zA-Z]{3,})', r'\1. \2', res)
    
    # Handle trailing x
    res = re.sub(r'\s+x\s*$', r'.', res)

    # 2. Clinical Term Normalization (Post-parsing cleanup)
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
        "rejuvenation": "elevation", # "Encourage rejuvenation when seated" -> "Encourage elevation..."
        "Education 3 x Forced": "Education reinforced",
        "Education 3 x reinforced": "Education reinforced",
        "Education reinforced": "Education was reinforced"
    }
    
    for old, new in norm_map.items():
        # Case-insensitive replacement
        res = re.sub(re.escape(old), new, res, flags=re.IGNORECASE)

    # 3. Final Punctuation Check
    # Ensure double periods don't happen
    res = res.replace("..", ".")
    res = res.replace(". .", ".")
    
    # Ensure start of sentence is capitalized (optional but nice)
    if res and res[0].islower():
        res = res[0].upper() + res[1:]

    return res.strip()
