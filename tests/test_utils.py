from src.utils import clean_narrative_text

def test_clean_narrative_text_basic():
    """Test basic punctuation cleanup."""
    assert clean_narrative_text("Hello x World") == "Hello. World"
    assert clean_narrative_text("patient is fine x no issues") == "Patient is fine. No issues"

def test_clean_narrative_text_measurements():
    """Test that measurements are preserved."""
    assert clean_narrative_text("Wound is 4 x 3 cm") == "Wound is 4 x 3 cm"
    assert clean_narrative_text("Size 4 x 3 x 0.5 cm") == "Size 4 x 3 x 0.5 cm"

def test_clean_narrative_text_frequency():
    """Test that frequencies are preserved."""
    assert clean_narrative_text("Apply TID x 7 days") == "Apply TID x 7 days"
    assert clean_narrative_text("Medihoney x 3") == "Medihoney x 3"

def test_clean_narrative_text_multiple_sentences():
    """Test multiple merged sentences."""
    input_text = "Wound cleaning x Apply dressing x To be changed daily"
    expected = "Wound cleaning. Apply dressing. To be changed daily"
    assert clean_narrative_text(input_text) == expected

def test_clinical_normalizations():
    """Test that specific clinical terms are corrected."""
    assert clean_narrative_text("Apply dry quartz") == "Apply dry gauze"
    assert clean_narrative_text("Encourage rejuvenation") == "Encourage elevation"
    assert clean_narrative_text("Education 3 x Forced") == "Education was reinforced"

def test_trailing_x():
    """Test removal of trailing x."""
    assert clean_narrative_text("End of note x") == "End of note."
