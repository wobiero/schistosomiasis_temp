from schisto_tool.user_guide import HELP_TEXT, manual_availability


def test_user_guide_help_text_and_manual_assets_packaged():
    required_keys = {
        "country",
        "disease_module",
        "mda_coverage",
        "psa_iterations",
        "m_prev",
        "h_prev",
        "target_year",
    }
    assert required_keys.issubset(HELP_TEXT.keys())
    availability = manual_availability()
    assert availability["pdf"] is True
    assert availability["docx"] is True
