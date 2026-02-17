from app.utils.quality_validator import collect_low_quality, flag_low_quality


def test_flag_low_quality_short():
    bad, reason = flag_low_quality("problem_statement", "hi")
    assert bad is True
    assert reason == "too_short"


def test_flag_low_quality_vague():
    bad, reason = flag_low_quality("problem_statement", "TBD")
    assert bad is True
    assert reason == "vague_placeholder"


def test_flag_low_quality_blob():
    text = "lorem " * 200
    bad, reason = flag_low_quality("problem_statement", text)
    assert bad is True
    assert reason == "context_blob"


def test_list_low_quality():
    bad, reason = flag_low_quality("target_users", ["TBD", ""])
    assert bad is True
    assert reason == "all_items_low_quality"


def test_collect_low_quality_nested():
    structured = {
        "project_name": "Demo",
        "tech_stack": {"frontend_choice": "TBD"},
    }
    results = collect_low_quality(structured, ["tech_stack.frontend_choice"])
    assert results[0]["field"] == "tech_stack.frontend_choice"
