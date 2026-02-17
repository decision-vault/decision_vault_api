from app.utils.requirements_scoring import compute_scores


def test_incomplete_score():
    required = ["a", "b", "c", "d", "e"]
    missing = ["b", "c", "d"]
    low_quality = []
    score = compute_scores(required, missing, low_quality)
    assert score["status"] == "incomplete"


def test_needs_clarification_score():
    required = ["a", "b", "c", "d", "e"]
    missing = ["d"]
    low_quality = [{"field": "c", "reason": "too_short"}]
    score = compute_scores(required, missing, low_quality)
    assert score["status"] == "needs_clarification"


def test_ready_for_prd_score():
    required = ["a", "b", "c", "d", "e"]
    missing = []
    low_quality = []
    score = compute_scores(required, missing, low_quality)
    assert score["status"] == "ready_for_prd"
