"""
Computes a completeness + quality score to inform UX:
- Low score => show more clarifying questions.
- High score => allow PRD generation.
"""

from __future__ import annotations


def compute_scores(
    required_fields: list[str],
    missing_fields: list[str],
    low_quality_fields: list[dict],
) -> dict:
    total = max(len(required_fields), 1)
    missing_set = set(missing_fields)
    present = total - len(missing_set)
    completeness_score = round((present / total) * 100)

    # Quality penalty: 2 points per low-quality field, capped at 20.
    penalty = min(len(low_quality_fields) * 2, 20)
    final_score = max(min(completeness_score - penalty, 100), 0)

    if final_score < 70:
        status = "incomplete"
    elif final_score < 90:
        status = "needs_clarification"
    else:
        status = "ready_for_prd"

    return {
        "completeness_score": completeness_score,
        "quality_penalty": penalty,
        "final_score": final_score,
        "status": status,
    }
