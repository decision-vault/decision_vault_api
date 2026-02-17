"""
Quality validation is necessary because schema validation only checks types,
not whether the content is usable. This complements missing-fields detection
by treating vague or placeholder values as effectively missing.
"""

from __future__ import annotations

import re


VAGUE_PATTERNS = [
    r"^tbd$",
    r"^to be decided$",
    r"^to be determined$",
    r"^n/a$",
    r"^none$",
    r"^unknown$",
    r"^later$",
]


def _is_vague(text: str) -> bool:
    cleaned = text.strip().lower()
    return any(re.match(pat, cleaned) for pat in VAGUE_PATTERNS)


def _looks_like_blob(text: str) -> bool:
    # Heuristic: very long with low punctuation density or repeated clauses.
    if len(text) < 300:
        return False
    punctuation = sum(text.count(ch) for ch in ".!?;:")
    density = punctuation / max(len(text), 1)
    return density < 0.01


def flag_low_quality(field_name: str, value) -> tuple[bool, str | None]:
    if value is None:
        return True, "missing"
    if isinstance(value, str):
        if len(value.strip()) < 3:
            return True, "too_short"
        if _is_vague(value):
            return True, "vague_placeholder"
        if _looks_like_blob(value):
            return True, "context_blob"
        return False, None
    if isinstance(value, list):
        if not value:
            return True, "empty_list"
        # Check each item for quality; flag if all items are low quality.
        all_bad = True
        for item in value:
            bad, _ = flag_low_quality(field_name, item)
            if not bad:
                all_bad = False
                break
        return (True, "all_items_low_quality") if all_bad else (False, None)
    return False, None


def collect_low_quality(structured: dict, field_paths: list[str]) -> list[dict]:
    results: list[dict] = []

    def _get_nested(obj: dict, path: str):
        current = obj
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    for field in field_paths:
        value = _get_nested(structured, field)
        bad, reason = flag_low_quality(field, value)
        if bad:
            results.append({"field": field, "reason": reason})
    return results
