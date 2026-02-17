"""
End-to-end demo:
- Parse raw requirements into a structured partial object.
- Validate quality of fields.
- Compute missing fields and user-friendly questions.
This mirrors the intake stage before PRD generation.
"""

from __future__ import annotations

import asyncio
import json

from app.services.requirements_service import parse_raw_requirements, _missing_fields, REQUIRED_FIELDS
from app.utils.quality_validator import collect_low_quality
from app.utils.question_builder import build_questions


SPENDWISE_RAW = """
SpendWise version 1.0, managed by the Product Manager of the Fintech Division as of February 11, 2026,
is a mobile application designed to help users track daily expenses, categorize spending, and generate
monthly reports. The primary goal is to simplify personal finance management for working professionals
by enabling them to log expenses instantly and gain insights into their spending patterns by category
and date range. The application must allow users to add, edit, and delete expenses, each containing
details such as title, amount, category, date, and an optional note. It should provide a clear view of
total spending by category, automatically generate monthly reports summarizing spending trends, and
support export options such as CSV or PDF. From a performance and security standpoint, the app should
load within three seconds, synchronize data securely across devices using encrypted (AES-256) cloud storage,
and offer both light and dark modes for better usability. Success will be measured by achieving a 90%
task success rate in expense management during usability testing, maintaining a data sync failure rate
under 1%, and reaching an average of at least 5,000 daily active users within three months after launch.
Dependencies for this release include integration with an external payment categorization API for automatic
expense tagging and the use of Google Drive or Apple iCloud SDK for seamless data synchronization.
"""


async def run_demo():
    state = {
        "raw_text": SPENDWISE_RAW,
        "structured": {},
        "answers": {},
        "questions": [],
        "missing_fields": [],
        "prd": None,
    }
    parsed = await parse_raw_requirements(state)
    structured_partial = parsed["structured"]

    low_quality_fields = collect_low_quality(structured_partial, REQUIRED_FIELDS)
    missing_fields = _missing_fields(structured_partial)
    questions = build_questions(missing_fields)

    output = {
        "structured_partial": structured_partial,
        "low_quality_fields": low_quality_fields,
        "missing_fields": missing_fields,
        "questions": questions,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    # In the PRD pipeline:
    # - structured_partial is stored and displayed for review
    # - low_quality_fields and missing_fields determine clarification questions
    # - questions are asked before allowing PRD generation
    asyncio.run(run_demo())
