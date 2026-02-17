from app.utils.question_builder import build_questions


def test_simple_fields():
    missing = ["project_name", "authentication_strategy", "success_metrics"]
    questions = build_questions(missing)
    assert questions == [
        "What is the project name?",
        "How should users authenticate?",
        "How will we measure success?",
    ]


def test_nested_fields():
    missing = ["tech_stack.infra_region", "non_functional.performance_goals"]
    questions = build_questions(missing)
    assert questions == [
        "Which region will the infrastructure be deployed in?",
        "What performance targets should we meet?",
    ]


def test_unknown_field_fallback():
    missing = ["unknown.new_field"]
    questions = build_questions(missing)
    assert questions == ["Can you clarify 'unknown.new_field'?"]
