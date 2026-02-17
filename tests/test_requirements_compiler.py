from app.utils.requirements_compiler import (
    extract_bullet_features,
    detect_backend,
    detect_database,
    detect_frontend,
    detect_infra,
    generate_questions,
    validate_schema,
)


def test_sentence_bleeding_prevention():
    text = "Core features:- Weekly survey creation- Anonymous response collection- Dashboard."
    features = extract_bullet_features(text)
    assert "Weekly survey creation" in features
    assert "Anonymous response collection" in features
    assert "Dashboard" in features


def test_sentence_bleeding_prevention_simple():
    text = "- Export reports to CSV. Additional text should not be captured."
    features = extract_bullet_features(text)
    assert features == ["Export reports to CSV"]


def test_tech_stack_detection():
    text = "We will use React Native with FastAPI and MongoDB on AWS."
    assert detect_frontend(text) == "React Native"
    assert detect_backend(text) == "FastAPI"
    assert detect_database(text) == "MongoDB"
    assert detect_infra(text) == "AWS"


def test_duplicate_missing_prevention():
    structured = {
        "project_name": None,
        "problem_statement": None,
        "target_users": None,
        "desired_features": [],
        "architecture_decisions": {},
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": None,
        "constraints": {},
        "out_of_scope": None,
    }
    missing = validate_schema(structured)
    assert len(missing) == len(set(missing))


def test_low_quality_rejection():
    structured = {
        "project_name": "TBD",
        "problem_statement": "should be defined",
        "target_users": None,
        "desired_features": [],
        "architecture_decisions": {},
        "tech_stack": {},
        "non_functional": {},
        "success_metrics": None,
        "constraints": {},
        "out_of_scope": None,
    }
    missing = validate_schema(structured)
    assert "project_name" in missing
    assert "problem_statement" in missing


def test_question_generation_dedupe():
    missing = ["project_name", "project_name", "tech_stack.infra_provider"]
    questions = generate_questions(missing)
    assert questions.count("What is the name of the product?") == 1
