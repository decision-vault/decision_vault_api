"""
Deterministic requirements compiler (no LLM).
Rules:
- No sentence bleeding.
- No state accumulation.
- Always recompute missing fields and questions from scratch.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    # Insert newline before dash bullets if missing
    text = re.sub(r"(?<!\n)-\s", r"\n- ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_bullet_features(text: str) -> list[str]:
    bullets = re.findall(r"-\s*([^-.][^-\n]+)", text)
    clean: list[str] = []
    for b in bullets:
        sentence = b.split(".", 1)[0].strip()
        if 3 < len(sentence) < 120:
            clean.append(sentence)
    return clean


def detect_frontend(text: str):
    if "Next.js" in text:
        return "Next.js"
    if "React Native" in text:
        return "React Native"
    return None


def detect_backend(text: str):
    if "FastAPI" in text:
        return "FastAPI"
    if "Node.js" in text:
        return "Node.js"
    if "Django" in text:
        return "Django"
    return None


def detect_database(text: str):
    if "PostgreSQL" in text:
        return "PostgreSQL"
    if "MongoDB" in text:
        return "MongoDB"
    if "MySQL" in text:
        return "MySQL"
    return None


def detect_infra(text: str):
    if "AWS" in text:
        return "AWS"
    if "GCP" in text:
        return "GCP"
    if "Azure" in text:
        return "Azure"
    return None


def build_structured_partial(raw_text: str) -> dict:
    normalized = normalize_text(raw_text)
    features = extract_bullet_features(raw_text)
    structured = {
        "project_name": None,
        "problem_statement": None,
        "target_users": None,
        "desired_features": features,
        "architecture_decisions": {
            "authentication_strategy": None,
            "authorization_rbac_model": None,
            "data_sync_strategy": None,
            "offline_support": None,
            "currency_handling": None,
            "multi_platform_support": None,
            "monitoring_and_logging": None,
        },
        "tech_stack": {
            "frontend_choice": detect_frontend(normalized),
            "backend_choice": detect_backend(normalized),
            "database_choice": detect_database(normalized),
            "infra_provider": detect_infra(normalized),
            "deployment_strategy": None,
        },
        "non_functional": {
            "security_requirements": None,
            "performance_goals": None,
            "compliance_requirements": None,
        },
        "success_metrics": None,
        "constraints": {"hard_constraints": None},
        "out_of_scope": None,
    }
    return structured


def is_low_quality(value: str) -> bool:
    if value is None:
        return True
    if len(value.strip()) < 5:
        return True
    lower = value.lower()
    if "should be" in lower or "will be built using" in lower:
        return True
    if len(value) > 200:
        return True
    if lower.count(".") > 1:
        return True
    for vague in ["tbd", "later", "maybe"]:
        if vague in lower:
            return True
    return False


def validate_schema(structured: dict) -> list[str]:
    missing = set()

    def _check(path: str, value):
        if value is None:
            missing.add(path)
        elif isinstance(value, str) and is_low_quality(value):
            missing.add(path)
        elif isinstance(value, list) and len(value) == 0:
            missing.add(path)

    _check("project_name", structured.get("project_name"))
    _check("problem_statement", structured.get("problem_statement"))
    _check("target_users", structured.get("target_users"))
    _check("desired_features", structured.get("desired_features"))

    arch = structured.get("architecture_decisions", {})
    _check("architecture_decisions.authentication_strategy", arch.get("authentication_strategy"))
    _check("architecture_decisions.authorization_rbac_model", arch.get("authorization_rbac_model"))
    _check("architecture_decisions.data_sync_strategy", arch.get("data_sync_strategy"))
    _check("architecture_decisions.offline_support", arch.get("offline_support"))
    _check("architecture_decisions.currency_handling", arch.get("currency_handling"))
    _check("architecture_decisions.multi_platform_support", arch.get("multi_platform_support"))
    _check("architecture_decisions.monitoring_and_logging", arch.get("monitoring_and_logging"))

    tech = structured.get("tech_stack", {})
    _check("tech_stack.frontend_choice", tech.get("frontend_choice"))
    _check("tech_stack.backend_choice", tech.get("backend_choice"))
    _check("tech_stack.database_choice", tech.get("database_choice"))
    _check("tech_stack.infra_provider", tech.get("infra_provider"))
    _check("tech_stack.deployment_strategy", tech.get("deployment_strategy"))

    nonfunc = structured.get("non_functional", {})
    _check("non_functional.security_requirements", nonfunc.get("security_requirements"))
    _check("non_functional.performance_goals", nonfunc.get("performance_goals"))
    _check("non_functional.compliance_requirements", nonfunc.get("compliance_requirements"))

    _check("success_metrics", structured.get("success_metrics"))
    constraints = structured.get("constraints", {})
    _check("constraints.hard_constraints", constraints.get("hard_constraints"))
    _check("out_of_scope", structured.get("out_of_scope"))

    return sorted(missing)


FIELD_QUESTION_MAP = {
    "project_name": "What is the name of the product?",
    "problem_statement": "What problem does the product solve?",
    "target_users": "Who are the primary user personas?",
    "desired_features": "What are the core features required?",
    "architecture_decisions.authentication_strategy": "How will users authenticate?",
    "architecture_decisions.authorization_rbac_model": "What roles and permissions are required?",
    "architecture_decisions.data_sync_strategy": "How should data sync across devices?",
    "architecture_decisions.offline_support": "Do you need offline support? If yes, how?",
    "architecture_decisions.currency_handling": "How should currencies be handled?",
    "architecture_decisions.multi_platform_support": "Which platforms must be supported?",
    "architecture_decisions.monitoring_and_logging": "What monitoring and logging is required?",
    "tech_stack.frontend_choice": "What frontend technology will be used?",
    "tech_stack.backend_choice": "What backend framework will be used?",
    "tech_stack.database_choice": "Which database will be used?",
    "tech_stack.infra_provider": "Which cloud provider will be used?",
    "tech_stack.deployment_strategy": "How will you deploy and release changes?",
    "non_functional.security_requirements": "What security requirements are mandatory?",
    "non_functional.performance_goals": "What performance targets must the system meet?",
    "non_functional.compliance_requirements": "Are there compliance requirements?",
    "success_metrics": "How will success be measured?",
    "constraints.hard_constraints": "What constraints are non-negotiable?",
    "out_of_scope": "What is explicitly out of scope?",
}


def generate_questions(missing_fields: list[str]) -> list[str]:
    questions = []
    for field in missing_fields:
        question = FIELD_QUESTION_MAP.get(field, f"Please provide details for {field.replace('_', ' ')}.")
        questions.append(question)
    # Deduplicate + sorted output
    return sorted(set(questions))


def recompute_state(structured: dict) -> dict:
    missing = validate_schema(structured)
    questions = generate_questions(missing)
    return {
        "missing_fields": sorted(set(missing)),
        "questions": sorted(set(questions)),
    }


def compile_requirements(raw_text: str) -> dict:
    structured = build_structured_partial(raw_text)
    state = recompute_state(structured)
    return {
        "structured_partial": structured,
        "missing_fields": state["missing_fields"],
        "questions": state["questions"],
    }
