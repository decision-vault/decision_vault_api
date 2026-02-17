"""
User-friendly clarifying question builder.
UX principles:
- Clarity: ask about intent, not schema terms.
- Succinctness: short, direct questions.
- No jargon: avoid words like "string", "list", "schema".
"""

from __future__ import annotations


QUESTION_MAP = {
    "project_name": "What is the project name?",
    "problem_statement": "What problem are we solving?",
    "target_users": "Who are the target users or personas?",
    "desired_features": "What are the key features we must deliver?",
    "authentication_strategy": "How should users authenticate?",
    "authorization_rbac_model": "What access control or role model should we use?",
    "data_sync_strategy": "How should data sync across devices or systems?",
    "offline_support": "Should the product work offline? If yes, how?",
    "currency_handling": "How should currencies be handled?",
    "multi_platform_support": "Which platforms must be supported?",
    "monitoring_and_logging": "What monitoring and logging do we need?",
    "frontend_choice": "What frontend technology will we use?",
    "backend_choice": "What backend language or framework will we use?",
    "database_choice": "What database will we use?",
    "infra_region": "Which region will the infrastructure be deployed in?",
    "deployment_strategy": "How will we deploy and release changes?",
    "security_requirements": "What security requirements are mandatory?",
    "performance_goals": "What performance targets should we meet?",
    "compliance_requirements": "Are there compliance requirements (e.g., SOC2, GDPR)?",
    "success_metrics": "How will we measure success?",
    "hard_constraints": "What constraints are non-negotiable?",
    "out_of_scope": "What is explicitly out of scope?",
}


NESTED_ALIASES = {
    "architecture_decisions.authentication_strategy": "authentication_strategy",
    "architecture_decisions.authorization_rbac_model": "authorization_rbac_model",
    "architecture_decisions.data_sync_strategy": "data_sync_strategy",
    "architecture_decisions.offline_support": "offline_support",
    "architecture_decisions.currency_handling": "currency_handling",
    "architecture_decisions.multi_platform_support": "multi_platform_support",
    "architecture_decisions.monitoring_and_logging": "monitoring_and_logging",
    "tech_stack.frontend_choice": "frontend_choice",
    "tech_stack.backend_choice": "backend_choice",
    "tech_stack.database_choice": "database_choice",
    "tech_stack.infra_region": "infra_region",
    "tech_stack.deployment_strategy": "deployment_strategy",
    "non_functional.security_requirements": "security_requirements",
    "non_functional.performance_goals": "performance_goals",
    "non_functional.compliance_requirements": "compliance_requirements",
    "constraints.hard_constraints": "hard_constraints",
}


def build_questions(missing_fields: list[str]) -> list[str]:
    questions: list[str] = []
    for field in missing_fields:
        key = NESTED_ALIASES.get(field, field.split(".")[-1])
        question = QUESTION_MAP.get(key, f"Can you clarify '{field}'?")
        questions.append(question)
    return questions
