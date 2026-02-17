from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TypedDict

from bson import ObjectId
from langgraph.graph import END, StateGraph

from app.db.mongo import get_db
from app.schemas.requirements import PRDSchema, RequirementsPartial
from app.utils.question_builder import build_questions

logger = logging.getLogger("decisionvault.requirements")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RequirementsState(TypedDict):
    raw_text: str | None
    structured: dict
    answers: dict
    questions: list[str]
    missing_fields: list[str]
    low_quality_fields: list[str]
    prd: dict | None


REQUIRED_FIELDS = [
    "project_name",
    "problem_statement",
    "target_users",
    "desired_features",
    "architecture_decisions.authentication_strategy",
    "architecture_decisions.authorization_rbac_model",
    "architecture_decisions.data_sync_strategy",
    "architecture_decisions.offline_support",
    "architecture_decisions.currency_handling",
    "architecture_decisions.multi_platform_support",
    "architecture_decisions.monitoring_and_logging",
    "tech_stack.frontend_choice",
    "tech_stack.backend_choice",
    "tech_stack.database_choice",
    "tech_stack.infra_region",
    "tech_stack.deployment_strategy",
    "non_functional.security_requirements",
    "non_functional.performance_goals",
    "non_functional.compliance_requirements",
    "success_metrics",
    "constraints.hard_constraints",
    "out_of_scope",
]


def _get_nested(structured: dict, path: str):
    current = structured
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def sanitize_field(value: str) -> str:
    # Normalize instead of rejecting: trim + collapse spaces.
    text = value.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_problem_statement(value: str | None) -> bool:
    if not value:
        return False
    if len(value.strip()) < 60:
        return False
    return True


def is_valid_project_name(name: str | None) -> bool:
    return name is not None and len(name.strip()) >= 3


PLACEHOLDER_TOKENS = {"tbd", "later", "maybe", "to be decided", "to be determined", "n/a", "unknown"}
ENUM_FIELDS = {
    "project_name",
    "tech_stack.frontend_choice",
    "tech_stack.backend_choice",
    "tech_stack.database_choice",
    "tech_stack.infra_region",
    "architecture_decisions.multi_platform_support",
}
DESCRIPTIVE_FIELDS = {
    "problem_statement",
    "non_functional.security_requirements",
    "non_functional.performance_goals",
    "architecture_decisions.data_sync_strategy",
}


def is_low_quality(field: str | None, value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = sanitize_field(value)
        if not text:
            return True
        if text.lower() in PLACEHOLDER_TOKENS:
            return True
        if field in ENUM_FIELDS:
            return False
        if field in DESCRIPTIVE_FIELDS:
            return len(text.strip()) < 40
        return False
    if isinstance(value, list):
        return len(value) == 0
    return False


def validate_structured(structured: dict) -> tuple[list[str], list[str]]:
    missing = set()
    low_quality = set()
    for field in REQUIRED_FIELDS:
        value = _get_nested(structured, field)
        if field == "project_name":
            if not is_valid_project_name(value if isinstance(value, str) else None):
                missing.add(field)
            continue
        if field == "problem_statement":
            if not is_valid_problem_statement(value if isinstance(value, str) else None):
                missing.add(field)
            continue
        if value is None:
            missing.add(field)
            continue
        if isinstance(value, list) and len(value) == 0:
            missing.add(field)
            continue
        if is_low_quality(field, value):
            low_quality.add(field)
    return sorted(missing), sorted(low_quality)


def compute_ready_for_prd(structured: dict, missing_fields: list[str], low_quality_fields: list[str]) -> bool:
    return len(missing_fields) == 0 and len(low_quality_fields) == 0


def _coerce_to_prd(structured: dict) -> PRDSchema:
    return PRDSchema.model_validate(structured)


def _extract_list(text: str) -> list[str] | None:
    if not text:
        return None
    parts = [item.strip() for item in re.split(r"[;\n•,-]+", text) if item.strip()]
    return parts or None


def _extract_field(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            if match.lastindex:
                return match.group(1).strip()
            return match.group(0).strip()
    return None


async def parse_raw_requirements(state: RequirementsState) -> RequirementsState:
    text = state["raw_text"] or ""
    structured = RequirementsPartial().model_dump()
    structured["project_name"] = _extract_field(
        [r"project name[:\s]+(.+)", r"^([A-Za-z0-9 _-]+)\s+version"], text
    )
    structured["problem_statement"] = _extract_field(
        [r"problem[:\s]+(.+)", r"primary goal is to\s+(.+?)(?:\.\s|$)"], text
    )
    structured["target_users"] = _extract_list(
        _extract_field([r"target users?[:\s]+(.+)", r"for\s+(.+?)\s+by enabling"], text)
        or ""
    )
    structured["desired_features"] = _extract_list(
        _extract_field([r"features?[:\s]+(.+)", r"must allow users to\s+(.+?)\."], text) or ""
    )
    structured["architecture_decisions"] = {
        "authentication_strategy": _extract_field([r"authentication[:\s]+(.+)"], text),
        "authorization_rbac_model": _extract_field([r"rbac[:\s]+(.+)", r"authorization[:\s]+(.+)"], text),
        "data_sync_strategy": _extract_field([r"sync(?:hronization)?[:\s]+(.+)"], text),
        "offline_support": _extract_field([r"offline support[:\s]+(.+)"], text),
        "currency_handling": _extract_field([r"currency[:\s]+(.+)"], text),
        "multi_platform_support": _extract_field([r"multi[-\s]platform[:\s]+(.+)", r"mobile application"], text),
        "monitoring_and_logging": _extract_field([r"monitoring[:\s]+(.+)", r"logging[:\s]+(.+)"], text),
    }
    structured["tech_stack"] = {
        "frontend_choice": _extract_field([r"frontend[:\s]+(.+)"], text),
        "backend_choice": _extract_field([r"backend[:\s]+(.+)"], text),
        "database_choice": _extract_field([r"database[:\s]+(.+)"], text),
        "infra_region": _extract_field([r"region[:\s]+(.+)", r"infra(?:structure)?[:\s]+(.+)"], text),
        "deployment_strategy": _extract_field([r"deployment[:\s]+(.+)"], text),
    }
    structured["non_functional"] = {
        "security_requirements": _extract_field([r"security[^.]*\.\s(.+?)\."], text)
        or _extract_field([r"security[:\s]+(.+)"], text),
        "performance_goals": _extract_field([r"performance[^.]*\.\s(.+?)\."], text)
        or _extract_field([r"load within\s+(\d+\s+seconds?)"], text),
        "compliance_requirements": _extract_field([r"compliance[:\s]+(.+)"], text),
    }
    structured["success_metrics"] = _extract_list(
        _extract_field([r"success will be measured by\s+(.+?)\."], text) or ""
    )
    structured["constraints"] = {
        "hard_constraints": _extract_list(_extract_field([r"constraints?[:\s]+(.+)"], text) or ""),
    }
    structured["out_of_scope"] = _extract_list(_extract_field([r"out of scope[:\s]+(.+)"], text) or "")
    return {**state, "structured": structured}


async def clarification_question_generator(state: RequirementsState) -> RequirementsState:
    # Questions are computed in complete_check to avoid any accumulation.
    return state


def deep_merge_structured(existing: dict, answers: dict) -> dict:
    updated = dict(existing)
    for key, value in answers.items():
        if "." in key:
            root, leaf = key.split(".", 1)
            updated.setdefault(root, {})
            if isinstance(updated[root], dict):
                updated[root][leaf] = value
        else:
            updated[key] = value
    return updated


async def answer_collector(state: RequirementsState) -> RequirementsState:
    structured = deep_merge_structured(state["structured"], state["answers"] or {})
    structured = _sanitize_structured(structured)
    return {**state, "structured": structured}


async def complete_check(state: RequirementsState) -> RequirementsState:
    sanitized = _sanitize_structured(state["structured"])
    missing, low_quality = validate_structured(sanitized)
    questions = build_questions(sorted(set(missing + low_quality)))
    return {
        **state,
        "structured": sanitized,
        "missing_fields": missing,
        "low_quality_fields": low_quality,
        "questions": sorted(set(questions)),
    }


async def prd_composer(state: RequirementsState) -> RequirementsState:
    if state["missing_fields"] or state["low_quality_fields"]:
        raise ValueError(
            f"Missing required fields: {state['missing_fields']}; Low quality fields: {state['low_quality_fields']}"
        )
    prd = _coerce_to_prd(state["structured"])
    return {**state, "prd": prd.model_dump()}


def build_intake_graph():
    graph = StateGraph(RequirementsState)
    graph.add_node("ParseRawInput", parse_raw_requirements)
    graph.add_node("ValidateStructure", complete_check)
    graph.add_node("ClarificationQuestionGenerator", clarification_question_generator)

    graph.set_entry_point("ParseRawInput")
    graph.add_edge("ParseRawInput", "ValidateStructure")
    graph.add_edge("ValidateStructure", "ClarificationQuestionGenerator")
    graph.add_edge("ClarificationQuestionGenerator", END)
    return graph.compile()


def build_respond_graph():
    graph = StateGraph(RequirementsState)
    graph.add_node("AcceptAnswers", answer_collector)
    graph.add_node("CompleteCheck", complete_check)
    graph.add_node("ClarificationQuestionGenerator", clarification_question_generator)

    graph.set_entry_point("AcceptAnswers")
    graph.add_edge("AcceptAnswers", "CompleteCheck")
    graph.add_edge("CompleteCheck", "ClarificationQuestionGenerator")
    graph.add_edge("ClarificationQuestionGenerator", END)
    return graph.compile()


def build_generate_graph():
    graph = StateGraph(RequirementsState)
    graph.add_node("CompleteCheck", complete_check)
    graph.add_node("PRDComposer", prd_composer)

    graph.set_entry_point("CompleteCheck")
    graph.add_edge("CompleteCheck", "PRDComposer")
    graph.add_edge("PRDComposer", END)
    return graph.compile()


async def start_intake(tenant_id: str, project_id: str, raw_text: str) -> dict:
    graph = build_intake_graph()
    state: RequirementsState = {
        "raw_text": raw_text,
        "structured": {},
        "answers": {},
        "questions": [],
        "missing_fields": [],
        "low_quality_fields": [],
        "prd": None,
    }
    result = await graph.ainvoke(state)
    intake = {
        "tenant_id": ObjectId(tenant_id),
        "project_id": ObjectId(project_id),
        "raw_text": raw_text,
        "structured": result["structured"],
        "questions": result["questions"],
        "missing_fields": result["missing_fields"],
        "low_quality_fields": result.get("low_quality_fields", []),
        "status": "needs_info" if result["missing_fields"] or result.get("low_quality_fields") else "ready",
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
        "version": 1,
        "cursor": 1,
    }
    db = get_db()
    existing = await db.requirements_intakes.find_one(
        {"tenant_id": ObjectId(tenant_id), "project_id": ObjectId(project_id)}
    )
    if existing:
        # overwrite current, and append history version
        last = await db.requirements_history.find_one(
            {"tenant_id": ObjectId(tenant_id), "project_id": ObjectId(project_id)},
            sort=[("version", -1)],
        )
        version = (last.get("version") if last else 0) + 1
        await db.requirements_history.insert_one(
            {
                "tenant_id": ObjectId(tenant_id),
                "project_id": ObjectId(project_id),
                "version": version,
                "structured": intake["structured"],
                "raw_text": raw_text,
                "created_at": _utcnow(),
            }
        )
        await db.requirements_intakes.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "raw_text": raw_text,
                    "structured": intake["structured"],
                    "questions": intake["questions"],
                    "missing_fields": intake["missing_fields"],
                    "low_quality_fields": intake["low_quality_fields"],
                    "status": intake["status"],
                    "updated_at": _utcnow(),
                    "version": version,
                    "cursor": version,
                }
            },
        )
        intake_id = str(existing["_id"])
    else:
        insert = await db.requirements_intakes.insert_one(intake)
        intake_id = str(insert.inserted_id)
        last = await db.requirements_history.find_one(
            {"tenant_id": ObjectId(tenant_id), "project_id": ObjectId(project_id)},
            sort=[("version", -1)],
        )
        version = (last.get("version") if last else 0) + 1
        await db.requirements_history.insert_one(
            {
                "tenant_id": ObjectId(tenant_id),
                "project_id": ObjectId(project_id),
                "version": version,
                "structured": intake["structured"],
                "raw_text": raw_text,
                "created_at": _utcnow(),
            }
        )
    return {
        "intake_id": intake_id,
        "structured_partial": result["structured"],
        "missing_fields": result["missing_fields"],
        "low_quality_fields": result.get("low_quality_fields", []),
        "questions": result["questions"],
    }


async def respond_intake(intake_id: str, answers: dict) -> dict:
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise ValueError("Intake not found")
    graph = build_respond_graph()
    state: RequirementsState = {
        "raw_text": intake.get("raw_text"),
        "structured": intake.get("structured", {}),
        "answers": answers,
        "questions": [],
        "missing_fields": [],
        "low_quality_fields": [],
        "prd": None,
    }
    result = await graph.ainvoke(state)
    status = "needs_info" if result["missing_fields"] or result.get("low_quality_fields") else "ready"
    last = await db.requirements_history.find_one(
        {"tenant_id": intake["tenant_id"], "project_id": intake["project_id"]},
        sort=[("version", -1)],
    )
    version = (last.get("version") if last else 0) + 1
    await db.requirements_intakes.update_one(
        {"_id": ObjectId(intake_id)},
        {
            "$set": {
                "structured": result["structured"],
                "questions": result["questions"],
                "missing_fields": result["missing_fields"],
                "low_quality_fields": result.get("low_quality_fields", []),
                "status": status,
                "updated_at": _utcnow(),
                "version": version,
                "cursor": version,
            }
        },
    )
    await db.requirements_history.insert_one(
        {
            "tenant_id": intake["tenant_id"],
            "project_id": intake["project_id"],
            "version": version,
            "structured": result["structured"],
            "created_at": _utcnow(),
        }
    )
    return {
        "structured_partial": result["structured"],
        "missing_fields": result["missing_fields"],
        "low_quality_fields": result.get("low_quality_fields", []),
        "questions": result["questions"],
        "ready_for_prd": compute_ready_for_prd(
            result["structured"], result["missing_fields"], result.get("low_quality_fields", [])
        ),
    }


async def generate_prd(intake_id: str) -> dict:
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise ValueError("Intake not found")
    graph = build_generate_graph()
    state: RequirementsState = {
        "raw_text": intake.get("raw_text"),
        "structured": intake.get("structured", {}),
        "answers": {},
        "questions": [],
        "missing_fields": [],
        "low_quality_fields": [],
        "prd": None,
    }
    result = await graph.ainvoke(state)
    await db.requirements_intakes.update_one(
        {"_id": ObjectId(intake_id)},
        {"$set": {"prd": result["prd"], "status": "completed", "updated_at": _utcnow()}},
    )
    return result["prd"]


async def undo_intake(tenant_id: str, project_id: str) -> dict:
    db = get_db()
    intake = await db.requirements_intakes.find_one(
        {"tenant_id": ObjectId(tenant_id), "project_id": ObjectId(project_id)}
    )
    if not intake:
        raise ValueError("Intake not found")
    cursor = int(intake.get("cursor", 1))
    if cursor <= 1:
        return intake
    new_cursor = cursor - 1
    snapshot = await db.requirements_history.find_one(
        {"tenant_id": intake["tenant_id"], "project_id": intake["project_id"], "version": new_cursor}
    )
    if not snapshot:
        raise ValueError("History not found")
    await db.requirements_intakes.update_one(
        {"_id": intake["_id"]},
        {"$set": {"structured": snapshot["structured"], "cursor": new_cursor, "updated_at": _utcnow()}},
    )
    return snapshot["structured"]


async def redo_intake(tenant_id: str, project_id: str) -> dict:
    db = get_db()
    intake = await db.requirements_intakes.find_one(
        {"tenant_id": ObjectId(tenant_id), "project_id": ObjectId(project_id)}
    )
    if not intake:
        raise ValueError("Intake not found")
    cursor = int(intake.get("cursor", 1))
    version = int(intake.get("version", 1))
    if cursor >= version:
        return intake
    new_cursor = cursor + 1
    snapshot = await db.requirements_history.find_one(
        {"tenant_id": intake["tenant_id"], "project_id": intake["project_id"], "version": new_cursor}
    )
    if not snapshot:
        raise ValueError("History not found")
    await db.requirements_intakes.update_one(
        {"_id": intake["_id"]},
        {"$set": {"structured": snapshot["structured"], "cursor": new_cursor, "updated_at": _utcnow()}},
    )
    return snapshot["structured"]
def _sanitize_structured(structured: dict) -> dict:
    updated = structured
    for field in REQUIRED_FIELDS:
        value = _get_nested(updated, field)
        if isinstance(value, str):
            parts = field.split(".")
            if len(parts) == 1:
                updated[parts[0]] = sanitize_field(value)
            else:
                cur = updated
                for part in parts[:-1]:
                    if not isinstance(cur, dict):
                        cur = None
                        break
                    cur = cur.get(part)
                if isinstance(cur, dict):
                    cur[parts[-1]] = sanitize_field(value)
        if isinstance(value, list):
            normalized = []
            for item in value:
                if isinstance(item, str):
                    cleaned = sanitize_field(item)
                    if field == "target_users" and cleaned:
                        cleaned = cleaned[0].upper() + cleaned[1:]
                    normalized.append(cleaned)
                else:
                    normalized.append(item)
            parts = field.split(".")
            if len(parts) == 1:
                updated[parts[0]] = normalized
            else:
                cur = updated
                for part in parts[:-1]:
                    if not isinstance(cur, dict):
                        cur = None
                        break
                    cur = cur.get(part)
                if isinstance(cur, dict):
                    cur[parts[-1]] = normalized
    return updated
