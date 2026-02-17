from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from app.db.mongo import get_db

from app.middleware.guard import withGuard
from app.schemas.requirements import (
    RequirementsGenerateRequest,
    RequirementsGenerateResponse,
    RequirementsRespondRequest,
    RequirementsRespondResponse,
    RequirementsStartRequest,
    RequirementsStartResponse,
)
from app.services.requirements_service import generate_prd, respond_intake, start_intake, undo_intake, redo_intake
from app.services.prd_service import generate_prd as generate_prd_text, save_prd
from app.services.system_design_service import generate_system_design


router = APIRouter(prefix="/api/requirements", tags=["requirements"])


@router.post("/start", response_model=RequirementsStartResponse)
async def start_requirements(
    payload: RequirementsStartRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id query parameter is required")
        resolved_project_id = project_id
        result = await start_intake(user.get("tenant_id"), resolved_project_id, payload.raw_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.post("/respond", response_model=RequirementsRespondResponse)
async def respond_requirements(
    payload: RequirementsRespondRequest,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        result = await respond_intake(payload.intake_id, payload.answers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@router.post("/generate", response_model=RequirementsGenerateResponse)
async def generate_requirements(
    payload: RequirementsGenerateRequest,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        result = await generate_prd(payload.intake_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"prd": result}


@router.post("/{intake_id}/generate-prd")
async def generate_full_prd(
    intake_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    if missing or low_quality:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requirement not complete. Cannot generate PRD.",
                "missing_fields": missing,
                "low_quality_fields": low_quality,
            },
        )
    structured = intake.get("structured") or {}
    try:
        content = generate_prd_text(structured)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    saved = await save_prd(intake_id, content)
    return {"content": content, "version": saved["version"]}


@router.post("/{intake_id}/generate-system-design")
async def generate_system_design_doc(
    intake_id: str,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    if missing or low_quality:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requirement not complete. Cannot generate system design.",
                "missing_fields": missing,
                "low_quality_fields": low_quality,
            },
        )
    structured = intake.get("structured") or {}
    content = generate_system_design(structured)
    return {"content": content}


@router.get("/{intake_id}")
async def get_requirements_status(
    intake_id: str,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    intake = await db.requirements_intakes.find_one({"_id": ObjectId(intake_id)})
    if not intake:
        raise HTTPException(status_code=404, detail="Intake not found")
    missing = intake.get("missing_fields") or []
    low_quality = intake.get("low_quality_fields") or []
    return {
        "structured_partial": intake.get("structured") or {},
        "missing_fields": missing,
        "low_quality_fields": low_quality,
        "questions": intake.get("questions") or [],
        "ready_for_prd": len(missing) == 0 and len(low_quality) == 0,
    }


@router.get("/{intake_id}/prd")
async def get_prd(
    intake_id: str,
    project_id: str | None = None,
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    db = get_db()
    doc = await db.prd_documents.find_one({"intake_id": ObjectId(intake_id)}, sort=[("version", -1)])
    if not doc:
        raise HTTPException(status_code=404, detail="PRD not found")
    return {
        "content": doc.get("content"),
        "version": doc.get("version"),
        "generated_at": doc.get("generated_at"),
    }


@router.post("/undo")
async def undo_requirements(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        structured = await undo_intake(user.get("tenant_id"), project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"structured_partial": structured}


@router.post("/redo")
async def redo_requirements(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id query parameter is required")
    try:
        structured = await redo_intake(user.get("tenant_id"), project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"structured_partial": structured}
