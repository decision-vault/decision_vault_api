from typing import Literal, Union
import logging
import httpx
import asyncio
from datetime import datetime, timezone
from bson import ObjectId
import re
from pathlib import Path
import struct
import zlib

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.middleware.guard import withGuard
from app.schemas.prd_generation import (
    PRDClarificationResponse,
    PRDClarificationRespondRequest,
    PRDGenerateRequest,
    PRDGenerateResponse,
    PRDMultiStepResponse,
)
from app.services.prd_multistep_service import generate_multistep_prd
from app.services.prd_pg_service import (
    get_latest_prd_version,
    get_prd_version,
    list_prd_versions,
    store_prd_version,
)
from app.core.config import settings
from app.services.token_limiter import TokenBudget, TokenLimiter
from langchain_openai import ChatOpenAI
from app.db.mongo import get_db


router = APIRouter(prefix="/api/prd", tags=["prd"])
logger = logging.getLogger("decisionvault.prd.api")


PRIMARY_ORDER = ["project_name", "problem_statement", "target_users", "desired_features"]
TOTAL_REQUIRED_FIELDS = 4
WATERMARK_TEXT = "DecisionVault"
PDF_LOGO_PATH = Path(__file__).resolve().parents[3] / "decision_vault_ui" / "src" / "assets" / "logo.png"
RUN_STALE_TIMEOUT_SECONDS = 120
_ACTIVE_PRD_TASKS: set[asyncio.Task] = set()
_ACTIVE_PRD_TASKS_BY_RUN_ID: dict[str, asyncio.Task] = {}
PRD_STAGE_SEQUENCE = [
    "stage_1_core_context",
    "stage_2_scope_user_stories",
    "stage_3_architecture",
    "stage_4_delivery_quality",
]


def _strip_markdown(md: str) -> str:
    text = md or ""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_right_aligned_x(text: str, font_size: float, right_x: float = 562.0) -> float:
    # Helvetica average glyph width approximation for simple right-alignment.
    estimated_width = max(1.0, len(text)) * font_size * 0.52
    return right_x - estimated_width


def _pdf_centered_x(text: str, font_size: float, page_width: float = 612.0) -> float:
    estimated_width = max(1.0, len(text)) * font_size * 0.52
    return (page_width - estimated_width) / 2.0


def _decode_png_rgba_for_pdf(png_path: Path) -> dict | None:
    if not png_path.exists():
        return None
    data = png_path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None

    pos = 8
    width = height = bit_depth = color_type = None
    idat_parts: list[bytes] = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if not width or not height or bit_depth != 8 or color_type != 6:
        return None

    raw = zlib.decompress(b"".join(idat_parts))
    bpp = 4
    stride = width * bpp
    out = bytearray(height * stride)
    src = 0
    prev = bytearray(stride)

    for row in range(height):
        filter_type = raw[src]
        src += 1
        scan = bytearray(raw[src : src + stride])
        src += stride
        recon = bytearray(stride)

        if filter_type == 0:
            recon[:] = scan
        elif filter_type == 1:
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                recon[i] = (scan[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                up = prev[i]
                recon[i] = (scan[i] + up) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                up = prev[i]
                recon[i] = (scan[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = recon[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                pr = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                recon[i] = (scan[i] + pr) & 0xFF
        else:
            return None

        out[row * stride : (row + 1) * stride] = recon
        prev = recon

    rgb_rows = bytearray()
    alpha_rows = bytearray()
    for row in range(height):
        rgb_rows.append(0)
        alpha_rows.append(0)
        row_data = out[row * stride : (row + 1) * stride]
        for px in range(0, len(row_data), 4):
            rgb_rows.extend(row_data[px : px + 3])
            alpha_rows.append(row_data[px + 3])

    return {
        "width": width,
        "height": height,
        "rgb_compressed": zlib.compress(bytes(rgb_rows), level=9),
        "alpha_compressed": zlib.compress(bytes(alpha_rows), level=9),
    }


def _markdown_to_pdf_bytes(markdown: str, watermark: str = WATERMARK_TEXT) -> bytes:
    plain = _strip_markdown(markdown)
    lines = []
    for line in plain.splitlines():
        if len(line) <= 105:
            lines.append(line)
            continue
        words = line.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 105:
                if current:
                    lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)

    lines_per_page = 45
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[]]
    objects: dict[int, bytes] = {}
    pages_object_index = 2
    font_object_index = 3
    logo_obj_index = 4
    logo_alpha_obj_index = 5
    page_object_indices = []
    content_object_indices = []
    first_dynamic_obj_index = 6

    logo = _decode_png_rgba_for_pdf(PDF_LOGO_PATH)

    for idx, page_lines in enumerate(pages):
        page_obj = first_dynamic_obj_index + idx * 2
        content_obj = page_obj + 1
        page_object_indices.append(page_obj)
        content_object_indices.append(content_obj)

        stream_lines = []
        # Header: centered logo + brand text, with centered subtitle.
        brand_size = 12
        logo_size = 34
        logo_gap = 0
        brand_x = _pdf_centered_x(watermark, brand_size)
        group_left = max(50.0, brand_x - (logo_size + logo_gap))
        brand_x = group_left + logo_size + logo_gap
        stream_lines.append(f"BT /F1 {brand_size} Tf 0.18 g {brand_x:.2f} 768 Td")
        stream_lines.append(f"({_pdf_escape(watermark)}) Tj ET")

        header_subtitle = "Product Requirements Document"
        subtitle_x = _pdf_centered_x(header_subtitle, 11)
        stream_lines.append(f"BT /F1 11 Tf 0.32 g {subtitle_x:.2f} 752 Td")
        stream_lines.append(f"({_pdf_escape(header_subtitle)}) Tj ET")
        stream_lines.append("0.85 g 50 748 m 562 748 l S")

        # Footer: left brand and right page number.
        footer_logo_size = 14
        footer_logo_x = 50
        footer_logo_y = 16
        footer_text_x = footer_logo_x + footer_logo_size
        footer_text_y = 20
        page_label = f"Page {idx + 1}"
        page_x = _pdf_right_aligned_x(page_label, 8, 562.0)
        stream_lines.append("0.85 g 50 34 m 562 34 l S")
        stream_lines.append(f"BT /F1 8 Tf 0.45 g {page_x:.2f} 20 Td")
        stream_lines.append(f"({_pdf_escape(page_label)}) Tj ET")

        # Logo only in header and footer.
        if logo:
            stream_lines.append("q")
            stream_lines.append(f"{logo_size} 0 0 {logo_size} {group_left:.2f} 754 cm")
            stream_lines.append("/ImLogo Do")
            stream_lines.append("Q")
            stream_lines.append("q")
            stream_lines.append(f"{footer_logo_size} 0 0 {footer_logo_size} {footer_logo_x} {footer_logo_y} cm")
            stream_lines.append("/ImLogo Do")
            stream_lines.append("Q")
        stream_lines.append(f"BT /F1 8 Tf 0.45 g {footer_text_x} {footer_text_y} Td")
        stream_lines.append(f"({_pdf_escape(watermark)}) Tj ET")
        # Body text block.
        stream_lines.append("BT /F1 11 Tf 0 g 50 730 Td 14 TL")
        for line in page_lines:
            stream_lines.append(f"({_pdf_escape(line)}) Tj T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        objects[content_obj] = (
            f"{content_obj} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream\nendobj\n"
        )

        xobject = f"/XObject << /ImLogo {logo_obj_index} 0 R >> " if logo else ""
        objects[page_obj] = (
            (
                f"{page_obj} 0 obj\n"
                f"<< /Type /Page /Parent {pages_object_index} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_index} 0 R >> {xobject}>> "
                f"/Contents {content_obj} 0 R >>\n"
                "endobj\n"
            ).encode("latin-1")
        )

    kids = " ".join([f"{pid} 0 R" for pid in page_object_indices])
    objects.update({
        1: b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        2: f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_object_indices)} >>\nendobj\n".encode("latin-1"),
        3: b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    })
    if logo:
        objects[logo_alpha_obj_index] = (
            f"{logo_alpha_obj_index} 0 obj\n"
            f"<< /Type /XObject /Subtype /Image /Width {logo['width']} /Height {logo['height']} "
            "/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
            f"/DecodeParms << /Predictor 15 /Colors 1 /BitsPerComponent 8 /Columns {logo['width']} >> "
            f"/Length {len(logo['alpha_compressed'])} >>\nstream\n".encode("latin-1")
            + logo["alpha_compressed"]
            + b"\nendstream\nendobj\n"
        )
        objects[logo_obj_index] = (
            f"{logo_obj_index} 0 obj\n"
            f"<< /Type /XObject /Subtype /Image /Width {logo['width']} /Height {logo['height']} "
            "/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns {logo['width']} >> "
            f"/SMask {logo_alpha_obj_index} 0 R "
            f"/Length {len(logo['rgb_compressed'])} >>\nstream\n".encode("latin-1")
            + logo["rgb_compressed"]
            + b"\nendstream\nendobj\n"
        )

    max_obj = max(objects.keys())
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (max_obj + 1)

    for obj_id in range(1, max_obj + 1):
        offsets[obj_id] = len(pdf)
        pdf += objects[obj_id]

    xref_pos = len(pdf)
    pdf += f"xref\n0 {max_obj + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for obj_id in range(1, max_obj + 1):
        pdf += f"{offsets[obj_id]:010d} 00000 n \n".encode("latin-1")
    pdf += (
        f"trailer\n<< /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("latin-1")
    )
    return bytes(pdf)


def _markdown_to_doc_bytes(markdown: str, watermark: str = WATERMARK_TEXT) -> bytes:
    plain = _strip_markdown(markdown)

    def esc(value: str) -> str:
        return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    body_lines = plain.splitlines() or ["Insufficient information provided."]
    body = "\\par\n".join([esc(line) for line in body_lines])
    rtf = (
        "{\\rtf1\\ansi\\deff0"
        "{\\fonttbl{\\f0 Arial;}}"
        "{\\header \\pard\\qr\\fs18 "
        + esc(watermark)
        + " - Product Requirements Document\\par}"
        "{\\footer \\pard\\ql\\fs18 "
        + esc(watermark)
        + "\\tab\\tab\\qr Page {\\field{\\*\\fldinst PAGE }}\\par}"
        "\\paperw12240\\paperh15840\\margl1440\\margr1440\\margt1440\\margb1440"
        "\\pard\\qc\\fs56\\cf1 "
        + esc(watermark)
        + "\\par\\pard\\fs22\\ql "
        + body
        + "\\par}"
    )
    return rtf.encode("utf-8", errors="replace")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_oid(value: str, field_name: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}")


def _as_aware_dt(value) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _prd_steps_are_complete(steps: list[dict] | None) -> bool:
    if not isinstance(steps, list):
        return False
    completed = {
        str(step.get("stage")): str(step.get("status") or "").lower()
        for step in steps
        if isinstance(step, dict) and step.get("stage")
    }
    return all(completed.get(stage) == "completed" for stage in PRD_STAGE_SEQUENCE)


def _apply_prd_version_label(markdown: str, version_number: int | str | None) -> str:
    if not markdown:
        return ""
    if version_number in (None, "", 0):
        return markdown
    version_label = f"{version_number}.0" if isinstance(version_number, int) else str(version_number)
    pattern = r"(?m)^\*\*Version:\*\*\s*.+$"
    replacement = f"**Version:** {version_label}"
    if re.search(pattern, markdown):
        return re.sub(pattern, replacement, markdown, count=1)
    return markdown


async def _fallback_prd_documents_for_project(project_id: str, tenant_id: str) -> list[dict]:
    db = get_db()
    tenant_oid = _as_oid(tenant_id, "tenant_id")
    project_oid = _as_oid(project_id, "project_id")

    docs = await db.prd_documents.find(
        {"tenant_id": tenant_oid, "project_id": project_oid}
    ).sort([("generated_at", -1), ("version", -1)]).to_list(length=500)
    if docs:
        return docs

    intake_docs = await db.requirements_intakes.find(
        {"tenant_id": tenant_oid, "project_id": project_oid},
        {"_id": 1},
    ).to_list(length=500)
    intake_ids = [d.get("_id") for d in intake_docs if d.get("_id")]
    if not intake_ids:
        return []
    return await db.prd_documents.find(
        {"intake_id": {"$in": intake_ids}}
    ).sort([("generated_at", -1), ("version", -1)]).to_list(length=500)


def _resolve_llm_stream_config() -> tuple[str, str, str | None]:
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "huggingface":
        model = settings.hf_openai_model or settings.llm_model
        api_key = settings.hf_api_token
        base_url = settings.hf_router_base_url
    elif provider == "lmstudio":
        model = settings.lmstudio_model or settings.llm_model
        api_key = settings.llm_api_key or "lm-studio"
        base_url = settings.lmstudio_base_url
    else:
        model = settings.llm_model or settings.llm_strong_model
        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
    logger.warning(
        "prd_stream_llm_selected provider=%s model=%s base_url=%s",
        provider or "default",
        model,
        base_url,
    )
    return model, api_key, base_url


def _extract_lmstudio_text(payload: dict) -> str:
    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                chunks.append(item["content"].strip())
        if chunks:
            return "\n".join(chunks).strip()
    return ""


def _is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, list):
        return len([item for item in value if isinstance(item, str) and item.strip()]) > 0
    return bool(value)


def _build_clarification(payload: PRDGenerateRequest) -> PRDClarificationResponse | None:
    missing_primary: list[str] = []
    if not _is_filled(payload.title):
        missing_primary.append("project_name")
    if not _is_filled(payload.problem_statement):
        missing_primary.append("problem_statement")
    if not _is_filled(payload.target_users):
        missing_primary.append("target_users")
    cleaned_features = [item.strip() for item in payload.features if item and item.strip()]
    if len(cleaned_features) < 3:
        missing_primary.append("desired_features")

    filled_primary = TOTAL_REQUIRED_FIELDS - len(missing_primary)
    completion_score = round(filled_primary / TOTAL_REQUIRED_FIELDS, 2)

    if missing_primary or completion_score < 0.6:
        questions: list[str] = []
        for field in PRIMARY_ORDER:
            if field == "project_name" and field in missing_primary:
                questions.append("What is the project name?")
            elif field == "problem_statement" and field in missing_primary:
                questions.append("Please provide a clear problem statement (minimum 100 characters).")
            elif field == "target_users" and field in missing_primary:
                questions.append("Who are the target users for this product?")
            elif field == "desired_features" and field in missing_primary:
                questions.append("List at least 3 core desired features.")
        return PRDClarificationResponse(
            status="clarification_required",
            questions=questions[:5],
            completion_score=completion_score,
        )

    # Primary required fields are complete; proceed to PRD generation.
    return None


def _split_feature_text(value: str) -> list[str]:
    tokens = [item.strip() for item in value.replace("\n", ",").split(",")]
    return [item for item in tokens if item]


def _apply_clarification_answers(
    draft: PRDGenerateRequest,
    answers: dict[str, str | list[str]],
) -> PRDGenerateRequest:
    title = draft.title
    problem_statement = draft.problem_statement
    target_users = draft.target_users
    features = list(draft.features)
    additional_notes = draft.additional_notes

    for key, value in answers.items():
        lowered = key.lower()
        if isinstance(value, list):
            value_text = ", ".join([str(v).strip() for v in value if str(v).strip()])
        else:
            value_text = str(value).strip()

        if not value_text:
            continue

        if "project" in lowered and "name" in lowered:
            title = value_text
        elif "problem" in lowered:
            problem_statement = value_text
        elif "target" in lowered and "user" in lowered:
            target_users = value_text
        elif "feature" in lowered:
            parsed = value if isinstance(value, list) else _split_feature_text(value_text)
            features = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            # Keep unknown answers in notes so user context is preserved.
            additional_notes = f"{additional_notes or ''}\n{key}: {value_text}".strip()

    return PRDGenerateRequest(
        title=title,
        problem_statement=problem_statement,
        target_users=target_users,
        features=features,
        additional_notes=additional_notes,
    )


async def _store_clarification_answers(
    *,
    tenant_id: str,
    project_id: str,
    user_id: str,
    draft: PRDGenerateRequest,
    answers: dict[str, str | list[str]],
    merged: PRDGenerateRequest,
) -> None:
    db = get_db()
    now = _utcnow()
    await db.prd_clarifications.update_one(
        {
            "tenant_id": _as_oid(tenant_id, "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
            "user_id": user_id,
        },
        {
            "$set": {
                "draft": draft.model_dump(),
                "answers": answers,
                "merged_payload": merged.model_dump(),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


async def _merge_payload_with_saved_clarification(
    *,
    tenant_id: str,
    project_id: str,
    user_id: str,
    payload: PRDGenerateRequest,
) -> PRDGenerateRequest:
    db = get_db()
    saved = await db.prd_clarifications.find_one(
        {
            "tenant_id": _as_oid(tenant_id, "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
            "user_id": user_id,
        }
    )
    if not saved:
        return payload

    merged_payload = saved.get("merged_payload")
    if isinstance(merged_payload, dict):
        try:
            return PRDGenerateRequest.model_validate(merged_payload)
        except Exception:
            pass

    saved_answers = saved.get("answers")
    if isinstance(saved_answers, dict):
        return _apply_clarification_answers(payload, saved_answers)
    return payload


async def _append_run_event(run_id: ObjectId, event: dict) -> None:
    db = get_db()
    now = _utcnow()
    stage = event.get("stage")
    status = event.get("status")
    if not stage:
        return
    if status == "running":
        # Set run started_at only once (first running stage).
        await db.prd_runs.update_one(
            {"_id": run_id, "started_at": None},
            {"$set": {"started_at": now}},
        )
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "running",
                    "updated_at": now,
                },
                "$push": {
                    "events": {"at": now, **event},
                    "steps": {
                        "stage": stage,
                        "status": "running",
                        "started_at": now,
                    },
                },
            },
        )
        return

    if status in {"completed", "failed"}:
        await db.prd_runs.update_one(
            {"_id": run_id, "steps.stage": stage},
            {
                "$set": {
                    "updated_at": now,
                    "steps.$.status": status,
                    "steps.$.ended_at": now,
                    "steps.$.input_tokens": event.get("input_tokens"),
                    "steps.$.output_tokens": event.get("output_tokens"),
                    "steps.$.retry_count": event.get("retry_count", 0),
                    "steps.$.error": event.get("error"),
                    "steps.$.stage_output": event.get("stage_output"),
                },
                "$push": {"events": {"at": now, **event}},
            },
        )
        return

    await db.prd_runs.update_one(
        {"_id": run_id},
        {"$set": {"updated_at": now}, "$push": {"events": {"at": now, **event}}},
    )


async def _get_run_controls(run_id: ObjectId) -> dict[str, bool]:
    db = get_db()
    doc = await db.prd_runs.find_one({"_id": run_id}, {"pause_requested": 1, "stop_requested": 1})
    if not doc:
        return {"pause": False, "stop": True}
    return {
        "pause": bool(doc.get("pause_requested")),
        "stop": bool(doc.get("stop_requested")),
    }


async def _run_prd_job(
    run_id: ObjectId,
    payload: PRDGenerateRequest,
    project_id: str,
    tenant_id: str,
    created_by: str,
) -> None:
    db = get_db()
    try:
        result = await generate_multistep_prd(
            payload,
            tenant_id=tenant_id,
            project_id=project_id,
            run_id=str(run_id),
            progress_cb=lambda ev: _append_run_event(run_id, ev),
            control_cb=lambda: _get_run_controls(run_id),
        )
        stored = await store_prd_version(
            project_id=project_id,
            created_by=created_by,
            markdown_content=result.prd_markdown,
        )
        steps_complete = False
        for _ in range(5):
            run_doc = await db.prd_runs.find_one({"_id": run_id}, {"steps": 1})
            steps_complete = _prd_steps_are_complete((run_doc or {}).get("steps"))
            if steps_complete:
                break
            await asyncio.sleep(0.2)
        if not steps_complete:
            raise ValueError("PRD run finished content generation but not all stages were recorded as completed.")
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "updated_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "result": {
                        "pages_estimated": result.pages_estimated,
                        "sections_generated": result.sections_generated,
                        "required_sections": result.required_sections,
                        "missing_sections": result.missing_sections,
                        "has_all_required_sections": result.has_all_required_sections,
                        "total_tokens_used": result.total_tokens_used,
                        "prd_markdown": result.prd_markdown,
                        "version": stored.get("version_number"),
                    },
                }
            },
        )
    except RuntimeError as exc:
        # User-initiated stop path.
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "stopped",
                    "updated_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "error": str(exc),
                }
            },
        )
    except asyncio.CancelledError:
        # Cancellation from /stop endpoint should persist as stopped.
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "stopped",
                    "updated_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "error": "Run stopped by user.",
                }
            },
        )
    except Exception as exc:
        logger.exception("prd_run_failed run_id=%s project_id=%s", str(run_id), project_id)
        await db.prd_runs.update_one(
            {"_id": run_id},
            {
                "$set": {
                    "status": "failed",
                    "updated_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "error": str(exc),
                }
            },
        )
    finally:
        _ACTIVE_PRD_TASKS_BY_RUN_ID.pop(str(run_id), None)


async def _enqueue_prd_run(
    *,
    payload: PRDGenerateRequest,
    project_id: str,
    tenant_id: str,
    created_by: str,
) -> dict:
    db = get_db()
    run_id = ObjectId()
    now = _utcnow()
    run_doc = {
        "_id": run_id,
        "tenant_id": _as_oid(tenant_id, "tenant_id"),
        "project_id": _as_oid(project_id, "project_id"),
        "created_by": created_by,
        "status": "queued",
        "request": payload.model_dump(),
        "steps": [],
        "events": [{"at": now, "status": "queued"}],
        "error": None,
        "result": None,
        "pause_requested": False,
        "stop_requested": False,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
    }
    await db.prd_runs.insert_one(run_doc)
    task = asyncio.create_task(
        _run_prd_job(
            run_id=run_id,
            payload=payload,
            project_id=project_id,
            tenant_id=tenant_id,
            created_by=created_by,
        )
    )
    _ACTIVE_PRD_TASKS.add(task)
    _ACTIVE_PRD_TASKS_BY_RUN_ID[str(run_id)] = task
    task.add_done_callback(lambda t: _ACTIVE_PRD_TASKS.discard(t))
    return {"run_id": str(run_id), "status": "queued"}


@router.post("/generate", response_model=Union[PRDGenerateResponse, PRDClarificationResponse])
async def generate_prd(
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    payload = await _merge_payload_with_saved_clarification(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        user_id=user.get("user_id"),
        payload=payload,
    )

    clarification = _build_clarification(payload)
    if clarification and clarification.questions:
        return clarification

    try:
        multi = await generate_multistep_prd(
            payload,
            tenant_id=user.get("tenant_id"),
            project_id=project_id,
            run_id=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("prd_generation_failed", extra={"error": str(exc), "project_id": project_id})
        raise HTTPException(status_code=500, detail=f"PRD generation failed: {str(exc)}")

    try:
        await store_prd_version(
            project_id=project_id,
            created_by=user.get("user_id"),
            markdown_content=multi.prd_markdown,
        )
    except RuntimeError as exc:
        # Postgres is optional for local/dev. PRD generation should still succeed.
        logger.warning("prd_store_skipped_runtime", extra={"error": str(exc), "project_id": project_id})
    except Exception as exc:
        # Do not fail generation because of persistence-only issues.
        logger.exception("prd_store_failed", extra={"error": str(exc), "project_id": project_id})

    return PRDGenerateResponse(
        status="ready_for_prd_generation",
        prd_markdown=multi.prd_markdown,
        confidence_score=0.95,
        sections_generated=multi.sections_generated,
    )


@router.post("/generate-multistep", response_model=PRDMultiStepResponse)
async def generate_prd_multistep(
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    payload = await _merge_payload_with_saved_clarification(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        user_id=user.get("user_id"),
        payload=payload,
    )
    try:
        result = await generate_multistep_prd(
            payload,
            tenant_id=user.get("tenant_id"),
            project_id=project_id,
            run_id=None,
        )
    except ValueError as exc:
        logger.warning(
            "prd_multistep_bad_request project_id=%s tenant_id=%s detail=%s",
            project_id,
            user.get("tenant_id"),
            str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("prd_multistep_failed", extra={"error": str(exc), "project_id": project_id})
        raise HTTPException(status_code=500, detail=f"Multi-step PRD generation failed: {str(exc)}")

    try:
        await store_prd_version(
            project_id=project_id,
            created_by=user.get("user_id"),
            markdown_content=result.prd_markdown,
        )
    except Exception:
        logger.warning("prd_multistep_store_skipped")
    return result


@router.post("/generate-multistep/run")
async def generate_prd_multistep_run(
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    payload = await _merge_payload_with_saved_clarification(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        user_id=user.get("user_id"),
        payload=payload,
    )
    clarification = _build_clarification(payload)
    if clarification and clarification.questions:
        logger.warning(
            "prd_multistep_run_clarification_required project_id=%s tenant_id=%s completion_score=%s",
            project_id,
            user.get("tenant_id"),
            clarification.completion_score,
        )
        return {
            "status": clarification.status,
            "questions": clarification.questions,
            "completion_score": clarification.completion_score,
        }
    return await _enqueue_prd_run(
        payload=payload,
        project_id=project_id,
        tenant_id=user.get("tenant_id"),
        created_by=user.get("user_id"),
    )


@router.get("/runs/{run_id}")
async def get_prd_run_status(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    doc = await db.prd_runs.find_one(
        {
            "_id": _as_oid(run_id, "run_id"),
            "tenant_id": _as_oid(user.get("tenant_id"), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Run not found")

    now = _utcnow()
    run_status = doc.get("status")
    updated_at = _as_aware_dt(doc.get("updated_at")) or _as_aware_dt(doc.get("created_at")) or now

    # Heal orphaned runs that are stuck in queued/running due worker interruption.
    stale_seconds = (now - updated_at).total_seconds()
    if run_status in {"queued", "running"} and stale_seconds > max(RUN_STALE_TIMEOUT_SECONDS, settings.prd_generation_timeout_seconds):
        fail_error = "Run timed out or background worker stopped before completion."
        await db.prd_runs.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "status": "failed",
                    "error": fail_error,
                    "updated_at": now,
                    "completed_at": now,
                }
            },
        )
        await db.prd_runs.update_many(
            {"_id": doc["_id"], "steps.status": "running"},
            {
                "$set": {
                    "steps.$[s].status": "failed",
                    "steps.$[s].ended_at": now,
                    "steps.$[s].error": fail_error,
                }
            },
            array_filters=[{"s.status": "running"}],
        )
        doc = await db.prd_runs.find_one({"_id": doc["_id"]})
        run_status = doc.get("status")

    steps = doc.get("steps", [])
    has_final_result = isinstance(doc.get("result"), dict) and bool((doc.get("result") or {}).get("prd_markdown"))
    if run_status == "completed" and (not has_final_result or not _prd_steps_are_complete(steps)):
        corrected_status = "running" if _as_aware_dt(doc.get("started_at")) else "queued"
        await db.prd_runs.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "status": corrected_status,
                    "updated_at": now,
                    "completed_at": None,
                }
            },
        )
        doc = await db.prd_runs.find_one({"_id": doc["_id"]})
        run_status = doc.get("status")
        steps = doc.get("steps", [])
    step_timings: list[dict] = []
    for step in steps:
        started_at = _as_aware_dt(step.get("started_at"))
        ended_at = _as_aware_dt(step.get("ended_at"))
        duration_seconds: float | None = None
        if started_at and ended_at:
            duration_seconds = round((ended_at - started_at).total_seconds(), 3)
        elif started_at and step.get("status") == "running":
            duration_seconds = round((now - started_at).total_seconds(), 3)
        step_timings.append(
            {
                **step,
                "duration_seconds": duration_seconds,
            }
        )

    run_started_at = _as_aware_dt(doc.get("started_at")) or _as_aware_dt(doc.get("created_at"))
    run_completed_at = _as_aware_dt(doc.get("completed_at"))
    total_elapsed_seconds: float | None = None
    if run_started_at and run_completed_at:
        total_elapsed_seconds = round((run_completed_at - run_started_at).total_seconds(), 3)
    elif run_started_at and doc.get("status") in {"queued", "running"}:
        total_elapsed_seconds = round((now - run_started_at).total_seconds(), 3)

    return {
        "run_id": str(doc["_id"]),
        "status": run_status,
        "steps": step_timings,
        "error": doc.get("error"),
        "result": doc.get("result"),
        "timing": {
            "total_elapsed_seconds": total_elapsed_seconds,
        },
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
    }


@router.post("/runs/{run_id}/pause")
async def pause_prd_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.prd_runs.find_one(
        {
            "_id": oid,
            "tenant_id": _as_oid(user.get("tenant_id"), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.prd_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": True, "status": "paused", "updated_at": _utcnow()}},
    )
    await _append_run_event(oid, {"stage": "run", "status": "paused"})
    return {"run_id": run_id, "status": "paused"}


@router.post("/runs/{run_id}/resume")
async def resume_prd_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.prd_runs.find_one(
        {
            "_id": oid,
            "tenant_id": _as_oid(user.get("tenant_id"), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}
    await db.prd_runs.update_one(
        {"_id": oid},
        {"$set": {"pause_requested": False, "status": "running", "updated_at": _utcnow()}},
    )
    await _append_run_event(oid, {"stage": "run", "status": "running"})
    return {"run_id": run_id, "status": "running"}


@router.post("/runs/{run_id}/stop")
async def stop_prd_run(
    run_id: str,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    oid = _as_oid(run_id, "run_id")
    doc = await db.prd_runs.find_one(
        {
            "_id": oid,
            "tenant_id": _as_oid(user.get("tenant_id"), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
        }
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Run not found")
    if doc.get("status") in {"completed", "failed", "stopped"}:
        return {"run_id": run_id, "status": doc.get("status")}

    now = _utcnow()
    await db.prd_runs.update_one(
        {"_id": oid},
        {
            "$set": {
                "stop_requested": True,
                "pause_requested": False,
                "status": "stopped",
                "error": "Run stopped by user.",
                "updated_at": now,
                "completed_at": now,
            }
        },
    )
    await db.prd_runs.update_many(
        {"_id": oid, "steps.status": "running"},
        {
            "$set": {
                "steps.$[s].status": "failed",
                "steps.$[s].ended_at": now,
                "steps.$[s].error": "Run stopped by user.",
            }
        },
        array_filters=[{"s.status": "running"}],
    )
    task = _ACTIVE_PRD_TASKS_BY_RUN_ID.get(run_id)
    if task and not task.done():
        task.cancel()
    await _append_run_event(oid, {"stage": "run", "status": "stopped", "error": "Run stopped by user."})
    return {"run_id": run_id, "status": "stopped"}


@router.post("/clarification/respond")
async def respond_prd_clarification(
    payload: PRDClarificationRespondRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    merged = _apply_clarification_answers(payload.draft, payload.answers)
    await _store_clarification_answers(
        tenant_id=user.get("tenant_id"),
        project_id=project_id,
        user_id=user.get("user_id"),
        draft=payload.draft,
        answers=payload.answers,
        merged=merged,
    )
    clarification = _build_clarification(merged)
    if clarification and clarification.questions:
        return {
            "status": clarification.status,
            "questions": clarification.questions,
            "completion_score": clarification.completion_score,
        }
    return await _enqueue_prd_run(
        payload=merged,
        project_id=project_id,
        tenant_id=user.get("tenant_id"),
        created_by=user.get("user_id"),
    )


@router.get("/clarification/latest")
async def get_latest_prd_clarification(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    db = get_db()
    doc = await db.prd_clarifications.find_one(
        {
            "tenant_id": _as_oid(user.get("tenant_id"), "tenant_id"),
            "project_id": _as_oid(project_id, "project_id"),
            "user_id": user.get("user_id"),
        }
    )
    if not doc:
        return {"answers": {}, "merged_payload": None, "updated_at": None}
    return {
        "answers": doc.get("answers") or {},
        "merged_payload": doc.get("merged_payload"),
        "updated_at": doc.get("updated_at"),
    }


@router.post("/generate/stream")
async def generate_prd_stream(
    request: Request,
    payload: PRDGenerateRequest,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    clarification = _build_clarification(payload)
    if clarification and clarification.questions:
        raise HTTPException(
            status_code=400,
            detail={
                "status": clarification.status,
                "questions": clarification.questions,
                "completion_score": clarification.completion_score,
            },
        )

    limiter = TokenLimiter(TokenBudget(settings.llm_max_input_tokens, settings.llm_max_output_tokens))
    input_tokens = max(1, int(len(payload.model_dump_json().split()) * 1.3))
    limiter.enforce(input_tokens=input_tokens, output_tokens=settings.llm_max_output_tokens)

    model, api_key, base_url = _resolve_llm_stream_config()
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM API key not configured")
    provider = (settings.llm_provider or "").strip().lower()
    prompt = (
        f"Title: {payload.title}\n"
        f"Problem: {payload.problem_statement}\n"
        f"Target users: {payload.target_users}\n"
        f"Features: {', '.join(payload.features)}\n"
        f"Additional notes: {payload.additional_notes or 'None'}\n\n"
        "Generate PRD markdown. No invented data."
    )

    async def event_stream():
        if provider == "lmstudio":
            url = f"{(settings.lmstudio_base_url or '').rstrip('/')}{settings.lmstudio_chat_path}"
            body = {
                "model": settings.lmstudio_model or model,
                "input": prompt,
                "temperature": 0.2,
            }
            logger.warning(
                "prd_stream_lmstudio_request base_url=%s chat_path=%s model=%s",
                settings.lmstudio_base_url,
                settings.lmstudio_chat_path,
                body["model"],
            )
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"})
                resp.raise_for_status()
                data = resp.json()
            text = _extract_lmstudio_text(data)
            if text:
                logger.warning(
                    "prd_stream_lmstudio_response model=%s output_chars=%s",
                    body["model"],
                    len(text),
                )
                yield text
            return

        llm = ChatOpenAI(
            model=model,
            temperature=0.2,
            top_p=0.8,
            max_tokens=settings.llm_max_output_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        async for chunk in llm.astream(prompt):
            if await request.is_disconnected():
                break
            text = getattr(chunk, "content", "") or ""
            if text:
                yield text

    return StreamingResponse(event_stream(), media_type="text/markdown")


@router.get("/latest")
async def get_latest_prd(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    latest_pg = await get_latest_prd_version(project_id)
    fallback_docs = await _fallback_prd_documents_for_project(project_id, user.get("tenant_id"))
    latest_fb = None
    if fallback_docs:
        fallback = fallback_docs[0]
        latest_fb = {
            "project_id": project_id,
            "version_number": fallback.get("version"),
            "created_by": str(fallback.get("created_by") or user.get("user_id") or ""),
            "created_at": fallback.get("generated_at") or fallback.get("created_at"),
            "markdown_content": fallback.get("content") or "",
        }

    if not latest_pg and not latest_fb:
        raise HTTPException(status_code=404, detail="PRD not found")

    if latest_pg and latest_fb:
        latest = latest_pg if int(latest_pg.get("version_number") or 0) >= int(latest_fb.get("version_number") or 0) else latest_fb
    else:
        latest = latest_pg or latest_fb
    return {
        "project_id": latest["project_id"],
        "version": latest["version_number"],
        "created_by": latest["created_by"],
        "created_at": latest["created_at"],
        "content": _apply_prd_version_label(latest["markdown_content"], latest["version_number"]),
        "source": "llm",
    }


@router.get("/export")
async def export_prd(
    project_id: str | None = None,
    type: Literal["md", "pdf", "doc"] = "md",
    version: int | None = None,
    doc_type: Literal["prd", "sdd"] = "prd",
    _user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})

    if doc_type == "sdd":
        db = get_db()
        tenant_oid = _as_oid(_user.get("tenant_id"), "tenant_id")
        project_oid = _as_oid(project_id, "project_id")
        query: dict = {"tenant_id": tenant_oid, "project_id": project_oid}
        if version is not None:
            query["version"] = version
            doc = await db.system_design_documents.find_one(query, sort=[("generated_at", -1)])
        else:
            doc = await db.system_design_documents.find_one(query, sort=[("version", -1)])
        if not doc:
            raise HTTPException(status_code=404, detail="SDD not found")
        markdown = doc.get("content", "")
        version_number = doc.get("version", "latest")
        base_name = f"decisionvault-sdd-v{version_number}"
    else:
        doc = await get_prd_version(project_id, version) if version else await get_latest_prd_version(project_id)
        if not doc:
            raise HTTPException(status_code=404, detail="PRD not found")
        version_number = doc.get("version_number", "latest")
        markdown = _apply_prd_version_label(doc.get("markdown_content", ""), version_number)
        base_name = f"decisionvault-prd-v{version_number}"

    if type == "md":
        content = markdown.encode("utf-8")
        media_type = "text/markdown; charset=utf-8"
        filename = f"{base_name}.md"
    elif type == "pdf":
        content = _markdown_to_pdf_bytes(markdown)
        media_type = "application/pdf"
        filename = f"{base_name}.pdf"
    else:
        content = _markdown_to_doc_bytes(markdown)
        media_type = "application/msword"
        filename = f"{base_name}.doc"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/versions")
async def get_prd_versions(
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    versions = await list_prd_versions(project_id)
    by_version: dict[int, dict] = {}
    for v in versions:
        num = int(v.get("version_number") or 0)
        if num <= 0:
            continue
        by_version[num] = v

    docs = await _fallback_prd_documents_for_project(project_id, user.get("tenant_id"))
    for d in docs:
        version = d.get("version")
        if version is None:
            continue
        num = int(version)
        existing = by_version.get(num)
        candidate = {
            "project_id": project_id,
            "version_number": num,
            "created_by": str(d.get("created_by") or user.get("user_id") or ""),
            "created_at": d.get("generated_at") or d.get("created_at"),
        }
        if not existing:
            by_version[num] = candidate
            continue
        existing_at = existing.get("created_at") or datetime.min.replace(tzinfo=timezone.utc)
        current_at = candidate.get("created_at") or datetime.min.replace(tzinfo=timezone.utc)
        if current_at > existing_at:
            by_version[num] = candidate

    versions = sorted(by_version.values(), key=lambda x: int(x.get("version_number") or 0), reverse=True)
    return {"items": versions}


@router.get("/versions/{version_number}")
async def get_prd_by_version(
    version_number: int,
    project_id: str | None = None,
    user=Depends(withGuard(feature="edit_decision", projectRole="contributor")),
):
    if not project_id:
        raise HTTPException(status_code=400, detail={"project_id": "project_id query parameter is required"})
    doc = await get_prd_version(project_id, version_number)
    if not doc:
        fallback_docs = await _fallback_prd_documents_for_project(project_id, user.get("tenant_id"))
        matching = [d for d in fallback_docs if int(d.get("version") or -1) == int(version_number)]
        fallback = matching[0] if matching else None
        if not fallback:
            raise HTTPException(status_code=404, detail="PRD version not found")
        doc = {
            "project_id": project_id,
            "version_number": fallback.get("version"),
            "created_by": str(fallback.get("created_by") or user.get("user_id") or ""),
            "created_at": fallback.get("generated_at") or fallback.get("created_at"),
            "markdown_content": fallback.get("content") or "",
        }
    return {
        "project_id": doc["project_id"],
        "version": doc["version_number"],
        "created_by": doc["created_by"],
        "created_at": doc["created_at"],
        "content": _apply_prd_version_label(doc["markdown_content"], doc["version_number"]),
        "source": "llm",
    }
