import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.ingestion.pipeline import (
    IngestionError,
    should_chunk,
    add_attachments,
    create_capture,
    delete_capture,
    finish_attachment_chunking,
    process_capture_fully,
    retry_capture_processing,
)
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.models import Capture
from app.schemas import CaptureOut

router = APIRouter(prefix="/api/captures", tags=["captures"])


def _get_capture_or_404(db: Session, capture_id: uuid.UUID) -> Capture:
    capture = db.get(Capture, capture_id)
    if capture is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Files with no filename are browser artifacts from an empty file
    input, not real uploads -- dropped rather than stored."""
    return [(f.filename or "upload", await f.read()) for f in files if f.filename]


@router.post("", response_model=CaptureOut, status_code=201)
async def create(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    metadata: str | None = Form(default=None),
    disable_followup_questions: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    """`metadata`, if present, is a JSON object of optional explicit
    fields (location/date/people/organizations/topics -- see
    docs/DECISIONS.md "Explicit metadata authority"); malformed JSON is
    treated as if none was given rather than rejecting the whole
    request over an optional field.

    `disable_followup_questions`, if true, removes the clarification
    round-trip from this capture's storage cycle entirely -- see
    docs/DECISIONS.md "Disable follow-up questions".
    """
    uploads = await _read_uploads(files)
    explicit_metadata = {}
    if metadata:
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                explicit_metadata = parsed
        except json.JSONDecodeError:
            pass
    try:
        capture = await create_capture(
            db, text=text, uploads=uploads, explicit_metadata=explicit_metadata,
            allow_clarification=not disable_followup_questions,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Phase 0 only just happened (capture + attachments durably saved,
    # zero LLM calls) -- this response returns now, the instant "saved"
    # acknowledgement, before any model call. ALL LLM-dependent work
    # (image understanding, segmentation, clarification, enrichment,
    # embedding, structuring) happens in this ONE background task,
    # scheduled here and never awaited (see docs/DECISIONS.md "Restore
    # the multi-call ingestion pipeline").
    background_tasks.add_task(process_capture_fully, capture.id)
    # Non-image attachment chunking doesn't depend on process_capture_fully
    # at all (no caption to wait for) -- scheduled independently, same as
    # before. An image attachment is never chunkable yet at this point
    # (no caption exists until process_capture_fully runs), so this loop
    # naturally only ever matches non-image attachments.
    for attachment in capture.attachments:
        if should_chunk(attachment.extracted_text):
            background_tasks.add_task(finish_attachment_chunking, attachment.id)
    return capture


@router.get("", response_model=list[CaptureOut])
def list_captures(db: Session = Depends(get_db)):
    """Newest first, no pagination -- a single-user localhost MVP's
    capture volume doesn't warrant it yet (see docs/TODO.md)."""
    captures = db.scalars(
        select(Capture)
        .options(selectinload(Capture.units), selectinload(Capture.attachments))
        .order_by(Capture.created_at.desc())
    ).all()
    return list(captures)


@router.get("/{capture_id}", response_model=CaptureOut)
def get(capture_id: uuid.UUID, db: Session = Depends(get_db)):
    return _get_capture_or_404(db, capture_id)


@router.delete("/{capture_id}", status_code=204)
def delete(capture_id: uuid.UUID, db: Session = Depends(get_db)):
    capture = _get_capture_or_404(db, capture_id)
    delete_capture(db, capture)


@router.post("/{capture_id}/attachments", response_model=CaptureOut)
async def add_attachment_endpoint(
    capture_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Attach one or more files to an already-saved memory in one call
    (see docs/DECISIONS.md "Multi-file posthoc attachment"). Validation
    is atomic across the batch -- if any file is rejected, none of them
    are attached, rather than leaving a partial mix for the caller to
    reconcile.
    """
    capture = _get_capture_or_404(db, capture_id)
    uploads = await _read_uploads(files)
    if not uploads:
        raise HTTPException(status_code=400, detail="No files provided")
    try:
        return await add_attachments(db, llm, capture, uploads)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{capture_id}/retry", response_model=CaptureOut)
async def retry_capture(
    capture_id: uuid.UUID,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Retry a capture whose process_capture_fully() never finished
    (e.g. the server restarted mid-task -- a scheduled BackgroundTask
    doesn't survive that) or failed outright. Mirrors
    /api/memory-units/{id}/retry's recovery role, one level up."""
    capture = _get_capture_or_404(db, capture_id)
    if capture.processing_status not in ("pending", "processing", "failed"):
        raise HTTPException(status_code=400, detail=f"Capture is '{capture.processing_status}', nothing to retry")
    return await retry_capture_processing(db, llm, capture)
