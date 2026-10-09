import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingestion.pipeline import (
    IngestionError,
    is_uncaptioned_image,
    should_chunk,
    add_attachments,
    create_capture,
    delete_capture,
    finish_attachment_chunking,
    finish_image_captioning,
    finish_structuring,
    finish_unit_processing,
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


@router.post("", response_model=CaptureOut, status_code=201)
async def create(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    metadata: str | None = Form(default=None),
    disable_followup_questions: bool = Form(default=False),
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
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
    uploads = [(f.filename or "upload", await f.read()) for f in files if f.filename]
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
            db, llm, text=text, uploads=uploads, explicit_metadata=explicit_metadata,
            allow_clarification=not disable_followup_questions,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Any unit that isn't waiting on a clarification question reaches
    # status="processing" rather than "ready" straight away -- the
    # background step now only needs one embedding call (semantic
    # fields were already produced by the single up-front analysis
    # call), but stays backgrounded for robustness/consistency with the
    # rest of this pipeline; see docs/DECISIONS.md.
    #
    # finish_structuring is scheduled right after, for the SAME unit,
    # on the SAME BackgroundTasks object -- Starlette runs these
    # sequentially in registration order, so by the time structuring
    # actually runs the unit is already "ready" with real semantics to
    # deepen (see docs/DECISIONS.md "The 'membase structurer'"). This is
    # the "start for any memory as soon as it is stored" trigger -- no
    # polling, no queue, no new infrastructure.
    for unit in capture.units:
        if unit.status == "processing":
            background_tasks.add_task(finish_unit_processing, unit.id)
            background_tasks.add_task(finish_structuring, unit.id)
    # Image captioning and chunk-level indexing are both independent of
    # segmentation/clarification and of each other -- scheduled here as
    # separate background tasks, never inline (see docs/DECISIONS.md
    # "Decouple image captioning" / "offloading to background processing").
    # Chunking is skipped entirely for attachments too small to need it.
    for attachment in capture.attachments:
        if is_uncaptioned_image(attachment):
            background_tasks.add_task(finish_image_captioning, attachment.id)
        elif should_chunk(attachment.extracted_text):
            background_tasks.add_task(finish_attachment_chunking, attachment.id)
    return capture


@router.get("", response_model=list[CaptureOut])
async def list_captures(db: Session = Depends(get_db)):
    """Newest first, no pagination -- a single-user localhost MVP's
    capture volume doesn't warrant it yet (see docs/TODO.md)."""
    captures = db.scalars(select(Capture).order_by(Capture.created_at.desc())).all()
    return list(captures)


@router.get("/{capture_id}", response_model=CaptureOut)
async def get(capture_id: uuid.UUID, db: Session = Depends(get_db)):
    return _get_capture_or_404(db, capture_id)


@router.delete("/{capture_id}", status_code=204)
async def delete(capture_id: uuid.UUID, db: Session = Depends(get_db)):
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
    uploads = [(f.filename or "upload", await f.read()) for f in files if f.filename]
    if not uploads:
        raise HTTPException(status_code=400, detail="No files provided")
    try:
        return await add_attachments(db, llm, capture, uploads)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
