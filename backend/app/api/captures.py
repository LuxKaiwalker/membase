import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingestion.pipeline import (
    IngestionError,
    add_attachment,
    create_capture,
    delete_capture,
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
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    uploads = [(f.filename or "upload", await f.read()) for f in files if f.filename]
    try:
        capture = await create_capture(db, llm, text=text, uploads=uploads)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Any unit that isn't waiting on a clarification question reaches
    # status="processing" rather than "ready" straight away now -- full
    # enrichment runs in the background so this response (and therefore
    # the UI) doesn't wait on it; see docs/DECISIONS.md. A unit the user
    # still needs to answer a question for has nothing scheduled yet.
    for unit in capture.units:
        if unit.status == "processing":
            background_tasks.add_task(finish_unit_processing, unit.id)
    return capture


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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    capture = _get_capture_or_404(db, capture_id)
    data = await file.read()
    try:
        return await add_attachment(db, llm, capture, file.filename or "upload", data)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
