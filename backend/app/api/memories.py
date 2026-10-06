import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingestion.pipeline import (
    IngestionError,
    add_attachment,
    create_memory,
    resolve_clarification,
)
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.models import Memory
from app.schemas import ClarifyRequest, MemoryOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/memories", tags=["memories"])


def _get_memory_or_404(db: Session, memory_id: uuid.UUID) -> Memory:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@router.post("", response_model=MemoryOut, status_code=201)
async def create(
    caption: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    uploads = [(f.filename or "upload", await f.read()) for f in files if f.filename]
    try:
        return await create_memory(db, llm, caption=caption, uploads=uploads)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{memory_id}", response_model=MemoryOut)
async def get(memory_id: uuid.UUID, db: Session = Depends(get_db)):
    return _get_memory_or_404(db, memory_id)


@router.post("/{memory_id}/clarify", response_model=MemoryOut)
async def clarify(
    memory_id: uuid.UUID,
    body: ClarifyRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    memory = _get_memory_or_404(db, memory_id)
    if memory.pending_clarification is None:
        raise HTTPException(status_code=400, detail="Memory has no pending clarification")
    try:
        return await resolve_clarification(db, llm, memory, body.answer)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{memory_id}/attachments", response_model=MemoryOut)
async def add_attachment_endpoint(
    memory_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    memory = _get_memory_or_404(db, memory_id)
    data = await file.read()
    try:
        return await add_attachment(db, memory, file.filename or "upload", data)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
