import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingestion.pipeline import (
    IngestionError,
    correct_unit,
    discard_unit,
    finish_unit_processing,
    resolve_clarification,
    retry_enrichment,
)
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.models import MemoryUnit
from app.schemas import ClarifyRequest, CorrectRequest, MemoryUnitOut

router = APIRouter(prefix="/api/memory-units", tags=["memory-units"])


def _get_unit_or_404(db: Session, unit_id: uuid.UUID) -> MemoryUnit:
    unit = db.get(MemoryUnit, unit_id)
    if unit is None:
        raise HTTPException(status_code=404, detail="Memory unit not found")
    return unit


def _schedule_if_processing(background_tasks: BackgroundTasks, unit: MemoryUnit) -> MemoryUnit:
    """Full enrichment runs in the background once clarification is fully
    resolved (status == "processing", set by _advance_unit) -- see
    docs/DECISIONS.md. Scheduled here, after the response-bound db
    session's work is done, so the background task opens its own fresh
    session rather than racing the request's.
    """
    if unit.status == "processing":
        background_tasks.add_task(finish_unit_processing, unit.id)
    return unit


@router.get("/{unit_id}", response_model=MemoryUnitOut)
async def get(unit_id: uuid.UUID, db: Session = Depends(get_db)):
    return _get_unit_or_404(db, unit_id)


@router.post("/{unit_id}/clarify", response_model=MemoryUnitOut)
async def clarify(
    background_tasks: BackgroundTasks,
    unit_id: uuid.UUID,
    body: ClarifyRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    unit = _get_unit_or_404(db, unit_id)
    if unit.pending_clarification is None:
        raise HTTPException(status_code=400, detail="Memory unit has no pending clarification")
    try:
        updated = await resolve_clarification(db, llm, unit, body.answer)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _schedule_if_processing(background_tasks, updated)


@router.post("/{unit_id}/retry", response_model=MemoryUnitOut)
async def retry(
    background_tasks: BackgroundTasks,
    unit_id: uuid.UUID,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Retry derived processing for a unit stuck in draft/failed, or one
    left in "processing" by a background task that never finished (e.g.
    the server restarted mid-task -- a scheduled BackgroundTask doesn't
    survive that, unlike everything else in this pipeline, see
    docs/DECISIONS.md)."""
    unit = _get_unit_or_404(db, unit_id)
    if unit.status not in ("draft", "failed", "processing"):
        raise HTTPException(status_code=400, detail=f"Unit is '{unit.status}', nothing to retry")
    updated = await retry_enrichment(db, llm, unit)
    return _schedule_if_processing(background_tasks, updated)


@router.post("/{unit_id}/correct", response_model=MemoryUnitOut)
async def correct(
    unit_id: uuid.UUID,
    body: CorrectRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """"Correct this memory": free-text user correction, re-runs
    enrichment (spec section 30). Derived metadata stays read-only
    otherwise -- this is the one sanctioned way to change it. Unlike
    fresh ingestion/clarification, this stays synchronous: the unit is
    already "ready" (searchable) going in, so there's no equivalent
    "user is blocked waiting to see a question" pressure to relieve."""
    unit = _get_unit_or_404(db, unit_id)
    try:
        return await correct_unit(db, llm, unit, body.correction)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{unit_id}/discard", response_model=MemoryUnitOut)
async def discard(unit_id: uuid.UUID, db: Session = Depends(get_db)):
    """Cancel a unit awaiting clarification (spec section 3: the user may
    cancel/discard a draft rather than answer)."""
    unit = _get_unit_or_404(db, unit_id)
    if unit.status != "awaiting_clarification":
        raise HTTPException(status_code=400, detail=f"Unit is '{unit.status}', nothing to discard")
    return await discard_unit(db, unit)
