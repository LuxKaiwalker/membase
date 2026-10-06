import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingestion.pipeline import IngestionError, resolve_clarification, retry_enrichment
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.models import MemoryUnit
from app.schemas import ClarifyRequest, MemoryUnitOut

router = APIRouter(prefix="/api/memory-units", tags=["memory-units"])


def _get_unit_or_404(db: Session, unit_id: uuid.UUID) -> MemoryUnit:
    unit = db.get(MemoryUnit, unit_id)
    if unit is None:
        raise HTTPException(status_code=404, detail="Memory unit not found")
    return unit


@router.get("/{unit_id}", response_model=MemoryUnitOut)
async def get(unit_id: uuid.UUID, db: Session = Depends(get_db)):
    return _get_unit_or_404(db, unit_id)


@router.post("/{unit_id}/clarify", response_model=MemoryUnitOut)
async def clarify(
    unit_id: uuid.UUID,
    body: ClarifyRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    unit = _get_unit_or_404(db, unit_id)
    if unit.pending_clarification is None:
        raise HTTPException(status_code=400, detail="Memory unit has no pending clarification")
    try:
        return await resolve_clarification(db, llm, unit, body.answer)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{unit_id}/retry", response_model=MemoryUnitOut)
async def retry(
    unit_id: uuid.UUID,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Retry derived processing for a unit stuck in draft/failed (spec
    section 25: "derived processing may be retried")."""
    unit = _get_unit_or_404(db, unit_id)
    if unit.status not in ("draft", "failed"):
        raise HTTPException(status_code=400, detail=f"Unit is '{unit.status}', nothing to retry")
    return await retry_enrichment(db, llm, unit)
