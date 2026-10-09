import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingestion.pipeline import IngestionError, create_capture, finish_unit_processing
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.models import RecallSession
from app.retrieval.pipeline import ABSTENTION_MESSAGE
from app.retrieval.session import (
    STATUS_SUCCESS,
    RecallSessionError,
    cancel_session,
    continue_session,
    start_session,
    user_supplied_texts,
)
from app.schemas import (
    CaptureOut,
    ChunkSourceOut,
    RecallRequest,
    RecallResponse,
    RecallSessionMessageRequest,
    RecallSessionOut,
    SaveAsMemoryRequest,
    SourceOut,
)

router = APIRouter(prefix="/api", tags=["recall"])


@router.post("/recall", response_model=RecallResponse)
async def recall(
    body: RecallRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Legacy one-shot recall (no multi-turn state for the caller). There
    is only one retrieval+answer algorithm (see docs/DECISIONS.md
    "RecallSession canonical") -- this runs a RecallSession's first
    automatic round exactly as POST /api/recall/sessions does (same
    ambiguity narrowing, same grounding decision) and reports its
    outcome in the flat legacy shape. The session row itself is still
    persisted (recall history, not memory), but its id is never returned
    here, so a caller of this endpoint has no way to continue it; if the
    first round can't produce a confident answer (ambiguous with no
    resolving angle, nothing found, or an error), that's reported as an
    abstention rather than a follow-up question, since a one-shot caller
    has no way to answer one.
    """
    try:
        session = await start_session(db, llm, body.query)
    except RecallSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if session.status != STATUS_SUCCESS:
        return RecallResponse(answer=ABSTENTION_MESSAGE, abstained=True, sources=[], chunk_sources=[])
    return RecallResponse(
        answer=session.final_answer or "",
        abstained=False,
        sources=[SourceOut(**s) for s in session.final_sources],
        chunk_sources=[ChunkSourceOut(**c) for c in session.final_chunk_sources],
    )


def _get_session_or_404(db: Session, session_id: uuid.UUID) -> RecallSession:
    session = db.get(RecallSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Recall session not found")
    return session


def _session_out(session: RecallSession) -> RecallSessionOut:
    return RecallSessionOut(
        id=session.id,
        original_query=session.original_query,
        status=session.status,
        current_question=session.current_question,
        messages=session.messages,
        cues=session.cues,
        hypotheses=session.hypotheses,
        round_count=session.round_count,
        last_candidate_count=session.last_candidate_count,
        final_answer=session.final_answer,
        sources=session.final_sources,
        chunk_sources=session.final_chunk_sources,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.post("/recall/sessions", response_model=RecallSessionOut, status_code=201)
async def create_session(
    body: RecallRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Start an iterative recall session (spec sections 2-11): the query
    may only be a weak/partial cue. The first round runs immediately;
    the response's `status` tells the caller whether it already has an
    answer, needs another cue (`current_question`), or found nothing.
    """
    session = await start_session(db, llm, body.query)
    return _session_out(session)


@router.get("/recall/sessions/{session_id}", response_model=RecallSessionOut)
async def get_session(session_id: uuid.UUID, db: Session = Depends(get_db)):
    return _session_out(_get_session_or_404(db, session_id))


@router.post("/recall/sessions/{session_id}/messages", response_model=RecallSessionOut)
async def post_message(
    session_id: uuid.UUID,
    body: RecallSessionMessageRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Add a user turn to an in-progress session: an answer to
    `current_question`, an unprompted extra recollection, or a
    correction ("no, it was Markus") -- all are treated as new cues
    (spec section 15).
    """
    session = _get_session_or_404(db, session_id)
    try:
        session = await continue_session(db, llm, session, body.text)
    except RecallSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _session_out(session)


@router.post("/recall/sessions/{session_id}/cancel", response_model=RecallSessionOut)
async def cancel(session_id: uuid.UUID, db: Session = Depends(get_db)):
    session = _get_session_or_404(db, session_id)
    try:
        session = cancel_session(db, session)
    except RecallSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _session_out(session)


@router.post("/recall/sessions/{session_id}/save-as-memory", response_model=CaptureOut, status_code=201)
async def save_as_memory(
    session_id: uuid.UUID,
    body: SaveAsMemoryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Explicit user action (see docs/DECISIONS.md "Save as memory"):
    creates a brand-new Capture through the ordinary storage pipeline --
    segmentation, clarification, enrichment, provenance -- exactly like
    POST /api/captures. Nothing in a RecallSession (a cue, a hypothesis,
    the final answer) is ever promoted directly into a MemoryUnit;
    `text` must match something the user actually said in this session
    (a message turn or an extracted cue), never arbitrary text or an
    agent-generated hypothesis/question, so this can't be used as a
    backdoor to skip the normal pipeline.
    """
    session = _get_session_or_404(db, session_id)
    text = body.text.strip()
    if not text or text not in user_supplied_texts(session):
        raise HTTPException(status_code=400, detail="text must match something you said in this session")
    try:
        capture = await create_capture(db, llm, text=text, uploads=[])
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for unit in capture.units:
        if unit.status == "processing":
            background_tasks.add_task(finish_unit_processing, unit.id)
    return capture
