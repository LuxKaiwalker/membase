import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.models import RecallSession
from app.retrieval.agent import run_agent
from app.retrieval.session import RecallSessionError, cancel_session, continue_session, start_session
from app.schemas import (
    ChunkSourceOut,
    RecallRequest,
    RecallResponse,
    RecallSessionMessageRequest,
    RecallSessionOut,
    SourceOut,
)

router = APIRouter(prefix="/api", tags=["recall"])


@router.post("/recall", response_model=RecallResponse)
async def recall(
    body: RecallRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    """Legacy one-shot recall (no multi-turn state). Kept for simple
    direct/paraphrased/indirect lookups and the eval harness; see
    POST /api/recall/sessions for iterative recall with clarifying
    questions.
    """
    result = await run_agent(db, llm, body.query)
    sources = [
        SourceOut(
            id=s.unit.id,
            capture_id=s.unit.capture_id,
            content=s.unit.authoritative_content,
            semantic=s.unit.semantic,
            created_at=s.unit.created_at,
            vector_score=s.vector_score,
            fts_score=s.fts_score,
            fused_score=s.fused_score,
        )
        for s in result.sources
    ]
    chunk_sources = [
        ChunkSourceOut(
            id=c.chunk.id,
            capture_id=c.chunk.capture_id,
            attachment_filename=c.chunk.attachment.original_filename,
            page_number=c.chunk.page_number,
            excerpt=c.chunk.text[:500],
            vector_score=c.vector_score,
            fts_score=c.fts_score,
            fused_score=c.fused_score,
        )
        for c in result.chunk_sources
    ]
    return RecallResponse(
        answer=result.answer, abstained=result.abstained, sources=sources, chunk_sources=chunk_sources
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
