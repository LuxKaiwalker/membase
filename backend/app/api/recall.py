from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.retrieval.agent import run_agent
from app.schemas import ChunkSourceOut, RecallRequest, RecallResponse, SourceOut

router = APIRouter(prefix="/api", tags=["recall"])


@router.post("/recall", response_model=RecallResponse)
async def recall(
    body: RecallRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
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
