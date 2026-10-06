from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.llm import get_llm_provider
from app.llm.base import LLMProvider
from app.retrieval.pipeline import answer as run_recall
from app.schemas import RecallRequest, RecallResponse, SourceOut

router = APIRouter(prefix="/api", tags=["recall"])


@router.post("/recall", response_model=RecallResponse)
async def recall(
    body: RecallRequest,
    db: Session = Depends(get_db),
    llm: LLMProvider = Depends(get_llm_provider),
):
    result = await run_recall(db, llm, body.query)
    sources = [
        SourceOut(
            id=s.memory.id,
            caption=s.memory.caption,
            derived=s.memory.derived,
            created_at=s.memory.created_at,
            vector_score=s.vector_score,
            fts_score=s.fts_score,
            fused_score=s.fused_score,
        )
        for s in result.sources
    ]
    return RecallResponse(answer=result.answer, abstained=result.abstained, sources=sources)
