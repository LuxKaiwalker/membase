import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.llm.base import LLMError, LLMProvider
from app.models import Memory

logger = logging.getLogger(__name__)

CANDIDATE_LIMIT = 20
TOP_K_FOR_ANSWER = 5
# Minimum vector cosine similarity to consider a memory relevant at all.
# Empirically, nomic-embed-text puts *unrelated* short English sentences
# around 0.42-0.65 cosine similarity -- there is no principled zero point,
# so this is a hand-picked value from a handful of manual trials, not a
# tuned threshold. See docs/TODO.md: retrieval threshold calibration.
VECTOR_SIM_THRESHOLD = 0.55

_ABSTENTION_MESSAGE = "I don't have a stored memory about this."

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "grounded": {
            "type": "boolean",
            "description": "true only if the evidence actually answers the question",
        },
    },
    "required": ["answer", "grounded"],
}

_ANSWER_SYSTEM = (
    "You are a personal memory assistant. You answer ONLY using the evidence "
    "provided below, which comes from the user's own stored memories. Never "
    "use outside/pretrained knowledge about people, places, or facts not "
    "present in the evidence. Give the most complete answer the evidence "
    "supports, even if it only partially answers the question -- e.g. if "
    "asked for a paper's title but only its topic and who recommended it is "
    "known, say that. Set grounded to false ONLY if the evidence is about "
    "something else entirely and has nothing relevant to the question; "
    "otherwise answer with what the evidence does say and set grounded to "
    "true. Do not guess or fabricate anything not present in the evidence."
)


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _vector_candidates(db: Session, query_vector: list[float], limit: int) -> dict[uuid.UUID, float]:
    rows = db.execute(
        text(
            """
            SELECT id, 1 - (embedding <=> (:qv)::vector) AS score
            FROM memories
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> (:qv)::vector
            LIMIT :limit
            """
        ),
        {"qv": _vector_literal(query_vector), "limit": limit},
    ).all()
    return {row.id: float(row.score) for row in rows}


def _fts_candidates(db: Session, query: str, limit: int) -> dict[uuid.UUID, float]:
    rows = db.execute(
        text(
            """
            SELECT id, ts_rank(fts, websearch_to_tsquery('english', :q)) AS score
            FROM memories
            WHERE fts @@ websearch_to_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT :limit
            """
        ),
        {"q": query, "limit": limit},
    ).all()
    return {row.id: float(row.score) for row in rows}


@dataclass
class ScoredMemory:
    memory: Memory
    vector_score: float | None
    fts_score: float | None
    fused_score: float


def _fuse(
    vector_scores: dict[uuid.UUID, float], fts_scores: dict[uuid.UUID, float]
) -> dict[uuid.UUID, tuple[float | None, float | None, float]]:
    max_fts = max(fts_scores.values(), default=0.0) or 1.0
    fused: dict[uuid.UUID, tuple[float | None, float | None, float]] = {}
    for mid in set(vector_scores) | set(fts_scores):
        v = vector_scores.get(mid)
        f = fts_scores.get(mid)
        norm_f = (f / max_fts) if f is not None else 0.0
        norm_v = v if v is not None else 0.0
        # Vector weighted higher: it's what lets paraphrased/indirect cues
        # (the whole point of this project, see docs/ARCHITECTURE.md) match
        # at all. FTS mainly breaks ties and rewards literal overlap.
        fused[mid] = (v, f, 0.65 * norm_v + 0.35 * norm_f)
    return fused


async def retrieve(
    db: Session, llm: LLMProvider, query: str, limit: int = TOP_K_FOR_ANSWER
) -> list[ScoredMemory]:
    """Hybrid retrieval: vector + full-text, fused by a simple weighted-score
    rerank (no separate cross-encoder reranker model in the MVP -- see
    docs/DECISIONS.md). Returns only candidates that clear a basic
    relevance bar, most relevant first.
    """
    query = query.strip()
    if not query:
        return []

    vector_scores: dict[uuid.UUID, float] = {}
    try:
        [query_vector] = await llm.embed([query])
        vector_scores = _vector_candidates(db, query_vector, CANDIDATE_LIMIT)
    except LLMError:
        logger.warning("Query embedding failed; falling back to full-text-only retrieval")

    fts_scores = _fts_candidates(db, query, CANDIDATE_LIMIT)

    fused = _fuse(vector_scores, fts_scores)
    relevant_ids = [
        mid
        for mid, (v, f, _score) in fused.items()
        if (v is not None and v >= VECTOR_SIM_THRESHOLD) or f is not None
    ]
    if not relevant_ids:
        return []

    ranked_ids = sorted(relevant_ids, key=lambda mid: fused[mid][2], reverse=True)[:limit]

    memories = {m.id: m for m in db.query(Memory).filter(Memory.id.in_(ranked_ids)).all()}
    return [
        ScoredMemory(
            memory=memories[mid],
            vector_score=fused[mid][0],
            fts_score=fused[mid][1],
            fused_score=fused[mid][2],
        )
        for mid in ranked_ids
        if mid in memories
    ]


def _format_evidence(scored: list[ScoredMemory]) -> str:
    blocks = []
    for i, s in enumerate(scored, start=1):
        m = s.memory
        derived = m.derived or {}
        lines = [f"[Memory {i}] (created {m.created_at.date().isoformat()})", f"Caption: {m.caption}"]
        if derived.get("summary"):
            lines.append(f"Summary: {derived['summary']}")
        if derived.get("people"):
            lines.append(f"People: {', '.join(derived['people'])}")
        if derived.get("location"):
            lines.append(f"Location: {derived['location']}")
        if derived.get("event_date"):
            lines.append(f"Event date: {derived['event_date']}")
        for a in m.attachments:
            if a.extracted_text:
                lines.append(f"Attachment '{a.original_filename}' excerpt: {a.extracted_text[:500]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@dataclass
class RecallResult:
    answer: str
    abstained: bool
    sources: list[ScoredMemory]


async def answer(db: Session, llm: LLMProvider, query: str) -> RecallResult:
    scored = await retrieve(db, llm, query)
    if not scored:
        return RecallResult(answer=_ABSTENTION_MESSAGE, abstained=True, sources=[])

    evidence = _format_evidence(scored)
    prompt = f"Question: {query}\n\nEvidence from stored memories:\n{evidence}"
    try:
        result = await llm.generate_structured(prompt, schema=_ANSWER_SCHEMA, system=_ANSWER_SYSTEM)
    except LLMError:
        logger.warning("Answer generation failed; abstaining rather than guessing")
        return RecallResult(answer=_ABSTENTION_MESSAGE, abstained=True, sources=scored)

    grounded = bool(result.get("grounded"))
    text_answer = result.get("answer") if isinstance(result.get("answer"), str) else ""
    if not grounded or not text_answer.strip():
        return RecallResult(answer=_ABSTENTION_MESSAGE, abstained=True, sources=[])

    return RecallResult(answer=text_answer.strip(), abstained=False, sources=scored)
