import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.llm.base import LLMError, LLMProvider
from app.models import MemoryUnit

logger = logging.getLogger(__name__)

CANDIDATE_LIMIT = 20
TOP_K_FOR_ANSWER = 5
MAX_CONTEXT_PER_PRIMARY = 2
MAX_CONTEXT_TOTAL = 6

# Minimum vector cosine similarity to consider a unit relevant at all.
# Empirically, nomic-embed-text puts *unrelated* short English sentences
# around 0.4-0.65 cosine similarity -- there is no principled zero point,
# so this is a hand-picked value from manual trials, not a tuned
# threshold. See docs/TODO.md: retrieval threshold calibration.
VECTOR_SIM_THRESHOLD = 0.55

ABSTENTION_MESSAGE = "I don't have a stored memory about this."

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
    "supports, even if it only partially answers the question. 'Related "
    "context' evidence is from the same original note as a 'primary' "
    "memory; USE it to resolve pronouns, names, or vague references in the "
    "primary evidence (e.g. if primary evidence says 'she recommended X' "
    "and related context names a person, answer with that person's name, "
    "not the pronoun). Prefer primary evidence for which fact to report, "
    "but resolve what it's referring to using the related context. Set "
    "grounded to false ONLY if the evidence is about something else "
    "entirely and has nothing relevant to "
    "the question; otherwise answer with what the evidence does say and set "
    "grounded to true. Do not guess or fabricate anything not present in "
    "the evidence."
)


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _vector_candidates(db: Session, query_vector: list[float], limit: int) -> dict[uuid.UUID, float]:
    rows = db.execute(
        text(
            """
            SELECT id, 1 - (embedding <=> (:qv)::vector) AS score
            FROM memory_units
            WHERE embedding IS NOT NULL AND status = 'ready'
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
            FROM memory_units
            WHERE fts @@ websearch_to_tsquery('english', :q) AND status = 'ready'
            ORDER BY score DESC
            LIMIT :limit
            """
        ),
        {"q": query, "limit": limit},
    ).all()
    return {row.id: float(row.score) for row in rows}


@dataclass
class ScoredUnit:
    unit: MemoryUnit
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
        # Vector weighted higher: it's what lets paraphrased/indirect/alias
        # cues match at all. FTS mainly breaks ties and rewards literal
        # overlap. No separate reranker model for the MVP (docs/DECISIONS.md).
        fused[mid] = (v, f, 0.65 * norm_v + 0.35 * norm_f)
    return fused


async def hybrid_search(
    db: Session, llm: LLMProvider, query: str, limit: int = TOP_K_FOR_ANSWER
) -> list[ScoredUnit]:
    """One retrieval round: vector + full-text over MemoryUnit, fused by a
    weighted-score rerank, filtered to a basic relevance bar. Used both as
    the agent's SEARCH action and directly for repeated/reformulated
    queries within one agent run.
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
    units = {u.id: u for u in db.query(MemoryUnit).filter(MemoryUnit.id.in_(ranked_ids)).all()}
    return [
        ScoredUnit(unit=units[mid], vector_score=fused[mid][0], fts_score=fused[mid][1], fused_score=fused[mid][2])
        for mid in ranked_ids
        if mid in units
    ]


def sibling_context(db: Session, primary: list[ScoredUnit]) -> list[MemoryUnit]:
    """Bounded expansion to sibling units from the same capture (spec
    section 20): extra context for interpreting a primary result, never a
    substitute for it, and never merged into one giant blob.
    """
    primary_ids = {s.unit.id for s in primary}
    capture_ids = {s.unit.capture_id for s in primary}
    if not capture_ids:
        return []

    siblings = (
        db.query(MemoryUnit)
        .filter(MemoryUnit.capture_id.in_(capture_ids), MemoryUnit.status == "ready")
        .filter(~MemoryUnit.id.in_(primary_ids))
        .all()
    )

    per_capture_count: dict[uuid.UUID, int] = {}
    context: list[MemoryUnit] = []
    for unit in siblings:
        count = per_capture_count.get(unit.capture_id, 0)
        if count >= MAX_CONTEXT_PER_PRIMARY or len(context) >= MAX_CONTEXT_TOTAL:
            continue
        per_capture_count[unit.capture_id] = count + 1
        context.append(unit)
    return context


def _format_unit(unit: MemoryUnit, label: str) -> str:
    lines = [f"[{label}] (created {unit.created_at.date().isoformat()})", f"Content: {unit.authoritative_content}"]
    semantic = unit.semantic or {}
    if semantic.get("meaning"):
        lines.append(f"Meaning: {semantic['meaning']}")
    if semantic.get("aliases"):
        lines.append(f"Also known as: {', '.join(semantic['aliases'])}")
    if semantic.get("definitions"):
        lines.append("Definitions: " + "; ".join(f"{d['term']} = {d['definition']}" for d in semantic["definitions"]))
    for a in unit.capture.attachments:
        if a.extracted_text:
            lines.append(f"Attachment '{a.original_filename}' excerpt: {a.extracted_text[:500]}")
    return "\n".join(lines)


def format_evidence(primary: list[ScoredUnit], context: list[MemoryUnit]) -> str:
    blocks = [_format_unit(s.unit, f"Primary evidence {i}") for i, s in enumerate(primary, start=1)]
    blocks += [_format_unit(u, "Related context (same note)") for u in context]
    return "\n\n".join(blocks)


async def generate_answer(
    llm: LLMProvider, query: str, primary: list[ScoredUnit], context: list[MemoryUnit]
) -> tuple[str, bool]:
    """Returns (answer_text, grounded). Caller is responsible for
    substituting the fixed abstention message when grounded is False --
    this function never does that itself, so the hard override lives in
    exactly one place (app/retrieval/agent.py)."""
    evidence = format_evidence(primary, context)
    prompt = f"Question: {query}\n\nEvidence from stored memories:\n{evidence}"
    try:
        result = await llm.generate_structured(prompt, schema=_ANSWER_SCHEMA, system=_ANSWER_SYSTEM)
    except LLMError:
        logger.warning("Answer generation failed; treating as ungrounded")
        return "", False

    grounded = bool(result.get("grounded"))
    answer_text = result.get("answer") if isinstance(result.get("answer"), str) else ""
    return answer_text.strip(), grounded and bool(answer_text.strip())
