import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session, selectinload

from app.llm.base import LLMError, LLMProvider
from app.models import Attachment, AttachmentChunk, Capture, MemoryUnit

logger = logging.getLogger(__name__)

CANDIDATE_LIMIT = 20
TOP_K_FOR_ANSWER = 5
MAX_CONTEXT_PER_PRIMARY = 2
MAX_CONTEXT_TOTAL = 6

# Minimum vector cosine similarity to consider a unit relevant at all.
# Calibrated via scripts/sweep-threshold.sh against the 30-case eval set:
# recall@10 is 100% from 0.35 up through 0.55, then drops (96%@0.60,
# 89%@0.65, 78%@0.70) as real matches start failing to clear the bar.
# 0.55 is the highest threshold that still gets full recall, which also
# has the best precision among those -- see docs/DECISIONS.md for the
# full numbers and the eval set's caveats (30 cases, shared-corpus
# cross-contamination in the precision metric specifically).
VECTOR_SIM_THRESHOLD = 0.55

ABSTENTION_MESSAGE = "I don't have a stored memory about this."

_EVALUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "reformulated_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient", "reformulated_queries"],
}

_EVALUATE_SYSTEM = (
    "You are deciding how to search a personal memory store, not answering "
    "the user's question yet. Given the question and the best snippets "
    "found so far (there may be none), decide whether there is already "
    "enough to answer. If not, propose up to 2 alternative search queries: "
    "different phrasings, synonyms, or terminology the user's own notes "
    "might use instead of the question's wording (their notes may use an "
    "abbreviation, nickname, or alias instead of the question's term, or "
    "vice versa). You may use your general knowledge to guess what "
    "alternate terminology might appear in the notes, purely to decide "
    "what to search for -- never state such knowledge as a fact about the "
    "user. If you have no promising alternative queries left to try, "
    "return an empty list."
)


async def evaluate_sufficiency(
    llm: LLMProvider, query: str, ranked: list["ScoredUnit"], tried: set[str]
) -> dict:
    """Narrow structured decision: is what's been found so far enough to
    answer `query`, and if not, what else is worth searching for? The
    reformulated queries this proposes are themselves a form of system
    hypothesis (spec section 6c/18) -- they may use general-knowledge
    guesses about likely terminology, never asserted as fact about the
    user. Used both by the bounded one-shot agent (app/retrieval/agent.py)
    and by each round of an iterative RecallSession
    (app/retrieval/session.py).
    """
    if ranked:
        snippets = "\n".join(f"- {s.unit.authoritative_content}" for s in ranked)
        evidence_block = f"Snippets found so far:\n{snippets}"
    else:
        evidence_block = "No snippets found yet."
    tried_block = f"Already tried: {', '.join(sorted(tried))}" if tried else ""
    prompt = f"Question: {query}\n\n{evidence_block}\n{tried_block}"
    try:
        result = await llm.generate_structured(prompt, schema=_EVALUATE_SCHEMA, system=_EVALUATE_SYSTEM)
    except LLMError:
        # Can't reformulate without the LLM; stop searching and answer
        # with whatever was already found (possibly nothing).
        return {"sufficient": True, "reformulated_queries": []}

    sufficient = bool(result.get("sufficient"))
    queries = result.get("reformulated_queries")
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()] if isinstance(queries, list) else []
    return {"sufficient": sufficient, "reformulated_queries": queries}


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
    "the evidence. If a piece of evidence has a '[Conflict]' note, two "
    "stored memories disagree with each other -- you MUST state that "
    "disagreement explicitly in your answer (e.g. \"You have two stored "
    "memories that conflict: one says X and another says Y.\"), never "
    "silently pick one side or omit the conflicting one."
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


# Sentinel distinguishing "caller didn't pass a vector, compute it
# yourself" from "caller passed query_vector=None because it already
# knows embedding is unavailable" -- see embed_query()/docs/DECISIONS.md
# "Dedupe query embeddings". Plain None can't do double duty for both
# meanings: a caller that already tried and failed must be able to say
# so without hybrid_search silently retrying the same failing call.
_UNSET = object()


async def embed_query(llm: LLMProvider, query: str) -> list[float] | None:
    """Embed once, for a caller (automatic_search) that's about to search
    both memory units and attachment chunks with the same query string --
    each previously called llm.embed(query) independently. Returns None
    on failure rather than raising; both hybrid_search and
    hybrid_search_chunks already degrade to full-text-only when given
    None, same as if they'd failed to embed it themselves.
    """
    try:
        [vector] = await llm.embed([query])
        return vector
    except LLMError:
        logger.warning("Query embedding failed; falling back to full-text-only retrieval")
        return None


async def hybrid_search(
    db: Session, llm: LLMProvider, query: str, limit: int = TOP_K_FOR_ANSWER, query_vector: Any = _UNSET
) -> list[ScoredUnit]:
    """One retrieval round: vector + full-text over MemoryUnit, fused by a
    weighted-score rerank, filtered to a basic relevance bar. Used both as
    the agent's SEARCH action and directly for repeated/reformulated
    queries within one agent run.

    `query_vector`, if given, is used as-is instead of embedding `query`
    again -- for a caller (automatic_search) that already embedded this
    exact query once via embed_query() to search chunks too. Omit it to
    have this function embed `query` itself, same as before.
    """
    query = query.strip()
    if not query:
        return []

    if query_vector is _UNSET:
        query_vector = await embed_query(llm, query)

    vector_scores: dict[uuid.UUID, float] = {}
    if query_vector is not None:
        vector_scores = _vector_candidates(db, query_vector, CANDIDATE_LIMIT)

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


#  Gating bug fix: attachment_chunks are indexed as soon as the attachment
#  is uploaded (_index_attachment_chunks runs during create_capture()
#  before segmentation/clarification even happens), with no status column
#  of their own -- so without this EXISTS check, a chunk stayed
#  searchable even while every sibling MemoryUnit in the same capture was
#  still "awaiting_clarification" or "processing" (not yet "ready"),
#  silently bypassing the status gate that hybrid_search() enforces for
#  memory_units (see docs/ARCHITECTURE.md "Status gate"). Requiring at
#  least one ready sibling unit applies the same gate at the capture
#  granularity attachment_chunks actually have.
_CHUNK_READY_GATE = """
            AND EXISTS (
                SELECT 1 FROM memory_units mu
                WHERE mu.capture_id = attachment_chunks.capture_id AND mu.status = 'ready'
            )"""


def _chunk_vector_candidates(db: Session, query_vector: list[float], limit: int) -> dict[uuid.UUID, float]:
    rows = db.execute(
        text(
            f"""
            SELECT id, 1 - (embedding <=> (:qv)::vector) AS score
            FROM attachment_chunks
            WHERE embedding IS NOT NULL{_CHUNK_READY_GATE}
            ORDER BY embedding <=> (:qv)::vector
            LIMIT :limit
            """
        ),
        {"qv": _vector_literal(query_vector), "limit": limit},
    ).all()
    return {row.id: float(row.score) for row in rows}


def _chunk_fts_candidates(db: Session, query: str, limit: int) -> dict[uuid.UUID, float]:
    rows = db.execute(
        text(
            f"""
            SELECT id, ts_rank(fts, websearch_to_tsquery('english', :q)) AS score
            FROM attachment_chunks
            WHERE fts @@ websearch_to_tsquery('english', :q){_CHUNK_READY_GATE}
            ORDER BY score DESC
            LIMIT :limit
            """
        ),
        {"q": query, "limit": limit},
    ).all()
    return {row.id: float(row.score) for row in rows}


@dataclass
class ScoredChunk:
    chunk: AttachmentChunk
    vector_score: float | None
    fts_score: float | None
    fused_score: float


async def hybrid_search_chunks(
    db: Session, llm: LLMProvider, query: str, limit: int = TOP_K_FOR_ANSWER, query_vector: Any = _UNSET
) -> list[ScoredChunk]:
    """Same hybrid vector+fts fusion as hybrid_search(), over attachment
    chunks instead of memory units (spec section 27: large-document
    content should contribute more than one short blind excerpt).

    `query_vector` behaves exactly as in hybrid_search() -- pass the
    vector already embedded for this same query (see embed_query()) to
    avoid embedding it a second time.
    """
    query = query.strip()
    if not query:
        return []

    if query_vector is _UNSET:
        query_vector = await embed_query(llm, query)

    vector_scores: dict[uuid.UUID, float] = {}
    if query_vector is not None:
        vector_scores = _chunk_vector_candidates(db, query_vector, CANDIDATE_LIMIT)

    fts_scores = _chunk_fts_candidates(db, query, CANDIDATE_LIMIT)

    fused = _fuse(vector_scores, fts_scores)
    relevant_ids = [
        cid
        for cid, (v, f, _score) in fused.items()
        if (v is not None and v >= VECTOR_SIM_THRESHOLD) or f is not None
    ]
    if not relevant_ids:
        return []

    ranked_ids = sorted(relevant_ids, key=lambda cid: fused[cid][2], reverse=True)[:limit]
    # Eager-load attachment -> capture -> units: _format_chunk (below)
    # walks that whole chain for every chunk, and without this each
    # chunk would cost 2 extra lazy-loaded round-trips.
    chunks = {
        c.id: c
        for c in db.query(AttachmentChunk)
        .options(selectinload(AttachmentChunk.attachment).selectinload(Attachment.capture).selectinload(Capture.units))
        .filter(AttachmentChunk.id.in_(ranked_ids))
        .all()
    }
    return [
        ScoredChunk(chunk=chunks[cid], vector_score=fused[cid][0], fts_score=fused[cid][1], fused_score=fused[cid][2])
        for cid in ranked_ids
        if cid in chunks
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
        # Deterministic order -- the loop below truncates to
        # MAX_CONTEXT_PER_PRIMARY per capture, and without an ORDER BY
        # Postgres gives no guarantee which siblings survive that
        # truncation, making answer context non-deterministic across
        # otherwise-identical calls.
        .order_by(MemoryUnit.created_at)
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
    # Attachment content is surfaced via ranked [Attachment evidence]
    # chunk blocks (see _format_chunk) rather than a blind excerpt here --
    # chunking means the relevant part of a large document is actually
    # findable instead of just whatever happens to be first.
    # Advisory conflict markers written by _detect_conflicts() at storage
    # time (see app/ingestion/pipeline.py) -- purely deterministic string
    # formatting, no LLM call, surfacing a disagreement the answer prompt
    # is instructed to state explicitly rather than silently resolve (see
    # docs/DECISIONS.md "Contradiction detection").
    for conflict in unit.potential_conflicts or []:
        note = conflict.get("note") if isinstance(conflict, dict) else None
        if note:
            lines.append(f"[Conflict] {note}")
    return "\n".join(lines)


def _format_chunk(scored: ScoredChunk) -> str:
    chunk = scored.chunk
    page = f", page {chunk.page_number}" if chunk.page_number else ""
    lines = [f"[Attachment evidence] (file '{chunk.attachment.original_filename}'{page})", chunk.text[:800]]
    sibling_units = [u for u in chunk.attachment.capture.units if u.status == "ready"]
    if sibling_units:
        names = "; ".join(u.authoritative_content for u in sibling_units[:2])
        lines.append(f"From the same note as: {names}")
    return "\n".join(lines)


def format_evidence(
    primary: list[ScoredUnit], context: list[MemoryUnit], chunks: list[ScoredChunk] | None = None
) -> str:
    blocks = [_format_unit(s.unit, f"Primary evidence {i}") for i, s in enumerate(primary, start=1)]
    blocks += [_format_unit(u, "Related context (same note)") for u in context]
    blocks += [_format_chunk(c) for c in (chunks or [])]
    return "\n\n".join(blocks)


async def generate_answer(
    llm: LLMProvider,
    query: str,
    primary: list[ScoredUnit],
    context: list[MemoryUnit],
    chunks: list[ScoredChunk] | None = None,
) -> tuple[str, bool]:
    """Returns (answer_text, grounded). Caller is responsible for
    substituting the fixed abstention message when grounded is False --
    this function never does that itself, so the hard override lives in
    exactly one place (app/retrieval/agent.py)."""
    evidence = format_evidence(primary, context, chunks)
    prompt = f"Question: {query}\n\nEvidence from stored memories:\n{evidence}"
    try:
        result = await llm.generate_structured(prompt, schema=_ANSWER_SCHEMA, system=_ANSWER_SYSTEM)
    except LLMError:
        logger.warning("Answer generation failed; treating as ungrounded")
        return "", False

    grounded = bool(result.get("grounded"))
    answer_text = result.get("answer") if isinstance(result.get("answer"), str) else ""
    return answer_text.strip(), grounded and bool(answer_text.strip())
