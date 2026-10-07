import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.llm.base import LLMError, LLMProvider
from app.retrieval.pipeline import (
    ABSTENTION_MESSAGE,
    TOP_K_FOR_ANSWER,
    ScoredChunk,
    ScoredUnit,
    evaluate_sufficiency,
    generate_answer,
    hybrid_search,
    hybrid_search_chunks,
    sibling_context,
)

logger = logging.getLogger(__name__)

# Bounded deterministic state machine (spec section 29), not a general
# tool-calling agent framework: the loop itself is the orchestration, the
# LLM only ever makes narrow structured decisions within it (sufficient?
# what else to search?). 3 search rounds, up to 2 reformulated queries
# per round -- hand-picked, not tuned; see docs/TODO.md. This is the
# "automatic iteration" layer (spec section 9/20): fast, bounded,
# no-user-interaction. app/retrieval/session.py wraps repeated calls to
# automatic_search() in user-mediated rounds for the full iterative-recall
# experience; this module's run_agent() is the simpler one-shot path
# (still used by the legacy POST /api/recall and the eval harness).
MAX_AGENT_ITERATIONS = 3
MAX_REFORMULATIONS_PER_ROUND = 2


@dataclass
class AutomaticSearchResult:
    ranked: list[ScoredUnit]
    ranked_chunks: list[ScoredChunk]
    tried_queries: set[str]
    # Reformulated queries the LLM proposed along the way -- useful search
    # aids, never answer content (spec section 6c/17/29: a hypothesis is
    # not a fact and must never be promoted to one).
    hypotheses: list[str] = field(default_factory=list)


async def automatic_search(
    db: Session,
    llm: LLMProvider,
    query: str,
    *,
    extra_queries: list[str] | None = None,
    tried: set[str] | None = None,
) -> AutomaticSearchResult:
    """Bounded SEARCH -> EVALUATE -> REFORMULATE loop.

    `query` and every string in `extra_queries` are always (re)searched in
    the first iteration regardless of `tried` -- `tried` only suppresses
    re-proposing an *already-seen reformulation* (across this call, and,
    if the caller passes in a previous call's `tried_queries`, across
    prior rounds of the same session too). This matters: a RecallSession
    calling this once per round must never "forget" a previously-found
    candidate just because the original query was already searched in an
    earlier round (spec section 22/23 -- retrieval accumulates, and the
    original query is never dropped to avoid query drift).
    """
    tried = set(tried) if tried else set()
    best: dict[uuid.UUID, ScoredUnit] = {}
    best_chunks: dict[uuid.UUID, ScoredChunk] = {}
    hypotheses: list[str] = []
    current_queries = [query, *(extra_queries or [])]
    ranked: list[ScoredUnit] = []
    ranked_chunks: list[ScoredChunk] = []

    for iteration in range(MAX_AGENT_ITERATIONS):
        for q in current_queries:
            tried.add(q)
            for scored in await hybrid_search(db, llm, q, limit=TOP_K_FOR_ANSWER):
                existing = best.get(scored.unit.id)
                if existing is None or scored.fused_score > existing.fused_score:
                    best[scored.unit.id] = scored
            for scored_chunk in await hybrid_search_chunks(db, llm, q, limit=TOP_K_FOR_ANSWER):
                existing_chunk = best_chunks.get(scored_chunk.chunk.id)
                if existing_chunk is None or scored_chunk.fused_score > existing_chunk.fused_score:
                    best_chunks[scored_chunk.chunk.id] = scored_chunk

        ranked = sorted(best.values(), key=lambda s: s.fused_score, reverse=True)[:TOP_K_FOR_ANSWER]
        ranked_chunks = sorted(best_chunks.values(), key=lambda s: s.fused_score, reverse=True)[:TOP_K_FOR_ANSWER]

        if iteration == MAX_AGENT_ITERATIONS - 1:
            break
        decision = await evaluate_sufficiency(llm, query, ranked, tried)
        for hq in decision["reformulated_queries"]:
            if hq not in hypotheses:
                hypotheses.append(hq)
        if decision["sufficient"]:
            break
        next_queries = [q for q in decision["reformulated_queries"][:MAX_REFORMULATIONS_PER_ROUND] if q not in tried]
        if not next_queries:
            break
        current_queries = next_queries  # only reformulations here; seed queries already searched above

    return AutomaticSearchResult(ranked=ranked, ranked_chunks=ranked_chunks, tried_queries=tried, hypotheses=hypotheses)


@dataclass
class RecallResult:
    answer: str
    abstained: bool
    sources: list[ScoredUnit]
    chunk_sources: list[ScoredChunk] = field(default_factory=list)


async def run_agent(db: Session, llm: LLMProvider, query: str) -> RecallResult:
    """One-shot recall: INTERPRET -> SEARCH -> EVALUATE -> (ANSWER |
    REFORMULATE -> SEARCH | ABSTAIN), bounded to MAX_AGENT_ITERATIONS
    rounds. Grounding (spec section 24/30) is enforced here, in exactly
    one place: whatever generate_answer returns, an ungrounded result is
    always replaced with the fixed abstention message and an empty source
    list. For the multi-turn, cue-accumulating version of recall, see
    app/retrieval/session.py.
    """
    query = query.strip()
    if not query:
        return RecallResult(answer=ABSTENTION_MESSAGE, abstained=True, sources=[])

    result = await automatic_search(db, llm, query)
    if not result.ranked and not result.ranked_chunks:
        return RecallResult(answer=ABSTENTION_MESSAGE, abstained=True, sources=[])

    context = sibling_context(db, result.ranked)
    answer_text, grounded = await generate_answer(llm, query, result.ranked, context, result.ranked_chunks)
    if not grounded:
        return RecallResult(answer=ABSTENTION_MESSAGE, abstained=True, sources=[])
    return RecallResult(
        answer=answer_text, abstained=False, sources=result.ranked, chunk_sources=result.ranked_chunks
    )
