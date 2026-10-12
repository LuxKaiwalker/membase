import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.llm.base import LLMProvider
from app.retrieval.pipeline import (
    ABSTENTION_MESSAGE,
    TOP_K_FOR_ANSWER,
    ScoredChunk,
    ScoredUnit,
    embed_query,
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
# what else to search?). MAX_AGENT_ITERATIONS search rounds, up to
# MAX_REFORMULATIONS_PER_ROUND reformulated queries per round (see the
# streamlining note below for the current values and why) -- hand-picked,
# not tuned; see docs/TODO.md. This is the
# "automatic iteration" layer (spec section 9/20): fast, bounded,
# no-user-interaction. app/retrieval/session.py wraps repeated calls to
# automatic_search() in user-mediated rounds for the full iterative-recall
# experience, which is now the canonical retrieval architecture (see
# docs/DECISIONS.md "RecallSession canonical") -- POST /api/recall itself
# goes through app/retrieval/session.py's start_session(), not run_agent(),
# so there's exactly one retrieval+answer-decision algorithm end to end.
# run_agent() below is kept as an internal primitive: a one-shot
# search-and-answer with no ambiguity narrowing/discriminating question
# and no persisted session row, for callers that want exactly that (the
# eval harness, latency benchmarks) rather than the full session flow.
# Streamlined down from 3/2 (see docs/DECISIONS.md "Reduce automatic
# retrieval iterations"): storage should be cheaper than recall, and
# recall itself should be cheap by default -- a query that still isn't
# answerable after one reformulation is better handed to the user as a
# cue request (RecallSession's needs_user_cue) than burned on a third
# automatic guess. Initial defaults per the streamlining spec, not
# re-tuned against the eval corpus yet; revisit if eval numbers call
# for it (see docs/TODO.md).
MAX_AGENT_ITERATIONS = 2
MAX_REFORMULATIONS_PER_ROUND = 1


# Deterministic bypass around the expensive evaluate_sufficiency LLM call
# (see docs/DECISIONS.md "Deterministic sufficiency short-circuit"): when
# the top candidate is overwhelmingly dominant AND its evidence is
# substantive (not a degraded fallback unit), there's nothing a reasoning
# call could usefully add -- skip straight to answering. Deliberately
# conservative: all three signals (absolute score, margin over the
# runner-up, evidence completeness) must hold, since a false early
# acceptance (treating a wrong top hit as "clearly enough") is worse than
# one extra LLM call. Thresholds are an initial engineering default
# (sanity-checked against the real model via scripts/run-eval.sh, not
# swept against it) -- see docs/DECISIONS.md for the measurement and
# docs/TODO.md for fuller calibration as a follow-up.
HIGH_CONFIDENCE_FUSED_SCORE = 0.80
HIGH_CONFIDENCE_MARGIN = 0.20


def _is_high_confidence(ranked: list[ScoredUnit]) -> bool:
    if not ranked:
        return False
    top = ranked[0]
    if top.fused_score < HIGH_CONFIDENCE_FUSED_SCORE:
        return False
    if len(ranked) > 1 and (top.fused_score - ranked[1].fused_score) < HIGH_CONFIDENCE_MARGIN:
        return False
    # Evidence completeness: a unit that fell back to empty/degraded
    # semantics (LLM was unavailable during ingestion, see
    # app/ingestion/analysis.py) is a weaker answer source even at a high
    # retrieval score -- don't skip the reasoning step for it.
    semantic = top.unit.semantic or {}
    return semantic.get("enrichment_status") != "unavailable"


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
            # Embed once per query string and reuse it for both searches
            # (see docs/DECISIONS.md "Dedupe query embeddings") -- this
            # was previously two independent llm.embed() calls for the
            # exact same text.
            q_vector = await embed_query(llm, q)
            for scored in await hybrid_search(db, llm, q, limit=TOP_K_FOR_ANSWER, query_vector=q_vector):
                existing = best.get(scored.unit.id)
                if existing is None or scored.fused_score > existing.fused_score:
                    best[scored.unit.id] = scored
            for scored_chunk in await hybrid_search_chunks(db, llm, q, limit=TOP_K_FOR_ANSWER, query_vector=q_vector):
                existing_chunk = best_chunks.get(scored_chunk.chunk.id)
                if existing_chunk is None or scored_chunk.fused_score > existing_chunk.fused_score:
                    best_chunks[scored_chunk.chunk.id] = scored_chunk

        ranked = sorted(best.values(), key=lambda s: s.fused_score, reverse=True)[:TOP_K_FOR_ANSWER]
        ranked_chunks = sorted(best_chunks.values(), key=lambda s: s.fused_score, reverse=True)[:TOP_K_FOR_ANSWER]

        if iteration == MAX_AGENT_ITERATIONS - 1:
            break
        if _is_high_confidence(ranked):
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
