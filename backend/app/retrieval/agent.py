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
# per round -- hand-picked, not tuned; see docs/TODO.md.
MAX_AGENT_ITERATIONS = 3
MAX_REFORMULATIONS_PER_ROUND = 2

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


@dataclass
class RecallResult:
    answer: str
    abstained: bool
    sources: list[ScoredUnit]
    chunk_sources: list[ScoredChunk] = field(default_factory=list)


async def _evaluate(llm: LLMProvider, query: str, ranked: list[ScoredUnit], tried: set[str]) -> dict:
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


async def run_agent(db: Session, llm: LLMProvider, query: str) -> RecallResult:
    """INTERPRET -> SEARCH -> EVALUATE -> (ANSWER | REFORMULATE -> SEARCH |
    ABSTAIN), bounded to MAX_AGENT_ITERATIONS rounds (spec section 29/19).
    Grounding (spec section 30) is enforced here, in exactly one place:
    whatever generate_answer returns, an ungrounded result is always
    replaced with the fixed abstention message and an empty source list.
    """
    query = query.strip()
    if not query:
        return RecallResult(answer=ABSTENTION_MESSAGE, abstained=True, sources=[])

    tried: set[str] = set()
    best: dict[uuid.UUID, ScoredUnit] = {}
    best_chunks: dict[uuid.UUID, ScoredChunk] = {}
    current_queries = [query]
    ranked: list[ScoredUnit] = []
    ranked_chunks: list[ScoredChunk] = []

    for iteration in range(MAX_AGENT_ITERATIONS):
        for q in current_queries:
            if q in tried:
                continue
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
        decision = await _evaluate(llm, query, ranked, tried)
        if decision["sufficient"]:
            break
        next_queries = [q for q in decision["reformulated_queries"][:MAX_REFORMULATIONS_PER_ROUND] if q not in tried]
        if not next_queries:
            break
        current_queries = next_queries

    if not ranked and not ranked_chunks:
        return RecallResult(answer=ABSTENTION_MESSAGE, abstained=True, sources=[])

    context = sibling_context(db, ranked)
    answer_text, grounded = await generate_answer(llm, query, ranked, context, ranked_chunks)
    if not grounded:
        return RecallResult(answer=ABSTENTION_MESSAGE, abstained=True, sources=[])
    return RecallResult(answer=answer_text, abstained=False, sources=ranked, chunk_sources=ranked_chunks)
