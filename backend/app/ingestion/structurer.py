"""The "membase structurer" (see docs/DECISIONS.md "The 'membase
structurer'"): a second, independent background pass that deepens an
already-`"ready"` unit's representation -- richer semantics, and
advisory links to related memories -- without ever affecting whether
the unit is searchable. Deliberately allowed to be slower/heavier than
anything in app/ingestion/analysis.py, since nothing user-facing is
waiting on it.

This reuses analysis.py's `Semantics` shape and parsing helpers (the
same fields, just derived more thoroughly) rather than inventing a
second representation retrieval would need to know about -- a
structured unit is a `ready` unit with better data in the exact same
columns `hybrid_search()` already reads.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.analysis import Semantics, _parse_semantics, empty_semantics
from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

# How many candidate related memories the structurer is shown per unit
# -- bounded for the same reason membase_conflict_check_candidates is:
# this stays a cheap, reused search, not an exhaustive scan of the store.
MAX_RELATED_CANDIDATES = 5

_SEMANTIC_PROPERTIES: dict[str, Any] = {
    "meaning": {"type": "string"},
    "event_date": {"type": "string"},
    "people": {"type": "array", "items": {"type": "string"}},
    "organizations": {"type": "array", "items": {"type": "string"}},
    "locations": {"type": "array", "items": {"type": "string"}},
    "aliases": {"type": "array", "items": {"type": "string"}},
    "topics": {"type": "array", "items": {"type": "string"}},
    "definitions": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
            "required": ["term", "definition"],
        },
    },
    "relations": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
            },
            "required": ["subject", "predicate", "object"],
        },
    },
}
_SEMANTIC_REQUIRED = [
    "meaning", "event_date", "people", "organizations", "locations", "aliases", "topics", "definitions", "relations",
]

_STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        **_SEMANTIC_PROPERTIES,
        "related_indices": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Indices (from the numbered candidate list) of memories that are genuinely related to this one. Empty if none are.",
        },
    },
    "required": [*_SEMANTIC_REQUIRED, "related_indices"],
}

_STRUCTURE_SYSTEM = (
    "You are doing a slow, careful SECOND PASS over a personal memory "
    "that is already saved and searchable -- there is no time pressure "
    "and no user waiting on this. Re-derive its full semantic "
    "representation as thoroughly as you can: meaning, people, "
    "organizations, locations, aliases, topics, definitions, relations, "
    "event date. Be more thorough than a quick first pass would be --  "
    "surface implicit entities, indirect relations, and useful aliases "
    "a fast pass might have missed. Do not invent facts not supported "
    "by the text.\n\n"
    "You are also given a numbered list of OTHER memories that might be "
    "related. For each, decide if it is genuinely related to THIS "
    "memory (same topic, event, person, or project -- not just a "
    "superficial word overlap). Return the indices of the ones that "
    "really are; an empty list is a perfectly fine answer if none are."
)


@dataclass
class StructureResult:
    semantic: Semantics
    related_indices: list[int] = field(default_factory=list)


def _format_candidates(candidates: list[tuple[str, str]]) -> str:
    if not candidates:
        return "No other memories to compare against."
    lines = [f"{i}. {text}" for i, (_unit_id, text) in enumerate(candidates)]
    return "Candidate other memories:\n" + "\n".join(lines)


async def structure_unit(
    llm: LLMProvider,
    content: str,
    clarifications: list[dict],
    attachment_text: str,
    candidates: list[tuple[str, str]],
) -> StructureResult:
    """One (deliberately unhurried) call: re-derive full semantics for
    an already-ready unit, and decide which of `candidates`
    (`[(unit_id, text), ...]`) are genuinely related. Never raises --
    degrades to empty/unavailable semantics and no related links on
    failure, exactly like analysis.py's calls, so a structuring failure
    never corrupts anything already stored (the caller is responsible
    for not committing a failed result over a good one).
    """
    clar_block = ""
    if clarifications:
        qa = "\n".join(f"Q: {c.get('question', '')}\nA: {c.get('answer', '')}" for c in clarifications)
        clar_block = f"Clarification history:\n{qa}\n\n"
    attachment_block = f"Attached file content:\n{attachment_text}\n\n" if attachment_text else ""
    prompt = (
        f"{attachment_block}{clar_block}Memory text:\n{content}\n\n"
        f"{_format_candidates(candidates)}\n\nAnalyze this thoroughly."
    )

    try:
        raw = await llm.generate_structured(prompt, schema=_STRUCTURE_SCHEMA, system=_STRUCTURE_SYSTEM)
    except LLMError:
        logger.warning("Structuring call failed; leaving unit's existing representation untouched")
        return StructureResult(semantic=empty_semantics("unavailable"), related_indices=[])

    semantic = _parse_semantics(raw)
    raw_indices = raw.get("related_indices")
    indices = [i for i in raw_indices if isinstance(i, int) and 0 <= i < len(candidates)] if isinstance(raw_indices, list) else []
    return StructureResult(semantic=semantic, related_indices=sorted(set(indices))[:MAX_RELATED_CANDIDATES])
