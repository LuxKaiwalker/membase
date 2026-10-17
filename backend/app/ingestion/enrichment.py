"""Multi-prompt semantic enrichment -- three narrow, focused calls
instead of one combined 9-field schema (restored, see docs/DECISIONS.md
"Restore the multi-call ingestion pipeline"). A single call asking for
meaning + three entity lists + aliases + topics + definitions +
relations + a date all at once spreads the model's attention across
every field regardless of how much any one of them actually warrants;
splitting along natural conceptual lines lets each call's full prompt
and output budget go toward just that slice, at the cost of two extra
sequential LLM calls per unit. This is now free latency-wise: enrichment
only ever runs inside a background task (see
app/ingestion/pipeline.py's `process_capture_fully`), never in the
synchronous request path.

Each call independently degrades to empty defaults on LLM failure
(never raises) -- a down/timeout/malformed failure in one slice must
not discard whatever the other slices already extracted.

Deliberately does not decide clarification questions -- that's
app/ingestion/clarify.py, run and fully resolved before this is ever
called.
"""

import datetime as dt
from typing import Any

from app.ingestion.analysis import (
    Semantics,
    _as_definitions,
    _as_relations,
    _as_str,
    _as_str_list,
    context_blocks,
    empty_semantics,
    format_clarifications,
)
from app.llm.base import LLMError, LLMProvider

_NARRATIVE_SYSTEM = (
    "Summarize the core meaning of this personal memory, meant to make it "
    "recoverable later via different wording than the original. Write "
    "'meaning' as one or two plain sentences capturing what this is about "
    "-- more than a restatement of the text, but nothing invented. Extract "
    "'event_date' only if the text states or clearly implies a specific "
    "date or time (resolve a relative date like \"yesterday\" to an "
    "absolute one using today's date, given below); leave it \"\" "
    "otherwise -- never guess."
)
_NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {"meaning": {"type": "string"}, "event_date": {"type": "string"}},
    "required": ["meaning", "event_date"],
}

_ENTITIES_SYSTEM = (
    "Extract every person, organization, and location explicitly named or "
    "clearly and specifically implied in this personal memory -- a "
    "well-known real-world name (a person, a company, a university, a "
    "city) counts even if the text doesn't define it. Use [] for a "
    "category with nothing in it; never invent an entity the text doesn't "
    "support."
)
_ENTITIES_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {"type": "array", "items": {"type": "string"}},
        "organizations": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["people", "organizations", "locations"],
}

_RETRIEVAL_AIDS_SYSTEM = (
    "You're finding alternative ways someone might later search for this "
    "personal memory, using different words than the original (synonyms, "
    "aliases, indirect descriptions) -- not summarizing it.\n\n"
    "- aliases: other names/terms this same thing could be called or "
    "searched by.\n"
    "- topics: general subjects/themes this relates to (a handful of "
    "short phrases, not a restatement of the text).\n"
    "- definitions: term -> meaning, for any abbreviation, jargon, or "
    "nickname actually used in the text whose meaning is already known "
    "(from the text itself or a prior clarification) -- never guess a "
    "definition you're not confident of; leave it out instead.\n"
    "- relations: subject-predicate-object triples capturing who/what did "
    "what (e.g. {subject: 'I', predicate: 'met', object: 'Anna'}).\n\n"
    "If the attached content below contains a 'Structured data:' block of "
    "label/value pairs (e.g. from a form, table, or a photographed "
    "prescription/measurement), promote EACH pair into its own `relations` "
    "entry (subject = the label, predicate = 'is', object = the value, "
    "transcribed exactly, numbers and units included) -- do not just "
    "summarize that block in prose; every labeled value in it should be "
    "individually retrievable as a relation.\n\n"
    "Use [] for anything not clearly applicable -- most memories don't "
    "need every category filled."
)
_RETRIEVAL_AIDS_SCHEMA = {
    "type": "object",
    "properties": {
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
    },
    "required": ["aliases", "topics", "definitions", "relations"],
}


async def _call(llm: LLMProvider, prompt: str, *, schema: dict, system: str) -> dict[str, Any] | None:
    """One sub-call of the multi-prompt split. Returns None (not raises)
    on LLM failure, so the caller can keep whatever the other slices
    already produced rather than discarding all of them."""
    try:
        return await llm.generate_structured(prompt, schema=schema, system=system)
    except LLMError:
        return None


async def enrich_unit(
    llm: LLMProvider,
    content: str,
    clarifications: list[dict],
    attachment_text: str,
    prior_context: list[str] | None = None,
    *,
    now: dt.datetime | None = None,
) -> Semantics:
    """Best-effort semantic enrichment, as three independent sub-calls
    (narrative, entities, retrieval aids).

    `attachment_text` is required, not an afterthought kwarg -- an
    attached file (including an uploaded image's caption and any
    structured data extracted from it) is real content about the
    memory, so meaning/entities/aliases extraction must see it, the same
    way clarify.generate_question() does.
    """
    today = (now or dt.datetime.now(dt.UTC)).date().isoformat()
    context = context_blocks(attachment_text, prior_context or [])
    clar_block = format_clarifications(clarifications)
    header = f"Today's date is {today}.\n{context}{clar_block}Memory text:\n{content}\n\n"

    narrative = await _call(
        llm, header + "Extract the core meaning and, if present, the event date.",
        schema=_NARRATIVE_SCHEMA, system=_NARRATIVE_SYSTEM,
    )
    entities = await _call(
        llm, header + "Extract the people, organizations, and locations involved.",
        schema=_ENTITIES_SCHEMA, system=_ENTITIES_SYSTEM,
    )
    aids = await _call(
        llm, header + "Extract aliases, topics, definitions, and relations.",
        schema=_RETRIEVAL_AIDS_SCHEMA, system=_RETRIEVAL_AIDS_SYSTEM,
    )

    if narrative is None and entities is None and aids is None:
        return empty_semantics("unavailable")

    narrative = narrative or {}
    entities = entities or {}
    aids = aids or {}

    return Semantics(
        meaning=_as_str(narrative.get("meaning")),
        event_date=_as_str(narrative.get("event_date")),
        people=_as_str_list(entities.get("people")),
        organizations=_as_str_list(entities.get("organizations")),
        locations=_as_str_list(entities.get("locations")),
        aliases=_as_str_list(aids.get("aliases")),
        topics=_as_str_list(aids.get("topics")),
        definitions=_as_definitions(aids.get("definitions")),
        relations=_as_relations(aids.get("relations")),
        enrichment_status="ok",
    )
