import datetime as dt
from typing import Any, TypedDict

from app.llm.base import LLMError, LLMProvider

# Multi-prompt extraction: three narrow, focused calls instead of one
# combined 9-field schema (see docs/DECISIONS.md). A single call asking
# for meaning + three entity lists + aliases + topics + definitions +
# relations + a date all at once spreads the model's attention across
# every field regardless of how much any one of them actually warrants;
# splitting along natural conceptual lines lets each call's full prompt
# and output budget go toward just that slice, at the cost of two extra
# sequential LLM calls per unit (~2x enrichment latency -- see the
# ingestion timing doc / docs/DECISIONS.md for the trade-off).
#
# Each call independently degrades to empty defaults on LLM failure
# (never raises) -- a down/timeout/malformed failure in one slice must
# not discard whatever the other slices already extracted.

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


class Definition(TypedDict):
    term: str
    definition: str


class Relation(TypedDict):
    subject: str
    predicate: str
    object: str


class Semantics(TypedDict):
    meaning: str
    aliases: list[str]
    people: list[str]
    organizations: list[str]
    locations: list[str]
    topics: list[str]
    definitions: list[Definition]
    relations: list[Relation]
    event_date: str
    enrichment_status: str  # "ok" | "unavailable", not part of the LLM schema


def _empty_semantics(status: str) -> Semantics:
    return Semantics(
        meaning="",
        aliases=[],
        people=[],
        organizations=[],
        locations=[],
        topics=[],
        definitions=[],
        relations=[],
        event_date="",
        enrichment_status=status,
    )


def _as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _as_definitions(value: Any) -> list[Definition]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        term = _as_str(item.get("term"))
        definition = _as_str(item.get("definition"))
        # Check the *stripped* values, not just key presence/truthiness --
        # a non-string or whitespace-only field would otherwise slip
        # through as a degenerate empty-string entry instead of being
        # filtered out, same class of defensiveness _as_str_list already
        # applies to flat lists.
        if term and definition:
            out.append(Definition(term=term, definition=definition))
    return out


def _as_relations(value: Any) -> list[Relation]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        subject = _as_str(item.get("subject"))
        predicate = _as_str(item.get("predicate"))
        obj = _as_str(item.get("object"))
        if subject and predicate and obj:
            out.append(Relation(subject=subject, predicate=predicate, object=obj))
    return out


def _format_clarifications(clarifications: list[dict]) -> str:
    if not clarifications:
        return ""
    lines = ["Previous clarifications:"]
    for c in clarifications:
        lines.append(f"Q: {c.get('question', '')}\nA: {c.get('answer', '')}")
    return "\n".join(lines)


async def _call(llm: LLMProvider, prompt: str, *, schema: dict, system: str) -> dict[str, Any] | None:
    """One sub-call of the multi-prompt split. Returns None (not raises)
    on LLM failure, so the caller can keep whatever the other slices
    already produced rather than discarding all of them."""
    try:
        return await llm.generate_structured(prompt, schema=schema, system=system)
    except LLMError:
        return None


async def enrich(
    llm: LLMProvider,
    content: str,
    clarifications: list[dict],
    *,
    now: dt.datetime | None = None,
) -> Semantics:
    """Best-effort semantic enrichment, as three independent sub-calls
    (narrative, entities, retrieval aids -- see the module docstring
    above the schemas for why). A unit must remain valid (authoritative_
    content + whatever enrichment already existed) even if the LLM is
    down for some or all of these -- see app/ingestion/pipeline.py.

    Deliberately does not decide clarification questions -- that's
    app/ingestion/clarify.py, run and fully resolved before this is ever
    called (see docs/DECISIONS.md for why these are separate calls).
    """
    today = (now or dt.datetime.now(dt.UTC)).date().isoformat()
    clar_block = _format_clarifications(clarifications)
    header = f"Today's date is {today}.\nMemory text:\n{content}\n{clar_block}\n\n"

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
        return _empty_semantics("unavailable")

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


def build_retrieval_text(
    authoritative_content: str, clarifications: list[dict], semantic: Semantics
) -> str:
    """Deterministic (non-LLM) assembly of the "derived retrieval text"
    (spec section 11): original wording + semantic enrichment, folded into
    one text embedded AND full-text indexed, so the same memory is
    reachable via its original phrasing, its aliases, or its definitions.
    Built in code rather than via another LLM call: it's a pure
    reformatting of already-extracted structured data, so a generation
    call would add latency/fragility for no benefit.
    """
    parts = [f"Original:\n{authoritative_content}"]

    if clarifications:
        qa = "\n".join(f"Q: {c.get('question', '')}\nA: {c.get('answer', '')}" for c in clarifications)
        parts.append(f"Clarifications:\n{qa}")

    if semantic.get("meaning"):
        parts.append(f"Meaning:\n{semantic['meaning']}")

    if semantic.get("aliases"):
        parts.append("Also known as: " + ", ".join(semantic["aliases"]))

    if semantic.get("definitions"):
        defs = "\n".join(f"{d['term']} = {d['definition']}" for d in semantic["definitions"])
        parts.append(f"Definitions:\n{defs}")

    if semantic.get("relations"):
        rels = "\n".join(f"{r['subject']} {r['predicate']} {r['object']}" for r in semantic["relations"])
        parts.append(f"Relations:\n{rels}")

    for label, key in (("People", "people"), ("Organizations", "organizations"), ("Locations", "locations"), ("Topics", "topics")):
        values = semantic.get(key)
        if values:
            parts.append(f"{label}: " + ", ".join(values))

    if semantic.get("event_date"):
        parts.append(f"Event date: {semantic['event_date']}")

    return "\n\n".join(parts)
