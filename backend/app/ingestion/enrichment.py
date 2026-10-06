import datetime as dt
from typing import Any, TypedDict

from app.llm.base import LLMError, LLMProvider

_SYSTEM = (
    "You derive a semantic representation of a personal memory, meant to "
    "make it recoverable later via different wording than the original "
    "(synonyms, aliases, indirect descriptions).\n\n"
    "From the memory text (and any clarifications already given), extract:\n"
    "- meaning, aliases, people, organizations, locations, topics, "
    "definitions, relations, event_date (see schema).\n"
    "- clarification_question / clarification_field: ask when the text "
    "contains a term/abbreviation/name you are NOT actually certain of the "
    "meaning of, and no clarification has resolved it yet. A wrong guessed "
    "definition is worse than asking.\n\n"
    "Example where clarification IS needed (an invented/unfamiliar term, "
    'not a real known abbreviation):\nText: "My Zorblex code is 42."\n'
    '-> clarification_question: "What is a Zorblex code?", '
    'clarification_field: "Zorblex code"\n'
    '(aliases/definitions for "Zorblex code" stay empty/unknown until '
    "answered -- do not invent a definition)\n\n"
    "Example where clarification is NOT needed (ordinary text, nothing "
    'uncertain):\nText: "I had lunch today."\n'
    '-> clarification_question: "", clarification_field: ""\n\n'
    'Leave clarification_question/_field "" whenever nothing qualifies -- '
    "most memories need no clarification. Be terse. Use \"\" or [] for "
    "anything not clearly stated or inferable."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "meaning": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "organizations": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
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
        "event_date": {"type": "string"},
        "clarification_question": {"type": "string"},
        "clarification_field": {"type": "string"},
    },
    "required": [
        "meaning",
        "aliases",
        "people",
        "organizations",
        "locations",
        "topics",
        "definitions",
        "relations",
        "event_date",
        "clarification_question",
        "clarification_field",
    ],
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
    clarification_question: str
    clarification_field: str
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
        clarification_question="",
        clarification_field="",
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
        if isinstance(item, dict) and item.get("term") and item.get("definition"):
            out.append(Definition(term=_as_str(item["term"]), definition=_as_str(item["definition"])))
    return out


def _as_relations(value: Any) -> list[Relation]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict) and all(k in item for k in ("subject", "predicate", "object")):
            out.append(
                Relation(
                    subject=_as_str(item["subject"]),
                    predicate=_as_str(item["predicate"]),
                    object=_as_str(item["object"]),
                )
            )
    return out


def _format_clarifications(clarifications: list[dict]) -> str:
    if not clarifications:
        return ""
    lines = ["Previous clarifications:"]
    for c in clarifications:
        lines.append(f"Q: {c.get('question', '')}\nA: {c.get('answer', '')}")
    return "\n".join(lines)


async def enrich(
    llm: LLMProvider,
    content: str,
    clarifications: list[dict],
    *,
    allow_clarification: bool,
    now: dt.datetime | None = None,
) -> Semantics:
    """Best-effort semantic enrichment. Never raises: a unit must remain
    valid (authoritative_content + whatever enrichment already existed)
    even if the LLM is down -- see app/ingestion/pipeline.py.

    `allow_clarification=False` discards any clarification question the
    model proposes (used once MAX_CLARIFICATION_ROUNDS is reached).
    """
    today = (now or dt.datetime.now(dt.UTC)).date().isoformat()
    clar_block = _format_clarifications(clarifications)
    prompt = f"Today's date is {today}.\nMemory text:\n{content}\n{clar_block}\n\nExtract the semantic representation."

    try:
        raw = await llm.generate_structured(prompt, schema=_SCHEMA, system=_SYSTEM)
    except LLMError:
        return _empty_semantics("unavailable")

    question = _as_str(raw.get("clarification_question")) if allow_clarification else ""
    field = _as_str(raw.get("clarification_field")) if allow_clarification else ""

    return Semantics(
        meaning=_as_str(raw.get("meaning")),
        aliases=_as_str_list(raw.get("aliases")),
        people=_as_str_list(raw.get("people")),
        organizations=_as_str_list(raw.get("organizations")),
        locations=_as_str_list(raw.get("locations")),
        topics=_as_str_list(raw.get("topics")),
        definitions=_as_definitions(raw.get("definitions")),
        relations=_as_relations(raw.get("relations")),
        event_date=_as_str(raw.get("event_date")),
        clarification_question=question,
        clarification_field=field,
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
