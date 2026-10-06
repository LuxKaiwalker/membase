import datetime as dt
from typing import Any, TypedDict

from app.llm.base import LLMError, LLMProvider

_SYSTEM = (
    "You extract structured metadata from a personal memory note. Be terse. "
    'Use empty string "" for any field that is not clearly stated in the text -- '
    "never guess or explain, never write a sentence into a field that expects a "
    "short value. Only fill clarification_question if resolving it would "
    "materially help retrieving this memory later (e.g. an ambiguous relative "
    "date like 'yesterday' or 'last week' that you cannot resolve from today's "
    "date, or a person referred to only by role with no name). Do NOT ask "
    'clarification questions for trivia, minor details, or anything already '
    'clear. Leave clarification_question as "" in most cases -- most memories '
    "need no clarification at all."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {"type": "array", "items": {"type": "string"}},
        "location": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "event_date": {"type": "string"},
        "summary": {"type": "string"},
        "clarification_question": {"type": "string"},
        "clarification_field": {"type": "string"},
    },
    "required": [
        "people",
        "location",
        "topics",
        "event_date",
        "summary",
        "clarification_question",
        "clarification_field",
    ],
}


class ExtractedMetadata(TypedDict):
    people: list[str]
    location: str
    topics: list[str]
    event_date: str
    summary: str
    clarification_question: str
    clarification_field: str


EMPTY_METADATA: ExtractedMetadata = {
    "people": [],
    "location": "",
    "topics": [],
    "event_date": "",
    "summary": "",
    "clarification_question": "",
    "clarification_field": "",
}


async def extract_metadata(
    llm: LLMProvider, caption: str, attachment_text: str, *, now: dt.datetime | None = None
) -> ExtractedMetadata | None:
    """Best-effort metadata extraction. Returns None (never raises) if the
    LLM is unavailable or returns something unusable -- ingestion must still
    succeed with the authoritative caption alone (see docs/DECISIONS.md).
    """
    today = (now or dt.datetime.now(dt.UTC)).date().isoformat()
    text = caption if not attachment_text else f"{caption}\n\n[Attached file text excerpt]\n{attachment_text[:2000]}"
    prompt = (
        f"Today's date is {today}.\nMemory text:\n{text}\n\n"
        'Extract: people (array of names), location (short string or ""), '
        "topics (array of short topic phrases), event_date (ISO 8601 date if "
        "resolvable from the text and today's date, else \"\"), summary (one "
        "short sentence), clarification_question, clarification_field."
    )
    try:
        raw = await llm.generate_structured(prompt, schema=_SCHEMA, system=_SYSTEM)
    except LLMError:
        return None

    return ExtractedMetadata(
        people=_as_str_list(raw.get("people")),
        location=_as_str(raw.get("location")),
        topics=_as_str_list(raw.get("topics")),
        event_date=_as_str(raw.get("event_date")),
        summary=_as_str(raw.get("summary")),
        clarification_question=_as_str(raw.get("clarification_question")),
        clarification_field=_as_str(raw.get("clarification_field")),
    )


def _as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]
