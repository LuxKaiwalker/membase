"""Shared substrate for the ingestion pipeline's restored multi-call
architecture (see docs/DECISIONS.md "Restore the multi-call ingestion
pipeline"): the `Semantics`/`UnitAnalysis` shapes, JSON-parsing helpers,
the verbatim/chunking utilities segmentation relies on, and
`build_retrieval_text`. The three LLM-calling steps themselves --
segmentation, clarification-question generation, and semantic enrichment
-- live in their own sibling modules (`segmentation.py`, `clarify.py`,
`enrichment.py`), each a dedicated, focused call rather than one combined
schema, since a unit now has no response-latency pressure forcing them
back together (all of this runs in a background task -- see
app/ingestion/pipeline.py's `process_capture_fully`).
"""

import re
from dataclasses import dataclass
from typing import Any, TypedDict

# A runaway segmentation (one unit per word, say) would be a bigger
# problem than under-splitting -- cap it rather than trust the model
# unconditionally. Unvalidated, hand-picked; see docs/TODO.md.
MAX_SEGMENT_UNITS = 24

_SENTENCE_END_RE = re.compile(r"[.!?](?=\s+[A-Z0-9]|\s*$)")


def _split_sentences(text: str) -> list[str]:
    """Deterministic, lossless sentence split -- every piece is an exact
    substring of `text` (never rewritten/rejoined), so anything built
    from this is trivially verbatim. Used both to chunk long input
    before it ever reaches the model, and as the degraded-fallback
    segmentation when a model call fails outright."""
    text = text.strip()
    if not text:
        return []
    positions = [m.end() for m in _SENTENCE_END_RE.finditer(text)]
    if not positions or positions[-1] != len(text):
        positions.append(len(text))
    sentences = []
    start = 0
    for end in positions:
        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = end
    return sentences


def _chunk_text(raw_text: str, max_chars: int) -> list[str]:
    """Split raw_text into sentence-bounded chunks each at most
    max_chars long (a single sentence longer than max_chars stays whole
    rather than being cut mid-word). Every chunk is an exact contiguous
    substring of raw_text -- never reconstructed -- so a unit verified
    verbatim against raw_text is unaffected by where chunk boundaries
    fall."""
    if len(raw_text) <= max_chars:
        return [raw_text]
    boundaries = [m.end() for m in _SENTENCE_END_RE.finditer(raw_text)]
    if not boundaries or boundaries[-1] != len(raw_text):
        boundaries.append(len(raw_text))

    chunks: list[str] = []
    chunk_start = 0
    last_boundary = 0
    for b in boundaries:
        if b - chunk_start > max_chars and last_boundary > chunk_start:
            chunks.append(raw_text[chunk_start:last_boundary])
            chunk_start = last_boundary
        last_boundary = b
    chunks.append(raw_text[chunk_start:last_boundary])
    return [c for c in chunks if c.strip()]


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


@dataclass
class UnitAnalysis:
    """Bridges segmentation + clarify + enrichment into one shape for
    `app/ingestion/pipeline.py`'s `_unit_from_analysis()` -- unchanged by
    which of those three modules actually populated each field."""

    content: str
    semantic: Semantics
    clarification_question: str = ""
    clarification_field: str = ""


# Observed for real against the production model (gemma4:12b): for an
# empty/unset field it sometimes emits the literal string "null" (or
# "none") instead of "" -- a known structured-output quirk, not a real
# value. Treated as empty everywhere a string is parsed out of a model
# response, so it never leaks into retrieval_text or a clarification
# conflict check as if it were real content.
_NULL_LIKE = {"null", "none", "n/a"}


def _as_str(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    return "" if stripped.lower() in _NULL_LIKE else stripped


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if not isinstance(v, str):
            continue
        stripped = v.strip()
        if stripped and stripped.lower() not in _NULL_LIKE:
            out.append(stripped)
    return out


def _as_definitions(value: Any) -> list[Definition]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        term = _as_str(item.get("term"))
        definition = _as_str(item.get("definition"))
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


def _parse_semantics(raw: dict) -> Semantics:
    """Parses a single combined dict carrying all 9 semantic fields at
    once -- used by app/ingestion/structurer.py's own unified schema
    (the "membase structurer" deep second pass still asks for everything
    in one call, since it's a quality-only pass with no clarification
    decision riding on it, unlike the fast pipeline's restored 3-way
    split in app/ingestion/enrichment.py)."""
    return Semantics(
        meaning=_as_str(raw.get("meaning")),
        event_date=_as_str(raw.get("event_date")),
        people=_as_str_list(raw.get("people")),
        organizations=_as_str_list(raw.get("organizations")),
        locations=_as_str_list(raw.get("locations")),
        aliases=_as_str_list(raw.get("aliases")),
        topics=_as_str_list(raw.get("topics")),
        definitions=_as_definitions(raw.get("definitions")),
        relations=_as_relations(raw.get("relations")),
        enrichment_status="ok",
    )


def empty_semantics(status: str = "unavailable") -> Semantics:
    return Semantics(
        meaning="", aliases=[], people=[], organizations=[], locations=[],
        topics=[], definitions=[], relations=[], event_date="",
        enrichment_status=status,
    )


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def is_verbatim(candidate: str, source: str) -> bool:
    """Word-boundary-aware verbatim check: candidate must appear as a
    contiguous run of whole words in source, not merely as a character
    substring (see docs/DECISIONS.md -- this safety net is unchanged by
    the restored multi-call pipeline: a proposed unit must still be an
    actual excerpt, never LLM-paraphrased authoritative text)."""
    candidate_words = _normalize(candidate).split()
    source_words = _normalize(source).split()
    n = len(candidate_words)
    if n == 0:
        return False
    return any(source_words[i : i + n] == candidate_words for i in range(len(source_words) - n + 1))


def context_blocks(attachment_text: str, prior_context: list[str]) -> str:
    """Shared prompt-context formatting -- used by clarify.generate_question()
    and enrichment.enrich_unit() alike, so both see exactly the same
    attachment/prior-memory context in exactly the same words."""
    blocks = []
    if attachment_text:
        blocks.append(f"Attached file content:\n{attachment_text}")
    if prior_context:
        joined = "\n---\n".join(prior_context)
        blocks.append(
            "Potentially related earlier memories (context only -- the "
            f"current text is authoritative if they conflict):\n{joined}"
        )
    return ("\n\n".join(blocks) + "\n\n") if blocks else ""


def format_clarifications(clarifications: list[dict]) -> str:
    if not clarifications:
        return ""
    lines = ["Clarification history:"]
    for c in clarifications:
        outcome = c.get("outcome", "answered")
        lines.append(f"Q: {c.get('question', '')}\nA ({outcome}): {c.get('answer', '')}")
    return "\n".join(lines) + "\n\n"


def build_retrieval_text(authoritative_content: str, clarifications: list[dict], semantic: Semantics) -> str:
    """Deterministic (non-LLM) assembly of the "derived retrieval text":
    original wording + semantic enrichment, folded into one text embedded
    AND full-text indexed, so the same memory is reachable via its
    original phrasing, its aliases, or its definitions."""
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
