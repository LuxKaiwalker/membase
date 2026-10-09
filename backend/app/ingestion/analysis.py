"""Unified semantic analysis for ingestion (streamlining phase -- see
docs/DECISIONS.md "Collapse ingestion LLM calls").

Replaces the previous five-call pipeline (segment_capture +
generate_questions + enrich's three narrative/entities/retrieval-aids
sub-calls) with two calls, used at different points in a unit's
lifecycle:

- `analyze_capture()`: one call per *capture*. Decides segmentation
  (how many independently-retrievable units), and for each proposed
  unit, its meaning/entities/aliases/topics/relations/definitions AND
  whether it needs a clarifying question -- all at once. This is the
  only call an ordinary memory with no clarification needed ever pays
  for (plus one embedding call -- see app/ingestion/pipeline.py).
- `reanalyze_unit()`: one call per *clarification round*, for a single
  already-segmented unit. Takes the accumulated clarification history
  and re-decides the same semantic fields plus whether a *further*
  question is still needed -- the explicit "re-evaluate after every
  answer" loop (see docs/DECISIONS.md "Sequential clarification"),
  never a pre-generated batch of questions.

Both calls share one goal: produce a *useful* retrieval representation
with the fewest model invocations, not a perfect one. A field left
empty is not a bug.

**Long/dense captures** (see docs/DECISIONS.md "Chunk long captures
before analysis"): asking for full per-unit semantics (9 fields) for
many units in one structured-output call can exceed the generation
model's context window on its OUTPUT side alone -- observed for real
against gemma4:12b's 4096-token serving context with a ~20-independent-
fact capture, which produced truncated/invalid JSON and silently
degraded to one giant unopposed unit with empty semantics. `raw_text`
is deterministically pre-split (by sentence, never reconstructed -- see
`_chunk_text`) into chunks that comfortably fit one call's output
budget *before* ever calling the model; an ordinary short capture is
exactly one chunk, so this adds no extra call for the common case.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

# A runaway segmentation (one unit per word, say) would be a bigger
# problem than under-splitting -- cap it rather than trust the model
# unconditionally. Raised from 8 once chunking (below) made a large,
# genuinely multi-fact capture a correctly-handled case instead of a
# single-call-runaway concern. Unvalidated, hand-picked; see
# docs/TODO.md.
MAX_SEGMENT_UNITS = 24

# Conservative character budget for ONE analyze_capture()/_analyze_one_call
# call's input text, chosen so a chunk's worth of full per-unit semantic
# output (several units x 9 fields each) comfortably fits a 4096-token
# context alongside the system prompt -- see the module docstring above.
# Not swept/measured against the model's actual token budget, just
# picked with generous headroom; see docs/TODO.md.
MAX_ANALYSIS_CHARS_PER_CALL = 600

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
    content: str
    semantic: Semantics
    clarification_question: str = ""
    clarification_field: str = ""
    independence: str = "independent"


# --- shared per-unit JSON-schema fragment -----------------------------
# Reused by both the capture-level (segmentation + analysis) schema and
# the unit-level re-analysis schema, so the two calls ask for exactly
# the same semantic shape -- only whether "content" is included differs
# (a unit's authoritative_content is fixed once segmentation picks it;
# re-analysis never gets to rewrite it).

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
    "clarification_question": {
        "type": "string",
        "description": "A concrete question, or \"\" if nothing needs asking.",
    },
    "clarification_field": {"type": "string"},
}
_SEMANTIC_REQUIRED = [
    "meaning", "event_date", "people", "organizations", "locations",
    "aliases", "topics", "definitions", "relations",
    "clarification_question", "clarification_field",
]

_CAPTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"content": {"type": "string"}, **_SEMANTIC_PROPERTIES},
                "required": ["content", *_SEMANTIC_REQUIRED],
            },
        }
    },
    "required": ["units"],
}

_UNIT_SCHEMA = {
    "type": "object",
    "properties": dict(_SEMANTIC_PROPERTIES),
    "required": list(_SEMANTIC_REQUIRED),
}

_SHARED_INSTRUCTIONS = (
    "Clarification policy: only ask about something INVENTED, PERSONAL, "
    "or genuinely AMBIGUOUS that a knowledgeable person could not resolve "
    "from general knowledge -- a made-up term, an unexplained personal "
    "nickname, an abbreviation with no obvious real-world meaning, or a "
    "pronoun/reference whose identity materially affects how this memory "
    "would later be searched for or understood (e.g. \"I talked to him "
    "about the project\" -- who 'him' is matters). Do NOT ask about a "
    "well-known real-world name, abbreviation, or institution just "
    "because you're inferring it from context. A missing field is not, "
    "by itself, grounds for a question -- store a sufficiently useful "
    "memory, not a perfectly complete record. Leave clarification_question "
    "as \"\" whenever nothing qualifies; most memories need no question "
    "at all. If attached file content or earlier related memories are "
    "given below, treat them as already-known context, not something to "
    "ask about -- a vague reference they already resolve is not "
    "ambiguous. If a field already has a declined/unknown answer noted "
    "below, do not ask about it again."
)


def _context_blocks(attachment_text: str, prior_context: list[str]) -> str:
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


def _format_clarifications(clarifications: list[dict]) -> str:
    if not clarifications:
        return ""
    lines = ["Clarification history:"]
    for c in clarifications:
        outcome = c.get("outcome", "answered")
        lines.append(f"Q: {c.get('question', '')}\nA ({outcome}): {c.get('answer', '')}")
    return "\n".join(lines) + "\n\n"


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
    the ingestion collapse: a proposed unit must still be an actual
    excerpt, never LLM-paraphrased authoritative text)."""
    candidate_words = _normalize(candidate).split()
    source_words = _normalize(source).split()
    n = len(candidate_words)
    if n == 0:
        return False
    return any(source_words[i : i + n] == candidate_words for i in range(len(source_words) - n + 1))


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


def _sentence_fallback(text: str, status: str) -> list[UnitAnalysis]:
    """Degraded-fallback segmentation: split `text` by sentence instead
    of keeping it as one opaque blob, so a failed/malformed/non-verbatim
    model response still leaves each individual fact independently
    retrievable (no semantics, but full-text searchable on its own) --
    see docs/DECISIONS.md "Chunk long captures before analysis". Every
    sentence is an exact substring of `text` (via `_split_sentences`),
    so this never risks fabricating content."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [UnitAnalysis(content=text, semantic=empty_semantics(status))]
    return [UnitAnalysis(content=s, semantic=empty_semantics(status)) for s in sentences[:MAX_SEGMENT_UNITS]]


async def _analyze_one_call(
    llm: LLMProvider,
    text: str,
    raw_text: str,
    attachment_text: str,
    prior_context: list[str],
) -> list[UnitAnalysis]:
    """One structured-output call covering `text` (either the whole
    capture, or one chunk of it -- see analyze_capture()). Verbatim-
    checks every proposed unit against the CAPTURE's full `raw_text`,
    not just `text`, so correctness never depends on where a chunk
    boundary happened to fall.
    """
    context = _context_blocks(attachment_text, prior_context)
    system = (
        "You analyze a personal memory note for a retrieval system, in one "
        "pass. First, split the text into one or more independently-useful "
        'retrieval units ("content" MUST be an exact verbatim excerpt -- '
        "copy the exact words, never paraphrase). Decide splits based on "
        "whether facts would likely be recalled independently later, not "
        "sentence boundaries: keep one event/episode together as ONE unit; "
        "split apart topically unrelated facts.\n\n"
        "For EACH unit, also extract its meaning, entities (people/"
        "organizations/locations), retrieval aids (aliases/topics/"
        "definitions/relations), and decide whether it needs a "
        "clarifying question.\n\n" + _SHARED_INSTRUCTIONS + "\n\n"
        "Example: \"My AHV number is 123456789. Anna is studying robotics. "
        "I need to renew my passport next March.\" -> THREE units (three "
        "unrelated facts). \"I met Anna at ETH yesterday and she "
        "recommended a paper on tactile sensing.\" -> ONE unit (one event)."
    )
    prompt = f"{context}Text:\n{text}\n\nAnalyze this."

    try:
        raw = await llm.generate_structured(prompt, schema=_CAPTURE_SCHEMA, system=system)
    except LLMError:
        logger.warning("Capture analysis LLM call failed; falling back to a deterministic sentence split")
        return _sentence_fallback(text, "unavailable")

    units_raw = raw.get("units")
    if not isinstance(units_raw, list) or not units_raw:
        return _sentence_fallback(text, "ok")

    proposed: list[UnitAnalysis] = []
    for u in units_raw[:MAX_SEGMENT_UNITS]:
        if not isinstance(u, dict):
            continue
        content = u.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if not is_verbatim(content, raw_text):
            logger.warning("Analysis proposed a non-verbatim unit; falling back to a deterministic sentence split")
            return _sentence_fallback(text, "ok")
        proposed.append(
            UnitAnalysis(
                content=content.strip(),
                semantic=_parse_semantics(u),
                clarification_question=_as_str(u.get("clarification_question")),
                clarification_field=_as_str(u.get("clarification_field")) or "clarification",
            )
        )

    if not proposed:
        return _sentence_fallback(text, "ok")
    return proposed


async def analyze_capture(
    llm: LLMProvider,
    raw_text: str,
    attachment_text: str = "",
    prior_context: list[str] | None = None,
) -> list[UnitAnalysis]:
    """Decide segmentation and analyze each resulting unit.

    An ordinary capture is exactly one call: see `_analyze_one_call`. A
    long/dense capture (more than `MAX_ANALYSIS_CHARS_PER_CALL`) is
    first deterministically split into sentence-bounded chunks
    (`_chunk_text` -- never reconstructed, so verbatim-ness is
    unaffected), analyzed one call per chunk, and the results
    concatenated -- see the module docstring and docs/DECISIONS.md
    "Chunk long captures before analysis" for why: asking one call to
    produce full semantics for many units at once can exceed the
    model's output budget and silently degrade the *entire* capture to
    one unopposed unit, which chunking prevents by construction.

    Never raises. Any individual call that fails (LLM unavailable,
    malformed output, a non-verbatim proposal) degrades to a
    deterministic sentence split of just that chunk, never the whole
    capture collapsing to one blob.
    """
    prior_context = prior_context or []
    chunks = _chunk_text(raw_text, MAX_ANALYSIS_CHARS_PER_CALL)
    if len(chunks) == 1:
        return await _analyze_one_call(llm, chunks[0], raw_text, attachment_text, prior_context)

    results: list[UnitAnalysis] = []
    for chunk in chunks:
        results.extend(await _analyze_one_call(llm, chunk, raw_text, attachment_text, prior_context))
        if len(results) >= MAX_SEGMENT_UNITS:
            break
    return results[:MAX_SEGMENT_UNITS] if results else [UnitAnalysis(content=raw_text, semantic=empty_semantics("ok"))]


async def reanalyze_unit(
    llm: LLMProvider,
    content: str,
    clarifications: list[dict],
    attachment_text: str = "",
    prior_context: list[str] | None = None,
) -> UnitAnalysis:
    """One call: re-derive a single already-segmented unit's semantic
    fields and decide whether a *further* clarification is still needed,
    given everything answered so far. Never raises -- degrades to empty
    semantics (or whatever a caller already had, which it keeps) on
    failure.
    """
    context = _context_blocks(attachment_text, prior_context or [])
    clar_block = _format_clarifications(clarifications)
    system = (
        "You are re-analyzing a personal memory after a clarification "
        "round. Extract its meaning, entities, and retrieval aids given "
        "everything known so far, and decide whether a FURTHER "
        "clarifying question is still warranted.\n\n" + _SHARED_INSTRUCTIONS
    )
    prompt = f"{context}{clar_block}Memory text:\n{content}\n\nAnalyze this."

    try:
        raw = await llm.generate_structured(prompt, schema=_UNIT_SCHEMA, system=system)
    except LLMError:
        return UnitAnalysis(content=content, semantic=empty_semantics("unavailable"))

    return UnitAnalysis(
        content=content,
        semantic=_parse_semantics(raw),
        clarification_question=_as_str(raw.get("clarification_question")),
        clarification_field=_as_str(raw.get("clarification_field")) or "clarification",
    )
