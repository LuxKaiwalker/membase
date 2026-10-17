"""Dedicated segmentation call -- restored as its own focused step (see
docs/DECISIONS.md "Restore the multi-call ingestion pipeline"). Decides
only how to split a capture into independently-retrievable units;
deliberately produces no semantic fields at all, so its output schema
and output budget are both tiny regardless of how many units result,
leaving the full budget for app/ingestion/clarify.py and
app/ingestion/enrichment.py's own calls.
"""

import logging

from app.ingestion.analysis import MAX_SEGMENT_UNITS, _chunk_text, _split_sentences, is_verbatim
from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

# Conservative character budget for one segment_capture() call's input
# text, chosen so a chunk's worth of proposed units comfortably fits a
# 4096-token context alongside the system prompt -- see
# docs/DECISIONS.md "Chunk long captures before analysis". Not
# swept/measured against the model's actual token budget, just picked
# with generous headroom; see docs/TODO.md.
MAX_SEGMENT_CHARS_PER_CALL = 600

_SYSTEM = (
    "You split a personal memory note into one or more independently-useful "
    'retrieval units, without destroying meaning. Each unit\'s "content" MUST '
    "be an exact verbatim excerpt from the original text -- copy the exact "
    "words, do not paraphrase or summarize.\n\n"
    "Decide splits based on whether facts would likely be recalled "
    "independently later, not based on sentence boundaries. Keep tightly "
    "connected information (same event, same people, one episode) together "
    "as ONE unit. Split apart facts that are topically unrelated and would "
    "be searched for separately.\n\n"
    "Examples:\n"
    '- "I went to the ETH seminar yesterday and Anna, who studies robotics, '
    'recommended a paper on tactile sensing." -> ONE unit (one event, one '
    "person, connected facts).\n"
    '- "My AHV number is 123456789. Anna is studying robotics. I need to '
    'renew my passport next March." -> THREE units (three unrelated '
    "facts)."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        }
    },
    "required": ["units"],
}


def _sentence_fallback(text: str) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]
    return sentences[:MAX_SEGMENT_UNITS]


async def _segment_one_call(llm: LLMProvider, text: str, raw_text: str) -> list[str]:
    """Covers `text` (either the whole capture, or one chunk of it -- see
    segment_capture()). Verbatim-checks every proposed unit against the
    CAPTURE's full `raw_text`, not just `text`, so correctness never
    depends on where a chunk boundary happened to fall."""
    prompt = f"Text:\n{text}\n\nSplit this into memory units."
    try:
        raw = await llm.generate_structured(prompt, schema=_SCHEMA, system=_SYSTEM)
    except LLMError:
        logger.warning("Segmentation LLM call failed; falling back to a deterministic sentence split")
        return _sentence_fallback(text)

    units_raw = raw.get("units")
    if not isinstance(units_raw, list) or not units_raw:
        return _sentence_fallback(text)

    proposed: list[str] = []
    for u in units_raw[:MAX_SEGMENT_UNITS]:
        if not isinstance(u, dict):
            continue
        content = u.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if not is_verbatim(content, raw_text):
            logger.warning("Segmentation proposed a non-verbatim unit; falling back to a deterministic sentence split")
            return _sentence_fallback(text)
        proposed.append(content.strip())

    return proposed or _sentence_fallback(text)


async def segment_capture(llm: LLMProvider, raw_text: str) -> list[str]:
    """Decide how to split `raw_text` into independently-retrievable
    units. Returns the verbatim `content` string for each proposed unit.

    Never raises. Falls back to a deterministic sentence split (or the
    whole text, if it's a single sentence) if the LLM is unavailable,
    returns something malformed, proposes zero units, or proposes any
    unit that isn't a verbatim excerpt of raw_text -- a paraphrased
    "unit" would silently promote LLM wording to authoritative status,
    which docs/DECISIONS.md rules out.

    A long/dense capture is first deterministically split into
    sentence-bounded chunks (never reconstructed, so verbatim-ness is
    unaffected), segmented one call per chunk, and the results
    concatenated.
    """
    chunks = _chunk_text(raw_text, MAX_SEGMENT_CHARS_PER_CALL)
    if len(chunks) == 1:
        return await _segment_one_call(llm, chunks[0], raw_text)

    results: list[str] = []
    for chunk in chunks:
        results.extend(await _segment_one_call(llm, chunk, raw_text))
        if len(results) >= MAX_SEGMENT_UNITS:
            break
    return results[:MAX_SEGMENT_UNITS] if results else [raw_text]
