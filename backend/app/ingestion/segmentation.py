import logging
from dataclasses import dataclass

from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

# A runaway segmentation (one unit per word, say) would be a bigger
# problem than under-splitting -- cap it rather than trust the model
# unconditionally. Unvalidated, hand-picked; see docs/TODO.md.
MAX_SEGMENT_UNITS = 8

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
    "facts).\n\n"
    'independence is "independent" if the unit stands alone, or "contextual" '
    "if it's part of a larger episode that was kept together with other "
    "units from the same note."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "reason": {"type": "string"},
                    "independence": {"type": "string"},
                },
                "required": ["content", "reason", "independence"],
            },
        }
    },
    "required": ["units"],
}


@dataclass
class ProposedUnit:
    content: str
    reason: str
    independence: str


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _is_verbatim(candidate: str, source: str) -> bool:
    return _normalize(candidate) in _normalize(source)


async def segment_capture(llm: LLMProvider, raw_text: str) -> list[ProposedUnit]:
    """Decide how to split `raw_text` into independently-retrievable units.

    Never raises. Falls back to a single whole-text unit (no split) if the
    LLM is unavailable, returns something malformed, proposes zero units,
    or proposes any unit that isn't a verbatim excerpt of raw_text -- a
    paraphrased "unit" would silently promote LLM wording to authoritative
    status, which docs/DECISIONS.md rules out. This fallback is also, in
    practice, what keeps genuinely coherent episodes from being shredded
    when the model's sentence-level punctuation drifts from the original
    (see docs/TODO.md for the failure mode this papers over).
    """
    prompt = f"Text:\n{raw_text}\n\nSplit this into memory units."
    try:
        raw = await llm.generate_structured(prompt, schema=_SCHEMA, system=_SYSTEM)
    except LLMError:
        logger.warning("Segmentation LLM call failed; keeping capture as one unit")
        return [ProposedUnit(content=raw_text, reason="segmentation unavailable", independence="independent")]

    units_raw = raw.get("units")
    if not isinstance(units_raw, list) or not units_raw:
        return [ProposedUnit(content=raw_text, reason="model proposed no split", independence="independent")]

    proposed: list[ProposedUnit] = []
    for u in units_raw[:MAX_SEGMENT_UNITS]:
        if not isinstance(u, dict):
            continue
        content = u.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if not _is_verbatim(content, raw_text):
            logger.warning("Segmentation proposed a non-verbatim unit; falling back to no-split")
            return [ProposedUnit(content=raw_text, reason="non-verbatim split rejected", independence="independent")]
        proposed.append(
            ProposedUnit(
                content=content.strip(),
                reason=u.get("reason", "") if isinstance(u.get("reason"), str) else "",
                independence=u.get("independence", "independent")
                if isinstance(u.get("independence"), str)
                else "independent",
            )
        )

    if not proposed:
        return [ProposedUnit(content=raw_text, reason="no valid units after filtering", independence="independent")]
    return proposed
