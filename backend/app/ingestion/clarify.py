"""Dedicated clarification-question call -- restored as its own focused
step, separate from app/ingestion/enrichment.py's semantic-derivation
call (see docs/DECISIONS.md "Restore the multi-call ingestion
pipeline").

docs/DECISIONS.md records a real quality regression from the collapsed
single-call design: a dedicated call, with no other task competing for
its attention, caught an invented term ("Flimvex") the unified call
missed. Splitting this back out restores that focus -- and since
clarification is decided before enrichment runs (see
app/ingestion/pipeline.py's `process_capture_fully`), a unit about to be
blocked on a question never wastes an enrichment call first.
"""

from app.ingestion.analysis import context_blocks, format_clarifications
from app.llm.base import LLMError, LLMProvider

_SCHEMA = {
    "type": "object",
    "properties": {
        "clarification_question": {
            "type": "string",
            "description": "A concrete question, or \"\" if nothing needs asking.",
        },
        "clarification_field": {"type": "string"},
    },
    "required": ["clarification_question", "clarification_field"],
}

_SYSTEM = (
    "You scan a personal memory for anything genuinely unclear enough to "
    "ask the user about -- nothing else. Do NOT summarize, categorize, or "
    "extract entities here; only decide what, if anything, is worth "
    "asking.\n\n"
    "Only ask about something INVENTED, PERSONAL, or genuinely AMBIGUOUS "
    "that a knowledgeable person could not resolve from general "
    "knowledge -- a made-up term, an unexplained personal nickname, an "
    "abbreviation with no obvious real-world meaning, or a "
    "pronoun/reference whose identity materially affects how this memory "
    "would later be searched for or understood (e.g. \"I talked to him "
    "about the project\" -- who 'him' is matters). Do NOT ask about a "
    "well-known real-world name, abbreviation, or institution just "
    "because you're inferring it from context. A missing field is not, "
    "by itself, grounds for a question -- store a sufficiently useful "
    "memory, not a perfectly complete record: e.g. \"Anna recommended a "
    "paper about tactile sensing\" needs NO question about the paper's "
    "title or author, even though neither is given -- that is an "
    "optional, not materially ambiguous, detail. Leave "
    "clarification_question as \"\" whenever nothing qualifies; most "
    "memories need no question at all. If attached file content or "
    "earlier related memories are "
    "given below, treat them as already-known context, not something to "
    "ask about -- a vague reference they already resolve is not "
    "ambiguous. If a field already has a declined/unknown answer noted "
    "below, do not ask about it again."
)


async def generate_question(
    llm: LLMProvider,
    content: str,
    clarifications: list[dict],
    attachment_text: str,
    prior_context: list[str] | None = None,
) -> dict | None:
    """Returns `{"question": str, "field": str}`, or `None` if nothing
    needs asking. Never raises: an LLM failure here just means no
    question is asked this round (enrichment still runs best-effort,
    same tolerance as the rest of the ingestion pipeline).

    `attachment_text` is a required positional parameter, not an
    afterthought kwarg -- every caller must pass whatever's already
    known about attached files (including an uploaded image's caption
    and any structured data extracted from it), so a vague reference the
    attachment already resolves is never asked about. The guarantee
    this relies on is call-order, enforced in code, not prompt wording:
    `process_capture_fully()` always finishes image captioning/
    structured extraction before this is ever called.
    """
    context = context_blocks(attachment_text, prior_context or [])
    clar_block = format_clarifications(clarifications)
    prompt = f"{context}{clar_block}Memory text:\n{content}\n\nWhat, if anything, is worth asking about?"

    try:
        raw = await llm.generate_structured(prompt, schema=_SCHEMA, system=_SYSTEM)
    except LLMError:
        return None

    question = raw.get("clarification_question")
    if not isinstance(question, str) or not question.strip():
        return None
    field = raw.get("clarification_field")
    field = field.strip() if isinstance(field, str) and field.strip() else "clarification"
    return {"question": question.strip(), "field": field}
