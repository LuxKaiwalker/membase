"""Fast, narrow clarification-question generation -- deliberately
separate from app/ingestion/enrichment.py's full semantic-derivation
call (see docs/DECISIONS.md).

The old design asked one combined LLM call to do everything: derive
meaning/aliases/people/..., *and* decide whether a clarifying question
was needed. That meant the user never saw a question before the whole
(larger, slower) enrichment schema finished generating -- and if a
question *was* needed, that enrichment work was wasted anyway, since
resolving the clarification re-runs enrichment from scratch with the
answer as context. This module exists to ask "does this need
clarification, and about what" on its own, with a schema containing
nothing else, so a unit reaches `awaiting_clarification` (and the user
sees a question) as fast as possible -- full enrichment only runs once
no more clarification is pending.
"""

from app.llm.base import LLMError, LLMProvider

_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "field": {"type": "string"},
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["question", "field", "relevance"],
            },
        },
    },
    "required": ["questions"],
}

_SYSTEM = (
    "You scan a personal memory for anything genuinely unclear enough to "
    "ask the user about -- nothing else. Do NOT summarize, categorize, or "
    "extract entities here; only decide what, if anything, is worth "
    "asking.\n\n"
    "Only ask about something INVENTED, PERSONAL, or genuinely AMBIGUOUS "
    "that a knowledgeable person could not resolve from general "
    "knowledge -- a made-up term, an unexplained personal nickname, an "
    "abbreviation with no obvious real-world meaning. Do NOT ask about a "
    "well-known real-world name, abbreviation, or institution (a "
    "university, a company, a country, a common technical term) just "
    "because you're inferring it from context rather than restating a "
    "dictionary definition -- if it plausibly refers to something real "
    "and the text's meaning is clear either way, that is not worth "
    "asking about. A wrong guessed meaning for an INVENTED term is worse "
    "than asking about it; asking about an ordinary real-world name you "
    "already understand just annoys the user. No prior clarification "
    "already resolved it (prior clarifications are given below the "
    "text). Rate each candidate question's relevance: 'high' for "
    "something central to understanding the memory, 'medium'/'low' for "
    "a minor or optional detail.\n\n"
    "Example where a question IS warranted (an invented/unfamiliar term, "
    'not a real known abbreviation):\nText: "My Zorblex code is 42."\n'
    '-> one question: {"question": "What is a Zorblex code?", "field": '
    '"Zorblex code", "relevance": "high"}\n\n'
    "Example where NO question is warranted (a real, well-known "
    'institution, even though not formally defined in the text):\nText: '
    '"I met Anna at ETH yesterday."\n-> questions: [] (ETH is a '
    "well-known university; nothing here is actually unclear)\n\n"
    "Example where NO question is warranted (ordinary text, nothing "
    'uncertain):\nText: "I had lunch today."\n-> questions: []\n\n'
    "If attached file content is given below the text, treat it as "
    "already-known information, not something to ask about -- a vague "
    "reference in the text that the attachment already answers is not "
    "ambiguous.\n\n"
    "Return an empty list whenever nothing qualifies -- most memories "
    "need no questions at all. Be terse."
)


def _format_clarifications(clarifications: list[dict]) -> str:
    if not clarifications:
        return ""
    lines = ["Already answered:"]
    for c in clarifications:
        lines.append(f"Q: {c.get('question', '')}\nA: {c.get('answer', '')}")
    return "\n".join(lines)


_RELEVANCE_ORDER = {"high": 0, "medium": 1, "low": 2}


async def generate_questions(
    llm: LLMProvider,
    content: str,
    clarifications: list[dict],
    attachment_text: str,
    *,
    max_questions: int,
) -> list[dict]:
    """Returns up to `max_questions` candidate clarification questions,
    each `{"question": str, "field": str}`, highest-relevance first.
    Never raises: an LLM failure here just means no question is asked
    this round (full enrichment still runs best-effort, same tolerance
    as the rest of the ingestion pipeline).

    `attachment_text` is a required positional parameter, not an
    afterthought kwarg -- every caller must pass whatever's already
    known about attached files (including an uploaded image's caption,
    see app/ingestion/pipeline.py's _describe_image), so a vague
    reference the attachment already resolves is never asked about.
    The guarantee this relies on is call-order, enforced in code, not
    prompt wording: pipeline.py always finishes image captioning before
    this is ever called, so `attachment_text` is complete by the time it
    gets here (see docs/DECISIONS.md).
    """
    clar_block = _format_clarifications(clarifications)
    attachment_block = f"\nAttached file content:\n{attachment_text}\n" if attachment_text else ""
    prompt = (
        f"Memory text:\n{content}\n{attachment_block}{clar_block}\n\n"
        "What, if anything, is worth asking about?"
    )

    try:
        raw = await llm.generate_structured(prompt, schema=_SCHEMA, system=_SYSTEM)
    except LLMError:
        return []

    items = raw.get("questions")
    if not isinstance(items, list):
        return []

    cleaned = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        field = item.get("field")
        field = field.strip() if isinstance(field, str) and field.strip() else "clarification"
        relevance = item.get("relevance") if item.get("relevance") in _RELEVANCE_ORDER else "low"
        cleaned.append({"question": question.strip(), "field": field, "relevance": relevance})

    cleaned.sort(key=lambda q: _RELEVANCE_ORDER[q["relevance"]])
    return [{"question": q["question"], "field": q["field"]} for q in cleaned[:max_questions]]
