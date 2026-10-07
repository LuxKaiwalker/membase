import logging

from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "cues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["certain", "likely", "uncertain"]},
                    "polarity": {"type": "string", "enum": ["positive", "negative"]},
                },
                "required": ["text", "confidence", "polarity"],
            },
        },
    },
    "required": ["cues"],
}

_SYSTEM = (
    "You are extracting retrieval cues from what a user just said while trying "
    "to recall something they stored earlier. You are NOT answering their "
    "question and NOT recording a new permanent memory -- only identifying "
    "short, standalone search cues (a name, place, topic, or approximate "
    "time) that could help search their notes.\n\n"
    "First, extract EVERY concrete cue the user's own words state, even if "
    "they also answered yes/no to the system's question -- a message can "
    "both confirm/deny the question AND add new information, and the new "
    "information matters more. For example if the system asked 'Did this "
    "involve Markus?' and the user replies 'Yes, it was at ETH', extract "
    "BOTH a cue for 'Markus' (confirmed) AND a separate cue for 'ETH' (the "
    "new detail) -- do not drop the new detail just because the message "
    "also contained a yes/no answer.\n\n"
    "Only when the message is a BARE confirmation/denial with no new "
    "information at all ('yes', 'no', 'exactly', 'correct') should you fall "
    "back to inferring a single cue from the question's own subject: "
    "confirming means a positive cue for whatever the question asked about, "
    "denying means a negative cue for it.\n\n"
    "For each cue set confidence: 'certain' for definite statements ('it was "
    "at ETH'), 'likely' for soft guesses ('I think it was October'), "
    "'uncertain' for weak guesses ('maybe Anna'). Set polarity 'negative' if "
    "the user is ruling something out (e.g. 'it wasn't Anna, it was Markus' "
    "-> one cue {text: Anna, polarity: negative} and one cue {text: Markus, "
    "polarity: positive, confidence: likely}), otherwise 'positive'.\n\n"
    "If the message truly contains no extractable cue, return an empty "
    "list -- do not invent one."
)


async def extract_cues(
    llm: LLMProvider, message: str, *, pending_question: str | None = None
) -> list[dict]:
    """Turn a free-text recall-session message into structured search cues
    (spec section 6a/15/16). This never writes anything durable -- the
    result only ever feeds search queries and session-local bookkeeping
    in app/retrieval/session.py, never Capture/MemoryUnit content (spec
    section 29).
    """
    message = message.strip()
    if not message:
        return []

    question_block = f"\nThe system had just asked: {pending_question}" if pending_question else ""
    prompt = f"User's message: {message}{question_block}"
    try:
        result = await llm.generate_structured(prompt, schema=_SCHEMA, system=_SYSTEM)
    except LLMError:
        logger.warning("Cue extraction failed; using the raw message as a single uncertain cue")
        return [{"text": message, "confidence": "uncertain", "polarity": "positive"}]

    raw_cues = result.get("cues")
    if not isinstance(raw_cues, list):
        return []

    cleaned: list[dict] = []
    for item in raw_cues:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        confidence = item.get("confidence")
        if confidence not in ("certain", "likely", "uncertain"):
            confidence = "uncertain"
        polarity = item.get("polarity")
        if polarity not in ("positive", "negative"):
            polarity = "positive"
        cleaned.append({"text": text.strip(), "confidence": confidence, "polarity": polarity})
    return cleaned
