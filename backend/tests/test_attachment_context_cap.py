
from app.ingestion.pipeline import MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS
from app.llm.base import LLMProvider
from tests.helpers import create_capture_and_finish, resolve_clarification_and_finish

# A generation call's context window is shared between its input prompt
# and its output -- an uncapped attachment_text (a large PDF, or several
# attachments) risks the same output-truncation failure fixed for
# many-fact captures, just from the input side (see docs/DECISIONS.md
# "Chunk long captures before analysis" and the attachment-text cap
# added alongside it). These tests verify the cap is actually applied at
# every LLM-prompt call site that receives attachment_text (clarify's
# call and each of enrichment's three sub-calls -- segmentation itself
# never sees attachment_text at all, see app/ingestion/segmentation.py).


class _PromptCapturingLLM(LLMProvider):
    """Records every prompt and returns a fixed response (which must
    satisfy every schema this gets asked for -- segmentation, clarify,
    and all three enrichment sub-calls -- since this doesn't dispatch by
    schema shape the way FakeLLMProvider does) -- used here to verify
    how much attachment text actually reaches a generation prompt."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.prompts: list[str] = []

    async def generate(self, prompt, *, system=None, temperature=0.2, images=None):
        self.prompts.append(prompt)
        return "ok"

    async def generate_structured(self, prompt, *, schema, system=None, images=None):
        self.prompts.append(prompt)
        return self._response

    async def embed(self, texts):
        return [[0.0] * 768 for _ in texts]

    @property
    def embedding_model_name(self) -> str:
        return "fake"

    @property
    def embedding_dim(self) -> int:
        return 768


_NO_QUESTION_ALL_FIELDS = {
    "units": [{"content": "A short memory."}],
    "clarification_question": "", "clarification_field": "",
    "meaning": "", "event_date": "", "people": [], "organizations": [], "locations": [],
    "aliases": [], "topics": [], "definitions": [], "relations": [],
}


async def test_create_capture_caps_attachment_text_in_analysis_prompt(db):
    huge_text = "X" * 5000
    llm = _PromptCapturingLLM(dict(_NO_QUESTION_ALL_FIELDS))

    await create_capture_and_finish(db, llm, text="A short memory.", uploads=[("notes.txt", huge_text.encode())])

    capped = "X" * MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS
    overcapped = "X" * (MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS + 1)
    # The capped excerpt reaches at least one prompt (clarify's, and
    # each enrichment sub-call's)...
    assert any(capped in p for p in llm.prompts)
    # ...but the full 5000-char attachment never reaches ANY prompt.
    assert all(overcapped not in p for p in llm.prompts)


async def test_resolve_clarification_caps_attachment_text_in_reanalysis_prompt(db):
    huge_text = "Y" * 5000
    llm = _PromptCapturingLLM({
        **_NO_QUESTION_ALL_FIELDS,
        "clarification_question": "What do you mean?", "clarification_field": "x",
    })
    capture = await create_capture_and_finish(db, llm, text="A short memory.", uploads=[("notes.txt", huge_text.encode())])
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    llm.prompts.clear()
    llm._response = {**_NO_QUESTION_ALL_FIELDS, "meaning": "It means this."}
    await resolve_clarification_and_finish(db, llm, unit, "It means this.")

    capped = "Y" * MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS
    overcapped = "Y" * (MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS + 1)
    assert any(capped in p for p in llm.prompts)
    assert all(overcapped not in p for p in llm.prompts)
