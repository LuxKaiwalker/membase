
from app.ingestion.pipeline import MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS, create_capture
from app.llm.base import LLMProvider
from tests.helpers import resolve_clarification_and_finish

# A generation call's context window is shared between its input prompt
# and its output -- an uncapped attachment_text (a large PDF, or several
# attachments) risks the same output-truncation failure fixed for
# many-fact captures, just from the input side (see docs/DECISIONS.md
# "Chunk long captures before analysis" and the attachment-text cap
# added alongside it). These tests verify the cap is actually applied
# at every LLM-prompt call site, not just the full-text search index.


class _PromptCapturingLLM(LLMProvider):
    """Records every prompt and returns a fixed response -- used here to
    verify how much attachment text actually reaches a generation
    prompt, since FakeLLMProvider dispatches purely by schema shape."""

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


async def test_create_capture_caps_attachment_text_in_analysis_prompt(db):
    huge_text = "X" * 5000
    llm = _PromptCapturingLLM({"units": [{"content": "A short memory."}]})

    await create_capture(db, llm, text="A short memory.", uploads=[("notes.txt", huge_text.encode())])

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    # The capped excerpt is present...
    assert "X" * MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS in prompt
    # ...but the full 5000-char attachment never reached the prompt.
    assert "X" * (MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS + 1) not in prompt


async def test_resolve_clarification_caps_attachment_text_in_reanalysis_prompt(db):
    huge_text = "Y" * 5000
    llm = _PromptCapturingLLM(
        {"units": [{"content": "A short memory.", "clarification_question": "What do you mean?", "clarification_field": "x"}]}
    )
    capture = await create_capture(db, llm, text="A short memory.", uploads=[("notes.txt", huge_text.encode())])
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    llm.prompts.clear()
    llm._response = {"meaning": "It means this."}
    await resolve_clarification_and_finish(db, llm, unit, "It means this.")

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "Y" * MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS in prompt
    assert "Y" * (MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS + 1) not in prompt
