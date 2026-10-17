from app.ingestion.clarify import generate_question
from app.llm.base import LLMProvider
from tests.fakes import FakeLLMProvider

# Dedicated clarification-question call, restored as its own focused
# step separate from enrichment (see docs/DECISIONS.md "Restore the
# multi-call ingestion pipeline") -- a dedicated call, with no other
# task competing for its attention, is what the pre-collapse pipeline
# used to catch an invented term the unified call missed.


async def test_no_question_by_default(fake_llm):
    result = await generate_question(fake_llm, "I had lunch today.", [], "")
    assert result is None


async def test_configured_question_is_returned():
    llm = FakeLLMProvider(
        clarification_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"}
    )
    result = await generate_question(llm, "My Flimvex PIN is 42.", [], "")
    assert result == {"question": "What is Flimvex?", "field": "Flimvex"}


async def test_blank_field_defaults_to_generic_label():
    llm = FakeLLMProvider(clarification_response={"clarification_question": "Who is 'him'?", "clarification_field": ""})
    result = await generate_question(llm, "I talked to him.", [], "")
    assert result == {"question": "Who is 'him'?", "field": "clarification"}


async def test_degrades_to_none_on_llm_failure():
    llm = FakeLLMProvider(fail="unavailable")
    result = await generate_question(llm, "Some memory text.", [], "")
    assert result is None


class _PromptCapturingLLM(LLMProvider):
    """Records every prompt it's asked to complete -- the only reliable
    way to verify attachment/prior-context/clarification-history text
    actually reaches a prompt, since FakeLLMProvider dispatches purely
    by schema shape and never looks at prompt content."""

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
        raise NotImplementedError

    @property
    def embedding_model_name(self) -> str:
        return "fake"

    @property
    def embedding_dim(self) -> int:
        return 1


async def test_includes_attachment_text_in_prompt():
    llm = _PromptCapturingLLM({"clarification_question": "", "clarification_field": ""})
    await generate_question(llm, "I like shoes of this colour.", [], "a photograph of a red running shoe")
    assert "a photograph of a red running shoe" in llm.prompts[0]


async def test_includes_prior_context_in_prompt():
    llm = _PromptCapturingLLM({"clarification_question": "", "clarification_field": ""})
    await generate_question(
        llm, "Made progress on Foobar today.", [], "",
        prior_context=["Project Foobar refers to the new API migration."],
    )
    assert "Project Foobar refers to the new API migration." in llm.prompts[0]


async def test_includes_clarification_history_in_prompt():
    llm = _PromptCapturingLLM({"clarification_question": "", "clarification_field": ""})
    await generate_question(
        llm, "I talked to him about the project.", [], "",
        prior_context=None,
    )
    await generate_question(
        llm, "I talked to him about the project.",
        [{"question": "Who is 'him'?", "answer": "Markus", "outcome": "answered"}],
        "",
    )
    assert "Markus" in llm.prompts[1]
    assert "Who is 'him'?" in llm.prompts[1]
