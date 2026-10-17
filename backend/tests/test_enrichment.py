from app.ingestion.enrichment import enrich_unit
from app.llm.base import LLMProvider
from tests.fakes import FakeLLMProvider

# Multi-prompt semantic enrichment, restored as three independent
# sub-calls (narrative/entities/retrieval-aids -- see
# docs/DECISIONS.md "Restore the multi-call ingestion pipeline").
# Deliberately does not decide clarification questions any more -- see
# tests/test_clarify.py for that.


async def test_degrades_to_unavailable_on_total_llm_failure():
    llm = FakeLLMProvider(fail="unavailable")
    semantic = await enrich_unit(llm, "Some memory text.", [], "")
    assert semantic["enrichment_status"] == "unavailable"
    assert semantic["meaning"] == ""


async def test_filters_malformed_definitions_and_relations():
    llm = FakeLLMProvider(
        enrichment_response={
            "definitions": [
                {"term": "real term", "definition": "real definition"},
                {"term": "", "definition": "orphaned definition"},
                {"term": 123, "definition": "non-string term"},
            ],
            "relations": [
                {"subject": "I", "predicate": "met", "object": "Anna"},
                {"subject": "", "predicate": "met", "object": "nobody"},
            ],
        }
    )
    semantic = await enrich_unit(llm, "I met Anna.", [], "")
    assert semantic["definitions"] == [{"term": "real term", "definition": "real definition"}]
    assert semantic["relations"] == [{"subject": "I", "predicate": "met", "object": "Anna"}]


async def test_literal_null_string_from_model_is_treated_as_empty():
    """Observed for real against the production model: for an unset
    field it sometimes emits the literal string "null" instead of ""
    -- a structured-output quirk, not an actual value."""
    llm = FakeLLMProvider(
        enrichment_response={
            "event_date": "null",
            "meaning": "NULL",
            "aliases": ["null", "real alias"],
            "topics": ["none"],
        }
    )
    semantic = await enrich_unit(llm, "I met Anna.", [], "")
    assert semantic["event_date"] == ""
    assert semantic["meaning"] == ""
    assert semantic["aliases"] == ["real alias"]
    assert semantic["topics"] == []


async def test_structured_data_instruction_present_for_image_attachments():
    """The one prompt line that fixes the glasses-prescription bug: the
    retrieval-aids sub-call must be told to promote a "Structured data:"
    block's label/value pairs into `relations`, not just summarize them.
    """
    from app.ingestion.enrichment import _RETRIEVAL_AIDS_SYSTEM

    assert "Structured data:" in _RETRIEVAL_AIDS_SYSTEM
    assert "relations" in _RETRIEVAL_AIDS_SYSTEM


class _PromptCapturingLLM(LLMProvider):
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


async def test_includes_attachment_text_in_every_sub_call_prompt():
    llm = _PromptCapturingLLM({"meaning": "x"})
    await enrich_unit(llm, "I like shoes of this colour.", [], "a photograph of a red running shoe")
    assert len(llm.prompts) == 3
    assert all("a photograph of a red running shoe" in p for p in llm.prompts)


async def test_includes_prior_context_in_prompt():
    llm = _PromptCapturingLLM({"meaning": "x"})
    await enrich_unit(
        llm, "Made progress on Foobar today.", [], "",
        prior_context=["Project Foobar refers to the new API migration."],
    )
    assert all("Project Foobar refers to the new API migration." in p for p in llm.prompts)


async def test_includes_clarification_history_in_prompt():
    llm = _PromptCapturingLLM({"meaning": "x"})
    await enrich_unit(
        llm, "I talked to him about the project.",
        [{"question": "Who is 'him'?", "answer": "Markus", "outcome": "answered"}],
        "",
    )
    assert all("Markus" in p and "Who is 'him'?" in p for p in llm.prompts)
