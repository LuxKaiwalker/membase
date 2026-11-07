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


async def test_visual_concepts_extracted_from_an_image_caption():
    """The product's headline demo ("that photo from the summer, there was
    a lake and some mountains") depends on this field, which
    app/memory/graph.py already projects as "object" entities but which
    nothing used to write. It costs NO extra LLM call: the retrieval-aids
    sub-call's prompt already carries the image caption."""
    llm = FakeLLMProvider(enrichment_response={"visual_concepts": ["lake", "mountains", "snow"]})
    semantic = await enrich_unit(
        llm, "That hike last summer.", [],
        "Image 1:\nA wide alpine lake with snow-capped mountains behind it.",
    )
    assert semantic["visual_concepts"] == ["lake", "mountains", "snow"]


async def test_visual_concepts_empty_for_a_capture_with_no_image():
    """Most memories have no image at all -- the field must stay [] rather
    than being filled from the memory's prose, or every unit would start
    emitting spurious "object" entities into the association graph."""
    llm = FakeLLMProvider(enrichment_response={"topics": ["rent"]})
    semantic = await enrich_unit(llm, "My rent went up to 1400.", [], "")
    assert semantic["visual_concepts"] == []


async def test_visual_concepts_instruction_and_schema_present():
    """Both halves have to be there: the schema field alone gets an
    unguided list, and the instruction alone gets a field the model is
    not allowed to return."""
    from app.ingestion.enrichment import _RETRIEVAL_AIDS_SCHEMA, _RETRIEVAL_AIDS_SYSTEM

    assert "visual_concepts" in _RETRIEVAL_AIDS_SCHEMA["properties"]
    assert "visual_concepts" in _RETRIEVAL_AIDS_SCHEMA["required"]
    assert "visual_concepts" in _RETRIEVAL_AIDS_SYSTEM
    assert "VISIBLE" in _RETRIEVAL_AIDS_SYSTEM


async def test_visual_concepts_literal_null_strings_filtered():
    llm = FakeLLMProvider(enrichment_response={"visual_concepts": ["null", "lake", "none"]})
    semantic = await enrich_unit(llm, "A photo.", [], "Image 1:\nA lake.")
    assert semantic["visual_concepts"] == ["lake"]


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
