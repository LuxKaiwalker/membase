import pytest

from app.ingestion.enrichment import build_retrieval_text, enrich
from app.llm.base import LLMError, LLMProvider
from tests.fakes import FakeLLMProvider

# enrich()/build_retrieval_text() had no dedicated unit tests at all --
# only ever exercised indirectly through pipeline tests with well-formed
# fixtures. Added while auditing the multi-prompt split's degradation
# behavior and a data-quality gap in malformed-entry filtering.


class _SelectiveFailureLLM(LLMProvider):
    """Fails generate_structured only for schemas matching `fail_schema_key`
    (a property name distinguishing which of enrich()'s three sub-calls is
    being made), succeeding for the rest with `response`. FakeLLMProvider's
    `fail` applies uniformly to every call, so it can't simulate "only the
    entities call failed, narrative and retrieval-aids succeeded" -- the
    exact scenario this module's partial-degradation contract needs
    covering.
    """

    def __init__(self, fail_schema_key: str, response: dict) -> None:
        self._fail_schema_key = fail_schema_key
        self._response = response

    async def generate(self, prompt, *, system=None, temperature=0.2):
        raise NotImplementedError

    async def generate_structured(self, prompt, *, schema, system=None):
        if self._fail_schema_key in schema["properties"]:
            raise LLMError("simulated failure for this sub-call")
        return self._response

    async def embed(self, texts):
        raise NotImplementedError

    @property
    def embedding_model_name(self) -> str:
        return "fake"

    @property
    def embedding_dim(self) -> int:
        return 1


@pytest.mark.asyncio
async def test_enrich_degrades_to_unavailable_only_when_every_sub_call_fails():
    llm = FakeLLMProvider(fail="unavailable")
    result = await enrich(llm, "Some memory text.", [], "")
    assert result["enrichment_status"] == "unavailable"
    assert result["meaning"] == ""
    assert result["people"] == []


@pytest.mark.asyncio
async def test_enrich_partial_failure_keeps_successful_slices():
    """If only one of the three sub-calls fails (e.g. the entities call
    times out), the slices that did succeed must not be discarded --
    enrich() is documented as best-effort per slice, not all-or-nothing."""
    # "people" only appears in the entities schema's properties, so this
    # fails exactly that sub-call; narrative and retrieval-aids schemas
    # don't contain "people" and succeed with this same response dict
    # (each only reads the fields relevant to it).
    llm = _SelectiveFailureLLM(
        fail_schema_key="people",
        response={"meaning": "A thing happened.", "event_date": "", "aliases": ["a thing"], "topics": [], "definitions": [], "relations": []},
    )
    result = await enrich(llm, "Some memory text.", [], "")
    assert result["enrichment_status"] == "ok"
    assert result["meaning"] == "A thing happened."
    assert result["aliases"] == ["a thing"]
    # The failed slice's fields are empty, not fabricated.
    assert result["people"] == []
    assert result["organizations"] == []


@pytest.mark.asyncio
async def test_enrich_filters_malformed_definitions_and_relations():
    llm = FakeLLMProvider(
        enrichment_response={
            "meaning": "x", "event_date": "", "people": [], "organizations": [], "locations": [],
            "aliases": [], "topics": [],
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
    result = await enrich(llm, "I met Anna.", [], "")
    assert result["definitions"] == [{"term": "real term", "definition": "real definition"}]
    assert result["relations"] == [{"subject": "I", "predicate": "met", "object": "Anna"}]


class _PromptCapturingLLM(LLMProvider):
    """Records every prompt it's asked to complete, so a test can assert
    on the exact text sent to the model -- the only reliable way to
    verify attachment context actually reaches a prompt, since
    FakeLLMProvider dispatches purely by schema shape and never looks at
    prompt content."""

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


@pytest.mark.asyncio
async def test_enrich_includes_attachment_text_in_every_sub_call_prompt():
    """An attached image's caption (or any attachment's extracted text)
    is real content about the memory -- meaning/entities/aliases
    extraction must see it, not just full-text search (see
    docs/DECISIONS.md "Gemma 4 migration" / attachment-context fix)."""
    llm = _PromptCapturingLLM(
        {"meaning": "x", "event_date": "", "people": [], "organizations": [], "locations": [],
         "aliases": [], "topics": [], "definitions": [], "relations": []}
    )
    await enrich(llm, "I like shoes of this colour.", [], "a photograph of a red running shoe")
    assert len(llm.prompts) == 3, "all three sub-calls (narrative/entities/retrieval-aids) must run"
    for prompt in llm.prompts:
        assert "a photograph of a red running shoe" in prompt


def test_build_retrieval_text_includes_all_present_fields():
    semantic = {
        "meaning": "A trip to Japan.",
        "aliases": ["Japan trip"],
        "people": ["Anna"],
        "organizations": [],
        "locations": ["Osaka"],
        "topics": ["travel"],
        "definitions": [{"term": "AHV", "definition": "Swiss social security number"}],
        "relations": [{"subject": "I", "predicate": "visited", "object": "Osaka"}],
        "event_date": "2026-01-01",
    }
    text = build_retrieval_text("My trip to Japan.", [], semantic)
    assert "My trip to Japan." in text
    assert "A trip to Japan." in text
    assert "Japan trip" in text
    assert "AHV = Swiss social security number" in text
    assert "I visited Osaka" in text
    assert "People: Anna" in text
    assert "Locations: Osaka" in text
    assert "Event date: 2026-01-01" in text


def test_build_retrieval_text_includes_clarifications():
    semantic = {"meaning": "", "aliases": [], "people": [], "organizations": [], "locations": [], "topics": [], "definitions": [], "relations": [], "event_date": ""}
    clarifications = [{"question": "What is X?", "answer": "X is Y."}]
    text = build_retrieval_text("Some content.", clarifications, semantic)
    assert "Q: What is X?" in text
    assert "A: X is Y." in text
