
from app.ingestion.analysis import (
    MAX_ANALYSIS_CHARS_PER_CALL,
    MAX_SEGMENT_UNITS,
    analyze_capture,
    build_retrieval_text,
    is_verbatim,
    reanalyze_unit,
)
from app.ingestion.pipeline import create_capture
from app.llm.base import LLMProvider
from tests.fakes import FakeLLMProvider

# Streamlining phase (see docs/DECISIONS.md "Collapse ingestion LLM
# calls"): segmentation, clarification-need, and semantic enrichment are
# now one call (analyze_capture) per capture, plus one call
# (reanalyze_unit) per clarification round. This file replaces the old
# test_segmentation.py/test_enrichment.py, covering the same safety
# properties (verbatim-excerpt enforcement, graceful degradation)
# against the new unified module.


async def test_no_split_by_default(fake_llm):
    text = "An atomic fact about one thing."
    units = await analyze_capture(fake_llm, text)
    assert len(units) == 1
    assert units[0].content == text


async def test_model_proposed_split_is_honored_when_verbatim():
    text = "Fact A. Fact B. Fact C."
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "Fact A."},
                {"content": "Fact B."},
                {"content": "Fact C."},
            ]
        }
    )
    units = await analyze_capture(llm, text)
    assert [u.content for u in units] == ["Fact A.", "Fact B.", "Fact C."]


async def test_non_verbatim_unit_falls_back_to_sentence_split():
    """A non-verbatim proposal degrades to a deterministic *sentence*
    split, not one giant blob covering the whole text -- see
    docs/DECISIONS.md "Chunk long captures before analysis": collapsing
    the entire capture to one unit on any call failure was itself the
    bug that let a single rejected proposal destroy all structure."""
    text = "I met Anna at ETH. She studies robotics."
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "I met Anna at ETH."},
                {"content": "She is a robotics student."},  # paraphrased, not verbatim
            ]
        }
    )
    units = await analyze_capture(llm, text)
    assert [u.content for u in units] == ["I met Anna at ETH.", "She studies robotics."]
    assert all(u.semantic["enrichment_status"] == "ok" for u in units)


def test_word_boundary_violating_candidate_rejected_as_non_verbatim():
    """A candidate that only matches by crossing word boundaries (e.g.
    chopped mid-word) is a character-substring of the source but not an
    actual excerpt of it -- is_verbatim() must reject it, not just check
    for "in" at the character level."""
    assert not is_verbatim("y dog ate foo", "my dog ate food happily")
    assert is_verbatim("dog ate food", "my dog ate food happily")


async def test_word_boundary_violating_unit_rejected_end_to_end():
    text = "my dog ate food happily"
    llm = FakeLLMProvider(segmentation_response={"units": [{"content": "y dog ate foo"}]})
    units = await analyze_capture(llm, text)
    assert len(units) == 1
    assert units[0].content == text


async def test_analysis_unavailable_falls_back_to_whole_text():
    text = "Some memory text."
    llm = FakeLLMProvider(fail="unavailable")
    units = await analyze_capture(llm, text)
    assert len(units) == 1
    assert units[0].content == text
    assert units[0].semantic["enrichment_status"] == "unavailable"


async def test_too_many_units_capped_at_max():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    text = " ".join(words)
    llm = FakeLLMProvider(segmentation_response={"units": [{"content": w} for w in words]})
    units = await analyze_capture(llm, text)
    assert len(units) <= MAX_SEGMENT_UNITS


async def test_coherent_compound_memory_stays_one_unit_end_to_end(db):
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [{"content": "I met Anna at ETH after the lecture and she recommended a paper."}]
        }
    )
    capture = await create_capture(
        db, llm, text="I met Anna at ETH after the lecture and she recommended a paper.", uploads=[]
    )
    assert len(capture.units) == 1


async def test_multiple_unrelated_facts_become_multiple_units_end_to_end(db):
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "My AHV number is 123456789."},
                {"content": "Anna studies robotics."},
                {"content": "I need to renew my passport next March."},
            ]
        }
    )
    text = "My AHV number is 123456789. Anna studies robotics. I need to renew my passport next March."
    capture = await create_capture(db, llm, text=text, uploads=[])
    assert len(capture.units) == 3
    assert capture.units[0].authoritative_content != capture.units[1].authoritative_content


async def test_analyze_capture_degrades_to_unavailable_on_llm_failure():
    llm = FakeLLMProvider(fail="unavailable")
    units = await analyze_capture(llm, "Some memory text.")
    assert units[0].semantic["enrichment_status"] == "unavailable"
    assert units[0].semantic["meaning"] == ""


async def test_analyze_capture_filters_malformed_definitions_and_relations():
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
    units = await analyze_capture(llm, "I met Anna.")
    assert units[0].semantic["definitions"] == [{"term": "real term", "definition": "real definition"}]
    assert units[0].semantic["relations"] == [{"subject": "I", "predicate": "met", "object": "Anna"}]


async def test_literal_null_string_from_model_is_treated_as_empty():
    """Observed for real against the production model: for an unset
    field it sometimes emits the literal string "null" instead of ""
    -- a structured-output quirk, not an actual value (see
    docs/DECISIONS.md)."""
    llm = FakeLLMProvider(
        enrichment_response={
            "event_date": "null",
            "meaning": "NULL",
            "aliases": ["null", "real alias"],
            "topics": ["none"],
        }
    )
    units = await analyze_capture(llm, "I met Anna.")
    semantic = units[0].semantic
    assert semantic["event_date"] == ""
    assert semantic["meaning"] == ""
    assert semantic["aliases"] == ["real alias"]
    assert semantic["topics"] == []


class _PromptCapturingLLM(LLMProvider):
    """Records every prompt it's asked to complete -- the only reliable
    way to verify attachment/prior-context text actually reaches a
    prompt, since FakeLLMProvider dispatches purely by schema shape and
    never looks at prompt content."""

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


async def test_analyze_capture_includes_attachment_text_in_prompt():
    llm = _PromptCapturingLLM({"units": [{"content": "I like shoes of this colour."}]})
    await analyze_capture(llm, "I like shoes of this colour.", "a photograph of a red running shoe")
    assert len(llm.prompts) == 1
    assert "a photograph of a red running shoe" in llm.prompts[0]


async def test_analyze_capture_includes_prior_context_in_prompt():
    llm = _PromptCapturingLLM({"units": [{"content": "Made progress on Foobar today."}]})
    await analyze_capture(llm, "Made progress on Foobar today.", prior_context=["Project Foobar refers to the new API migration."])
    assert "Project Foobar refers to the new API migration." in llm.prompts[0]


async def test_reanalyze_unit_degrades_to_unavailable_on_llm_failure():
    llm = FakeLLMProvider(fail="unavailable")
    result = await reanalyze_unit(llm, "Some memory text.", [], "")
    assert result.semantic["enrichment_status"] == "unavailable"


async def test_reanalyze_unit_includes_clarification_history_in_prompt():
    llm = _PromptCapturingLLM({"meaning": "x"})
    await reanalyze_unit(
        llm, "I talked to him about the project.",
        [{"question": "Who is 'him'?", "answer": "Markus", "outcome": "answered"}],
        "",
    )
    assert "Markus" in llm.prompts[0]
    assert "Who is 'him'?" in llm.prompts[0]


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


# --- Chunking long/dense captures (see docs/DECISIONS.md "Chunk long
# captures before analysis"): a capture with many independent facts can
# make one structured-output call exceed the model's output budget
# (full per-unit semantics for many units at once), which previously
# degraded the ENTIRE capture to one unopposed, empty-semantics unit.
# Reproduces the exact failure mode found live against gemma4:12b with a
# ~20-fact capture, using the fake model's deterministic behavior
# instead so it's not dependent on a real model being reachable.

_MANY_SENTENCES = " ".join(f"Fact number {i} is independently true and stands on its own." for i in range(20))


async def test_short_capture_is_still_exactly_one_call(fake_llm):
    """Ordinary short captures must be completely unaffected by
    chunking -- same single call as before."""
    text = "Fact one. Fact two. Fact three."
    assert len(text) <= MAX_ANALYSIS_CHARS_PER_CALL
    await analyze_capture(fake_llm, text)
    assert fake_llm.calls.count("analyze_capture") == 1


async def test_long_capture_is_split_into_multiple_calls(fake_llm):
    assert len(_MANY_SENTENCES) > MAX_ANALYSIS_CHARS_PER_CALL
    units = await analyze_capture(fake_llm, _MANY_SENTENCES)
    assert fake_llm.calls.count("analyze_capture") > 1
    # Every chunk's content is an exact, lossless substring of the
    # original -- concatenating them (modulo whitespace) reproduces it.
    assert " ".join(u.content for u in units) == _MANY_SENTENCES
    assert len(units) <= MAX_SEGMENT_UNITS


async def test_total_llm_outage_on_a_long_capture_degrades_per_sentence_not_to_one_blob():
    """The bug this fixes: a long capture whose analysis call(s) fail
    entirely must NOT collapse into one giant empty-semantics unit --
    each sentence becomes its own (still unenriched) unit instead, so
    retrieval over individual facts still works."""
    llm = FakeLLMProvider(fail="unavailable")
    units = await analyze_capture(llm, _MANY_SENTENCES)
    assert len(units) > 1
    assert all(u.semantic["enrichment_status"] == "unavailable" for u in units)
    for u in units:
        assert u.content in _MANY_SENTENCES


async def test_malformed_output_on_a_long_capture_degrades_per_sentence_not_to_one_blob():
    """Same guarantee as above, for the specific failure mode observed
    live: the model's JSON came back truncated/invalid (not a
    connectivity error) while trying to fit many units' full semantics
    in one call."""
    llm = FakeLLMProvider(fail="malformed")
    units = await analyze_capture(llm, _MANY_SENTENCES)
    assert len(units) > 1
    for u in units:
        assert u.content in _MANY_SENTENCES


async def test_a_single_chunks_failure_does_not_affect_other_chunks(fake_llm):
    """Chunking also bounds the blast radius of a failure: only the
    chunk whose own call failed degrades, the rest keep their real
    (fake-model-provided) semantics."""
    calls = {"n": 0}

    class _FlakyLLM(FakeLLMProvider):
        async def generate_structured(self, prompt, *, schema, system=None, images=None):
            if "units" in schema.get("properties", {}):
                calls["n"] += 1
                if calls["n"] == 2:
                    from app.llm.base import LLMUnavailableError

                    raise LLMUnavailableError("simulated failure on the second chunk only")
            return await super().generate_structured(prompt, schema=schema, system=system, images=images)

    llm = _FlakyLLM(enrichment_response={"topics": ["test-topic"]})
    units = await analyze_capture(llm, _MANY_SENTENCES)

    statuses = {u.semantic["enrichment_status"] for u in units}
    assert "ok" in statuses
    assert "unavailable" in statuses
