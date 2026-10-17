from app.ingestion.segmentation import MAX_SEGMENT_CHARS_PER_CALL, MAX_SEGMENT_UNITS, segment_capture
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish

# Dedicated segmentation call, restored as its own focused step (see
# docs/DECISIONS.md "Restore the multi-call ingestion pipeline") --
# produces no semantic fields at all any more, just the list of
# verbatim unit contents. Covers the same safety properties the old
# collapsed analyze_capture() used to (verbatim-excerpt enforcement,
# graceful degradation, chunking of long captures).


async def test_no_split_by_default(fake_llm):
    text = "An atomic fact about one thing."
    units = await segment_capture(fake_llm, text)
    assert units == [text]


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
    units = await segment_capture(llm, text)
    assert units == ["Fact A.", "Fact B.", "Fact C."]


async def test_non_verbatim_unit_falls_back_to_sentence_split():
    """A non-verbatim proposal degrades to a deterministic *sentence*
    split, not one giant blob covering the whole text -- collapsing the
    entire capture to one unit on any call failure was itself the bug
    that let a single rejected proposal destroy all structure."""
    text = "I met Anna at ETH. She studies robotics."
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "I met Anna at ETH."},
                {"content": "She is a robotics student."},  # paraphrased, not verbatim
            ]
        }
    )
    units = await segment_capture(llm, text)
    assert units == ["I met Anna at ETH.", "She studies robotics."]


def test_word_boundary_violating_candidate_rejected_as_non_verbatim():
    from app.ingestion.analysis import is_verbatim

    assert not is_verbatim("y dog ate foo", "my dog ate food happily")


async def test_word_boundary_violating_unit_rejected_end_to_end():
    text = "my dog ate food happily"
    llm = FakeLLMProvider(segmentation_response={"units": [{"content": "y dog ate foo"}]})
    units = await segment_capture(llm, text)
    assert units == [text]


async def test_segmentation_unavailable_falls_back_to_whole_text():
    text = "Some memory text."
    llm = FakeLLMProvider(fail="unavailable")
    units = await segment_capture(llm, text)
    assert units == [text]


async def test_too_many_units_capped_at_max():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    text = " ".join(words)
    llm = FakeLLMProvider(segmentation_response={"units": [{"content": w} for w in words]})
    units = await segment_capture(llm, text)
    assert len(units) <= MAX_SEGMENT_UNITS


async def test_coherent_compound_memory_stays_one_unit_end_to_end(db):
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [{"content": "I met Anna at ETH after the lecture and she recommended a paper."}]
        }
    )
    capture = await create_capture_and_finish(
        db, llm, text="I met Anna at ETH after the lecture and she recommended a paper."
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
    capture = await create_capture_and_finish(db, llm, text=text)
    assert len(capture.units) == 3
    assert capture.units[0].authoritative_content != capture.units[1].authoritative_content


# --- Chunking long/dense captures: a capture with many independent
# facts can make one structured-output call exceed the model's output
# budget, which previously degraded the ENTIRE capture to one unopposed
# unit. Reproduces that failure mode using the fake model's
# deterministic behavior.

_MANY_SENTENCES = " ".join(f"Fact number {i} is independently true and stands on its own." for i in range(20))


async def test_short_capture_is_still_exactly_one_call(fake_llm):
    """Ordinary short captures must be completely unaffected by
    chunking -- same single call as before."""
    text = "Fact one. Fact two. Fact three."
    assert len(text) <= MAX_SEGMENT_CHARS_PER_CALL
    await segment_capture(fake_llm, text)
    assert fake_llm.calls.count("segment") == 1


async def test_long_capture_is_split_into_multiple_calls(fake_llm):
    assert len(_MANY_SENTENCES) > MAX_SEGMENT_CHARS_PER_CALL
    units = await segment_capture(fake_llm, _MANY_SENTENCES)
    assert fake_llm.calls.count("segment") > 1
    # Every chunk's content is an exact, lossless substring of the
    # original -- concatenating them (modulo whitespace) reproduces it.
    assert " ".join(units) == _MANY_SENTENCES
    assert len(units) <= MAX_SEGMENT_UNITS


async def test_total_llm_outage_on_a_long_capture_degrades_per_sentence_not_to_one_blob():
    """The bug this fixes: a long capture whose segmentation call(s)
    fail entirely must NOT collapse into one giant unit -- each sentence
    becomes its own unit instead, so retrieval over individual facts
    still works."""
    llm = FakeLLMProvider(fail="unavailable")
    units = await segment_capture(llm, _MANY_SENTENCES)
    assert len(units) > 1
    for u in units:
        assert u in _MANY_SENTENCES


async def test_malformed_output_on_a_long_capture_degrades_per_sentence_not_to_one_blob():
    """Same guarantee as above, for the specific failure mode observed
    live: the model's JSON came back truncated/invalid (not a
    connectivity error)."""
    llm = FakeLLMProvider(fail="malformed")
    units = await segment_capture(llm, _MANY_SENTENCES)
    assert len(units) > 1
    for u in units:
        assert u in _MANY_SENTENCES


async def test_a_single_chunks_failure_does_not_affect_other_chunks(fake_llm):
    """Chunking also bounds the blast radius of a failure: only the
    chunk whose own call failed degrades (per-sentence), the rest keep
    their real (fake-model-provided) segmentation."""
    calls = {"n": 0}

    class _FlakyLLM(FakeLLMProvider):
        async def generate_structured(self, prompt, *, schema, system=None, images=None):
            if "units" in schema.get("properties", {}):
                calls["n"] += 1
                if calls["n"] == 2:
                    from app.llm.base import LLMUnavailableError

                    raise LLMUnavailableError("simulated failure on the second chunk only")
            return await super().generate_structured(prompt, schema=schema, system=system, images=images)

    llm = _FlakyLLM()
    units = await segment_capture(llm, _MANY_SENTENCES)
    assert len(units) >= 1
    for u in units:
        assert u in _MANY_SENTENCES
