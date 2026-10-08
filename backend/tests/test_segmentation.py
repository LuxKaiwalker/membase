import pytest

from app.ingestion.pipeline import create_capture
from app.ingestion.segmentation import MAX_SEGMENT_UNITS, segment_capture
from tests.fakes import FakeLLMProvider


@pytest.mark.asyncio
async def test_no_split_by_default(fake_llm):
    text = "An atomic fact about one thing."
    units = await segment_capture(fake_llm, text)
    assert len(units) == 1
    assert units[0].content == text


@pytest.mark.asyncio
async def test_model_proposed_split_is_honored_when_verbatim():
    text = "Fact A. Fact B. Fact C."
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "Fact A.", "reason": "unrelated", "independence": "independent"},
                {"content": "Fact B.", "reason": "unrelated", "independence": "independent"},
                {"content": "Fact C.", "reason": "unrelated", "independence": "independent"},
            ]
        }
    )
    units = await segment_capture(llm, text)
    assert [u.content for u in units] == ["Fact A.", "Fact B.", "Fact C."]


@pytest.mark.asyncio
async def test_non_verbatim_unit_falls_back_to_no_split():
    text = "I met Anna at ETH. She studies robotics."
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "I met Anna at ETH.", "reason": "x", "independence": "contextual"},
                {"content": "She is a robotics student.", "reason": "paraphrased, not verbatim", "independence": "independent"},
            ]
        }
    )
    units = await segment_capture(llm, text)
    assert len(units) == 1
    assert units[0].content == text


@pytest.mark.asyncio
async def test_word_boundary_violating_unit_rejected_as_non_verbatim():
    """A candidate that only matches by crossing word boundaries (e.g.
    chopped mid-word) is a character-substring of the source but not an
    actual excerpt of it -- the verbatim check must reject it, not just
    check for "in" at the character level."""
    text = "my dog ate food happily"
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                # "y dog ate foo" is a substring of "my dog ate food" at
                # the character level, but chops both "my" and "food"
                # mid-word -- not a real excerpt.
                {"content": "y dog ate foo", "reason": "x", "independence": "independent"},
            ]
        }
    )
    units = await segment_capture(llm, text)
    assert len(units) == 1
    assert units[0].content == text


@pytest.mark.asyncio
async def test_segmentation_unavailable_falls_back_to_whole_text():
    text = "Some memory text."
    llm = FakeLLMProvider(fail="unavailable")
    units = await segment_capture(llm, text)
    assert len(units) == 1
    assert units[0].content == text


@pytest.mark.asyncio
async def test_too_many_units_capped_at_max():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    text = " ".join(words)
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [{"content": w, "reason": "x", "independence": "independent"} for w in words]
        }
    )
    units = await segment_capture(llm, text)
    assert len(units) <= MAX_SEGMENT_UNITS


@pytest.mark.asyncio
async def test_coherent_compound_memory_stays_one_unit_end_to_end(db):
    """A coherent episode (one event, one person, connected facts) should
    be allowed to remain a single memory unit end-to-end through
    create_capture, not just at the segmentation function level."""
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {
                    "content": "I met Anna at ETH after the lecture and she recommended a paper.",
                    "reason": "one coherent event",
                    "independence": "independent",
                }
            ]
        }
    )
    capture = await create_capture(
        db, llm, text="I met Anna at ETH after the lecture and she recommended a paper.", uploads=[]
    )
    assert len(capture.units) == 1


@pytest.mark.asyncio
async def test_multiple_unrelated_facts_become_multiple_units_end_to_end(db):
    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "My AHV number is 123456789.", "reason": "unrelated", "independence": "independent"},
                {"content": "Anna studies robotics.", "reason": "unrelated", "independence": "independent"},
                {"content": "I need to renew my passport next March.", "reason": "unrelated", "independence": "independent"},
            ]
        }
    )
    text = "My AHV number is 123456789. Anna studies robotics. I need to renew my passport next March."
    capture = await create_capture(db, llm, text=text, uploads=[])
    assert len(capture.units) == 3
    assert capture.units[0].authoritative_content != capture.units[1].authoritative_content
