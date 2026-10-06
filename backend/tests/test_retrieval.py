import pytest

from app.ingestion.pipeline import create_capture
from app.retrieval.pipeline import hybrid_search, sibling_context


@pytest.mark.asyncio
async def test_exact_query_finds_unit(db, fake_llm):
    cap = await create_capture(db, fake_llm, text="The quick brown fox jumps over the lazy dog", uploads=[])
    await create_capture(db, fake_llm, text="Grocery list: milk, eggs, bread", uploads=[])
    await create_capture(db, fake_llm, text="Notes about the weather this week", uploads=[])

    scored = await hybrid_search(db, fake_llm, "quick brown fox")
    assert scored, "expected at least one candidate"
    assert scored[0].unit.capture_id == cap.id


@pytest.mark.asyncio
async def test_irrelevant_units_do_not_dominate(db, fake_llm):
    # FakeLLMProvider's embeddings are content-independent hashes, so the
    # vector channel contributes only noise here -- this specifically
    # checks that noise doesn't bury the one unit full-text search
    # correctly identifies. Real embedding quality is covered by
    # tests/test_integration_real_llm.py.
    cap = await create_capture(db, fake_llm, text="Paris trip notes about the Eiffel Tower", uploads=[])
    for i in range(5):
        await create_capture(db, fake_llm, text=f"distractor memory number {i} about nothing related", uploads=[])

    scored = await hybrid_search(db, fake_llm, "Eiffel Tower")
    ids = [s.unit.capture_id for s in scored]
    assert cap.id in ids
    assert ids[0] == cap.id


@pytest.mark.asyncio
async def test_no_match_returns_empty(db, fake_llm):
    await create_capture(db, fake_llm, text="Something about gardening", uploads=[])
    scored = await hybrid_search(db, fake_llm, "xyzzy nonexistent plugh quux")
    assert scored == []


@pytest.mark.asyncio
async def test_sibling_context_expands_within_same_capture(db):
    from tests.fakes import FakeLLMProvider

    llm = FakeLLMProvider(
        segmentation_response={
            "units": [
                {"content": "Anna studies robotics.", "reason": "x", "independence": "independent"},
                {"content": "Anna recommended a tactile sensing paper.", "reason": "x", "independence": "independent"},
            ]
        }
    )
    cap = await create_capture(
        db, llm, text="Anna studies robotics. Anna recommended a tactile sensing paper.", uploads=[]
    )
    assert len(cap.units) == 2

    scored = await hybrid_search(db, llm, "tactile sensing paper")
    assert scored
    primary_ids = {s.unit.id for s in scored}
    context = sibling_context(db, scored)
    context_ids = {u.id for u in context}
    # Siblings exclude whatever's already primary, and come from the same capture.
    assert context_ids.isdisjoint(primary_ids)
    assert all(u.capture_id == cap.id for u in context)
