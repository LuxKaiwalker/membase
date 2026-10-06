import pytest

from app.ingestion.pipeline import create_memory
from app.retrieval.pipeline import retrieve


@pytest.mark.asyncio
async def test_exact_query_finds_memory(db, fake_llm):
    target = await create_memory(
        db, fake_llm, caption="The quick brown fox jumps over the lazy dog", uploads=[]
    )
    await create_memory(db, fake_llm, caption="Grocery list: milk, eggs, bread", uploads=[])
    await create_memory(db, fake_llm, caption="Notes about the weather this week", uploads=[])

    scored = await retrieve(db, fake_llm, "quick brown fox")
    assert scored, "expected at least one candidate"
    assert scored[0].memory.id == target.id


@pytest.mark.asyncio
async def test_irrelevant_memories_do_not_dominate(db, fake_llm):
    # FakeLLMProvider's embeddings are content-independent hashes, so the
    # vector channel contributes only noise here -- this specifically
    # checks that noise doesn't bury the one memory full-text search
    # correctly identifies. Real embedding quality is covered by
    # tests/test_integration_real_llm.py.
    target = await create_memory(db, fake_llm, caption="Paris trip notes about the Eiffel Tower", uploads=[])
    for i in range(5):
        await create_memory(db, fake_llm, caption=f"distractor memory number {i} about nothing related", uploads=[])

    scored = await retrieve(db, fake_llm, "Eiffel Tower")
    ids = [s.memory.id for s in scored]
    assert target.id in ids
    assert ids[0] == target.id


@pytest.mark.asyncio
async def test_no_match_returns_empty(db, fake_llm):
    await create_memory(db, fake_llm, caption="Something about gardening", uploads=[])
    scored = await retrieve(db, fake_llm, "xyzzy nonexistent plugh quux")
    assert scored == []
