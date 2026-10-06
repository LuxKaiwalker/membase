import pytest

from app.ingestion.pipeline import create_memory
from app.retrieval.pipeline import answer
from tests.fakes import FakeLLMProvider


@pytest.mark.asyncio
async def test_ingestion_survives_llm_unavailable(db):
    llm = FakeLLMProvider(fail="unavailable")
    memory = await create_memory(db, llm, caption="Stored even if the LLM is down", uploads=[])
    assert memory.caption == "Stored even if the LLM is down"
    assert memory.derived.get("enrichment_status") == "unavailable"
    assert memory.embedding is None


@pytest.mark.asyncio
async def test_ingestion_survives_llm_timeout(db):
    llm = FakeLLMProvider(fail="timeout")
    memory = await create_memory(db, llm, caption="Stored even if the LLM times out", uploads=[])
    assert memory.embedding is None
    assert memory.derived.get("enrichment_status") == "unavailable"


@pytest.mark.asyncio
async def test_ingestion_survives_malformed_structured_output(db):
    # Metadata extraction (generate_structured) fails, but embedding is a
    # separate call and independently succeeds -- the memory should reflect
    # exactly that: no derived metadata, but still vector-searchable.
    llm = FakeLLMProvider(fail="malformed")
    memory = await create_memory(db, llm, caption="Stored even with garbled LLM output", uploads=[])
    assert memory.derived.get("enrichment_status") == "unavailable"
    assert memory.embedding is not None


@pytest.mark.asyncio
async def test_recall_falls_back_to_fulltext_when_embedding_unavailable(db, fake_llm):
    memory = await create_memory(db, fake_llm, caption="quokka sighting at the zoo", uploads=[])

    broken_embed_only = FakeLLMProvider(fail="unavailable")
    # Even though embed() will raise for this provider, retrieve() must
    # fall back to full-text search rather than erroring out.
    from app.retrieval.pipeline import retrieve

    scored = await retrieve(db, broken_embed_only, "quokka sighting")
    assert scored and scored[0].memory.id == memory.id


@pytest.mark.asyncio
async def test_recall_abstains_rather_than_erroring_when_fully_unavailable(db, fake_llm):
    await create_memory(db, fake_llm, caption="Some memory", uploads=[])
    broken = FakeLLMProvider(fail="unavailable")
    result = await answer(db, broken, "a query matching nothing")
    assert result.abstained is True
