import pytest

from app.retrieval.pipeline import hybrid_search
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish


@pytest.mark.asyncio
async def test_ingestion_survives_llm_unavailable(db):
    llm = FakeLLMProvider(fail="unavailable")
    capture = await create_capture_and_finish(db, llm, text="Stored even if the LLM is down")
    unit = capture.units[0]
    assert unit.authoritative_content == "Stored even if the LLM is down"
    assert unit.status == "ready"
    assert unit.semantic.get("enrichment_status") == "unavailable"
    assert unit.embedding is None


@pytest.mark.asyncio
async def test_ingestion_survives_llm_timeout(db):
    llm = FakeLLMProvider(fail="timeout")
    capture = await create_capture_and_finish(db, llm, text="Stored even if the LLM times out")
    assert capture.units[0].embedding is None
    assert capture.units[0].status == "ready"


@pytest.mark.asyncio
async def test_ingestion_survives_malformed_structured_output(db):
    # Metadata extraction (generate_structured) fails, but embedding is a
    # separate call and independently succeeds -- the unit should reflect
    # exactly that: no derived semantics, but still vector-searchable.
    llm = FakeLLMProvider(fail="malformed")
    capture = await create_capture_and_finish(db, llm, text="Stored even with garbled LLM output")
    unit = capture.units[0]
    assert unit.status == "ready"
    assert unit.semantic.get("enrichment_status") == "unavailable"
    assert unit.embedding is not None


@pytest.mark.asyncio
async def test_recall_falls_back_to_fulltext_when_embedding_unavailable(db, fake_llm):
    capture = await create_capture_and_finish(db, fake_llm, text="quokka sighting at the zoo")

    broken_embed_only = FakeLLMProvider(fail="unavailable")
    # Even though embed() will raise for this provider, hybrid_search()
    # must fall back to full-text search rather than erroring out.
    scored = await hybrid_search(db, broken_embed_only, "quokka sighting")
    assert scored and scored[0].unit.capture_id == capture.id
