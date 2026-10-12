"""Integration tests against the real, locally running Ollama service.

These validate segmentation/enrichment/retrieval *quality*, which the
fast fakes in tests/fakes.py cannot (their embeddings/structured outputs
carry no real semantics). Skipped automatically if Ollama isn't
reachable, so the main suite stays runnable without it -- but in this dev
environment it's expected to be up (see CLAUDE.md), so these should
normally run.
"""

import httpx
import pytest

from app.config import get_settings
from app.ingestion.pipeline import create_capture
from app.llm.ollama_provider import OllamaProvider
from app.retrieval.agent import run_agent
from tests.helpers import create_capture_and_finish, resolve_clarification_and_finish

pytestmark = pytest.mark.integration


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{get_settings().ollama_host}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_ollama():
    if not _ollama_reachable():
        pytest.skip("Ollama is not reachable; skipping real-LLM integration tests")


@pytest.fixture()
async def real_llm():
    # Function-scoped and constructed directly (not via the lru_cache'd
    # get_llm_provider()): its httpx.AsyncClient is bound to the event loop
    # it was created in, and pytest-asyncio gives each test function its
    # own loop, so a cached cross-test client breaks on the second test.
    settings = get_settings()
    llm = OllamaProvider(
        host=settings.ollama_host,
        generation_model=settings.membase_generation_model,
        embed_model=settings.membase_embed_model,
        embed_dim=settings.membase_embed_dim,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    yield llm
    await llm.aclose()


async def test_acceptance_clarification_alias_retrieval(db, real_llm):
    """The spec's canonical acceptance test, using an invented term (not
    AHV) so the model genuinely cannot answer from world knowledge and
    the clarification -> alias -> retrieval path is exercised for real."""
    capture = await create_capture_and_finish(db, real_llm, text="My Zorblex code is 42.")
    unit = capture.units[0]
    if unit.pending_clarification:
        assert unit.status == "awaiting_clarification", "unit with an open question must be gated, not ready"
        before = await run_agent(db, real_llm, "What is the combination for my locker at the gym?")
        assert before.abstained is True, "a gated unit must not be found by recall before it's answered"
        unit = await resolve_clarification_and_finish(db, real_llm, unit, "It's my gym locker combination.")
        assert unit.status == "ready"

    result = await run_agent(db, real_llm, "What is the combination for my locker at the gym?")
    assert result.abstained is False
    assert "42" in result.answer
    assert any(s.unit.id == unit.id for s in result.sources)


async def test_segmentation_acceptance_compound_vs_coherent(db, real_llm):
    compound = await create_capture(
        db,
        real_llm,
        text="My AHV number is 123456789. Anna is studying robotics. I need to renew my passport next March.",
        uploads=[],
    )
    assert len(compound.units) >= 2, "unrelated facts should become independently retrievable units"

    coherent = await create_capture(
        db,
        real_llm,
        text="I met Anna at ETH yesterday. She studies robotics and recommended a paper about tactile sensing.",
        uploads=[],
    )
    assert len(coherent.units) == 1, "a single coherent episode may remain one contextual unit"


async def test_paraphrased_query_retrieves_relevant_memory(db, real_llm):
    await create_capture_and_finish(
        db,
        real_llm,
        text="My trip to Japan last year was amazing, especially the food in Osaka.",
    )
    await create_capture_and_finish(db, real_llm, text="Went grocery shopping for pasta and olive oil.")

    # Paraphrase: no word overlap with "food"/"Osaka"/"Japan" beyond the
    # implicit topic, forcing this through the semantic/vector channel.
    result = await run_agent(db, real_llm, "How was the cuisine on that trip abroad?")
    assert result.abstained is False
    assert any(s.unit.authoritative_content.startswith("My trip to Japan") for s in result.sources)


async def test_indirect_multiconcept_relationship_query(db, real_llm):
    await create_capture_and_finish(
        db,
        real_llm,
        text=(
            "I met Anna at ETH yesterday. She studies robotics and recommended "
            "a paper about tactile sensing."
        ),
    )
    await create_capture_and_finish(db, real_llm, text="Finished reading a sci-fi novel, loved the ending.")

    result = await run_agent(db, real_llm, "Who was that person from ETH who recommended a robotics paper?")
    assert result.abstained is False
    assert "anna" in result.answer.lower()


async def test_no_answer_case_abstains(db, real_llm):
    await create_capture_and_finish(db, real_llm, text="Bought a new pair of running shoes.")
    result = await run_agent(db, real_llm, "What is my favorite pizza topping?")
    assert result.abstained is True
    assert result.answer == "I don't have a stored memory about this."
