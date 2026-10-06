"""Integration tests against the real, locally running Ollama service.

These validate retrieval/grounding *quality*, which the fast fakes in
tests/fakes.py cannot (their embeddings carry no real semantics). Skipped
automatically if Ollama isn't reachable, so the main suite stays runnable
without it -- but in this dev environment it's expected to be up (see
CLAUDE.md), so these should normally run.
"""

import httpx
import pytest

from app.config import get_settings
from app.ingestion.pipeline import create_memory
from app.llm.ollama_provider import OllamaProvider
from app.retrieval.pipeline import answer

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


@pytest.mark.asyncio
async def test_paraphrased_query_retrieves_relevant_memory(db, real_llm):
    await create_memory(
        db,
        real_llm,
        caption="My trip to Japan last year was amazing, especially the food in Osaka.",
        uploads=[],
    )
    await create_memory(db, real_llm, caption="Went grocery shopping for pasta and olive oil.", uploads=[])

    # Paraphrase: no word overlap with "food"/"Osaka"/"Japan" beyond the
    # implicit topic, forcing this through the semantic/vector channel.
    result = await answer(db, real_llm, "How was the cuisine on that trip abroad?")
    assert result.abstained is False
    assert any(s.memory.caption.startswith("My trip to Japan") for s in result.sources)


@pytest.mark.asyncio
async def test_indirect_multiconcept_query_is_grounded(db, real_llm):
    await create_memory(
        db,
        real_llm,
        caption=(
            "I met Anna at ETH yesterday. She studies robotics and recommended "
            "a paper about tactile sensing."
        ),
        uploads=[],
    )
    await create_memory(db, real_llm, caption="Finished reading a sci-fi novel, loved the ending.", uploads=[])

    result = await answer(db, real_llm, "Who was that person from ETH who recommended a robotics paper?")
    assert result.abstained is False
    assert "anna" in result.answer.lower()
    assert len(result.sources) >= 1


@pytest.mark.asyncio
async def test_no_answer_case_abstains(db, real_llm):
    await create_memory(db, real_llm, caption="Bought a new pair of running shoes.", uploads=[])
    result = await answer(db, real_llm, "What is my favorite pizza topping?")
    assert result.abstained is True
    assert result.answer == "I don't have a stored memory about this."
