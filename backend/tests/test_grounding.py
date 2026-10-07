import pytest

from app.ingestion.pipeline import create_capture
from app.retrieval.agent import run_agent
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish

_ABSTENTION = "I don't have a stored memory about this."


@pytest.mark.asyncio
async def test_abstains_with_no_memories(db, fake_llm):
    result = await run_agent(db, fake_llm, "anything")
    assert result.abstained is True
    assert result.answer == _ABSTENTION
    assert result.sources == []


@pytest.mark.asyncio
async def test_grounded_answer_cites_the_memory_it_used(db, fake_llm):
    capture = await create_capture_and_finish(db, fake_llm, text="Eiffel Tower trip notes")

    answering_llm = FakeLLMProvider(
        answer_response={"answer": "You visited the Eiffel Tower.", "grounded": True}
    )
    result = await run_agent(db, answering_llm, "Eiffel Tower")
    assert result.abstained is False
    assert result.answer == "You visited the Eiffel Tower."
    assert any(s.unit.capture_id == capture.id for s in result.sources)


@pytest.mark.asyncio
async def test_ungrounded_model_output_is_overridden_with_abstention(db, fake_llm):
    # Even if the model writes *something*, grounded=False must force the
    # canonical abstention message and drop the source list -- the system
    # must never let a low-confidence guess reach the user framed as a fact.
    await create_capture_and_finish(db, fake_llm, text="Eiffel Tower trip notes")

    answering_llm = FakeLLMProvider(answer_response={"answer": "Maybe Rome?", "grounded": False})
    result = await run_agent(db, answering_llm, "Eiffel Tower")
    assert result.abstained is True
    assert result.answer == _ABSTENTION
    assert result.sources == []
