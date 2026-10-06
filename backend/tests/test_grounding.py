import pytest

from app.ingestion.pipeline import create_memory
from app.retrieval.pipeline import answer
from tests.fakes import FakeLLMProvider

_ABSTENTION = "I don't have a stored memory about this."


@pytest.mark.asyncio
async def test_abstains_with_no_memories(db, fake_llm):
    result = await answer(db, fake_llm, "anything")
    assert result.abstained is True
    assert result.answer == _ABSTENTION
    assert result.sources == []


@pytest.mark.asyncio
async def test_grounded_answer_cites_the_memory_it_used(db, fake_llm):
    memory = await create_memory(db, fake_llm, caption="Eiffel Tower trip notes", uploads=[])

    answering_llm = FakeLLMProvider(
        answer_response={"answer": "You visited the Eiffel Tower.", "grounded": True}
    )
    result = await answer(db, answering_llm, "Eiffel Tower")
    assert result.abstained is False
    assert result.answer == "You visited the Eiffel Tower."
    assert any(s.memory.id == memory.id for s in result.sources)


@pytest.mark.asyncio
async def test_ungrounded_model_output_is_overridden_with_abstention(db, fake_llm):
    # Even if the model writes *something*, grounded=False must force the
    # canonical abstention message and drop the source list -- the system
    # must never let a low-confidence guess reach the user framed as a fact.
    await create_memory(db, fake_llm, caption="Eiffel Tower trip notes", uploads=[])

    answering_llm = FakeLLMProvider(answer_response={"answer": "Maybe Rome?", "grounded": False})
    result = await answer(db, answering_llm, "Eiffel Tower")
    assert result.abstained is True
    assert result.answer == _ABSTENTION
    assert result.sources == []
