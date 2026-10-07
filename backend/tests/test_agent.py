import pytest

from app.ingestion.pipeline import create_capture
from app.retrieval.agent import MAX_AGENT_ITERATIONS, run_agent
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish

_ABSTENTION = "I don't have a stored memory about this."


@pytest.mark.asyncio
async def test_direct_search_stops_after_one_round_when_sufficient(db, fake_llm):
    cap_foo = await create_capture_and_finish(db, fake_llm, text="foo content here")
    await create_capture_and_finish(db, fake_llm, text="bar content elsewhere")

    # Default evaluate_response is sufficient=True immediately, so the
    # agent must never even consider searching for "bar".
    agent_llm = FakeLLMProvider(answer_response={"answer": "about foo", "grounded": True})
    result = await run_agent(db, agent_llm, "foo")

    assert "evaluate" not in agent_llm.calls or agent_llm.calls.count("evaluate") <= MAX_AGENT_ITERATIONS - 1
    assert result.abstained is False
    assert all(s.unit.capture_id == cap_foo.id for s in result.sources)


@pytest.mark.asyncio
async def test_reformulation_pulls_in_additional_evidence(db, fake_llm):
    cap_foo = await create_capture_and_finish(db, fake_llm, text="foo content here")
    cap_bar = await create_capture_and_finish(db, fake_llm, text="bar content elsewhere")

    agent_llm = FakeLLMProvider(
        evaluate_response=[
            {"sufficient": False, "reformulated_queries": ["bar"]},
            {"sufficient": True, "reformulated_queries": []},
        ],
        answer_response={"answer": "about foo and bar", "grounded": True},
    )
    result = await run_agent(db, agent_llm, "foo")

    found_captures = {s.unit.capture_id for s in result.sources}
    assert cap_foo.id in found_captures
    assert cap_bar.id in found_captures, "reformulated query 'bar' should have been searched and merged in"


@pytest.mark.asyncio
async def test_iteration_limit_is_bounded(db, fake_llm):
    await create_capture_and_finish(db, fake_llm, text="foo content here")

    # The model always wants to try one more reformulation; the loop must
    # still terminate at MAX_AGENT_ITERATIONS regardless.
    always_insufficient = {"sufficient": False, "reformulated_queries": ["something else new"]}
    agent_llm = FakeLLMProvider(
        evaluate_response=[always_insufficient] * (MAX_AGENT_ITERATIONS + 5),
        answer_response={"answer": "best effort", "grounded": True},
    )
    result = await run_agent(db, agent_llm, "foo")

    # iteration 0..MAX-1 each do a search; evaluate is only called when
    # there's a next iteration to use it for, i.e. MAX_AGENT_ITERATIONS - 1 times.
    assert agent_llm.calls.count("evaluate") == MAX_AGENT_ITERATIONS - 1
    assert result.abstained is False


@pytest.mark.asyncio
async def test_malformed_evaluate_output_degrades_to_best_effort_answer(db, fake_llm):
    await create_capture_and_finish(db, fake_llm, text="foo content here")

    broken = FakeLLMProvider(fail="malformed")
    result = await run_agent(db, broken, "foo")
    # evaluate and answer-generation both get malformed output; the agent
    # must not crash, and must not fabricate an answer either.
    assert result.abstained is True
    assert result.answer == _ABSTENTION


@pytest.mark.asyncio
async def test_unavailable_llm_degrades_to_fulltext_then_abstains_or_answers(db, fake_llm):
    await create_capture_and_finish(db, fake_llm, text="findable by fulltext only")

    broken = FakeLLMProvider(fail="unavailable")
    result = await run_agent(db, broken, "findable by fulltext")
    # embed() fails -> vector channel empty, fts still works -> candidates
    # found -> but answer generation also fails (LLM down) -> must abstain,
    # never guess.
    assert result.abstained is True
    assert result.answer == _ABSTENTION


@pytest.mark.asyncio
async def test_no_answer_case_abstains_without_any_search_noise(db, fake_llm):
    result = await run_agent(db, fake_llm, "nothing stored matches this at all")
    assert result.abstained is True
    assert result.sources == []
