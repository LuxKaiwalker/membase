import pytest

from app.ingestion.pipeline import create_capture, discard_unit, resolve_clarification
from app.models import MemoryUnit
from app.retrieval.pipeline import hybrid_search
from tests.fakes import FakeLLMProvider

_ASKS_QUESTION = {
    "questions": [{"question": "What is a Zorblex code?", "field": "Zorblex code", "relevance": "high"}]
}
_NO_QUESTIONS = {"questions": []}
_GYM_SEMANTICS = {
    "meaning": "The user's gym locker combination is 42.",
    "aliases": ["Zorblex code", "gym locker combination"],
    "people": [],
    "organizations": [],
    "locations": [],
    "topics": ["gym"],
    "definitions": [{"term": "Zorblex code", "definition": "gym locker combination"}],
    "relations": [],
    "event_date": "",
}


@pytest.mark.asyncio
async def test_awaiting_clarification_survives_a_fresh_session_lookup(db):
    """Durability of the gated state itself: a unit awaiting clarification
    must survive a restart-equivalent lookup, still gated, still carrying
    its authoritative content."""
    llm = FakeLLMProvider(question_response=_ASKS_QUESTION)
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit_id = capture.units[0].id

    fetched = db.get(MemoryUnit, unit_id)
    assert fetched.status == "awaiting_clarification"
    assert fetched.authoritative_content == "My Zorblex code is 42."
    assert fetched.pending_clarification["question"] == "What is a Zorblex code?"


@pytest.mark.asyncio
async def test_discard_removes_unit_from_pending_without_deleting_it(db):
    llm = FakeLLMProvider(question_response=_ASKS_QUESTION)
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    discarded = await discard_unit(db, unit)
    assert discarded.status == "discarded"
    assert discarded.pending_clarification is None

    # Still exists (soft, not a hard delete) -- authoritative content intact.
    fetched = db.get(MemoryUnit, unit.id)
    assert fetched is not None
    assert fetched.authoritative_content == "My Zorblex code is 42."

    # And, crucially, still not searchable.
    scored = await hybrid_search(db, llm, "Zorblex code")
    assert scored == []


@pytest.mark.asyncio
async def test_clarifying_transitions_from_gated_to_searchable(db):
    llm = FakeLLMProvider(
        question_response=[_ASKS_QUESTION, _NO_QUESTIONS],
        enrichment_response=_GYM_SEMANTICS,
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert (await hybrid_search(db, llm, "Zorblex code")) == []

    unit = await resolve_clarification(db, llm, unit, "It's my gym locker combination.")
    assert unit.status == "ready"
    scored = await hybrid_search(db, llm, "Zorblex code")
    assert scored and scored[0].unit.id == unit.id
