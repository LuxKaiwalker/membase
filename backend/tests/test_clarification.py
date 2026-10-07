import pytest

from app.ingestion.pipeline import MAX_CLARIFICATION_ROUNDS, create_capture, resolve_clarification
from app.retrieval.pipeline import hybrid_search
from tests.fakes import FakeLLMProvider
from tests.helpers import resolve_clarification_and_finish

_ZORBLEX_QUESTION = {
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
async def test_clarification_question_generated_and_surfaced(db):
    """The question-generation pass (app/ingestion/clarify.py) runs before
    -- and independently of -- full enrichment: a unit reaches
    awaiting_clarification without enrich() ever having been called."""
    llm = FakeLLMProvider(question_response=_ZORBLEX_QUESTION)

    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.pending_clarification == {"question": "What is a Zorblex code?", "field": "Zorblex code", "round": 1}
    assert unit.clarification_queue == []
    # Gated: a unit with an open clarification question is durably saved
    # but not yet searchable (see docs/DECISIONS.md).
    assert unit.status == "awaiting_clarification"
    assert "enrichment" not in llm.calls, "full enrichment must not run while a question is pending"


@pytest.mark.asyncio
async def test_clarification_answer_is_persisted(db):
    llm = FakeLLMProvider(
        question_response=[_ZORBLEX_QUESTION, _NO_QUESTIONS],
        enrichment_response=_GYM_SEMANTICS,
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    # Full enrichment (and therefore embedding) is deferred entirely while
    # a question is pending -- see docs/DECISIONS.md.
    assert unit.embedding is None

    unit = await resolve_clarification_and_finish(db, llm, unit, "It's my gym locker combination.")

    assert unit.pending_clarification is None
    assert len(unit.clarifications) == 1
    assert unit.clarifications[0]["question"] == "What is a Zorblex code?"
    assert unit.clarifications[0]["answer"] == "It's my gym locker combination."
    assert unit.semantic["aliases"] == ["Zorblex code", "gym locker combination"]
    assert "gym locker combination" in unit.retrieval_text
    assert unit.embedding is not None
    assert unit.status == "ready"


@pytest.mark.asyncio
async def test_blank_clarification_answer_skips_without_recording(db):
    """Answering is optional (soft-guarded client-side only, see
    docs/DECISIONS.md) -- a blank/omitted answer must still unblock the
    unit, but must not fabricate a clarification entry for an answer the
    user never actually gave."""
    llm = FakeLLMProvider(
        question_response=[_ZORBLEX_QUESTION, _NO_QUESTIONS],
        enrichment_response=dict(_GYM_SEMANTICS, aliases=[]),
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]

    unit = await resolve_clarification_and_finish(db, llm, unit, "")

    assert unit.pending_clarification is None
    assert unit.clarifications == []
    assert unit.status == "ready"


@pytest.mark.asyncio
async def test_clarification_rounds_are_bounded(db):
    """The model keeps wanting to ask more questions every round, but
    MAX_CLARIFICATION_ROUNDS caps it -- see docs/DECISIONS.md. Each round
    still only ever surfaces one question at a time to the user."""
    always_asks = {
        "questions": [{"question": "Another question?", "field": "x", "relevance": "high"}]
    }
    llm = FakeLLMProvider(question_response=always_asks, enrichment_response=dict(_GYM_SEMANTICS, aliases=[]))

    capture = await create_capture(db, llm, text="Ambiguous thing.", uploads=[])
    unit = capture.units[0]

    for expected_round in range(1, MAX_CLARIFICATION_ROUNDS + 1):
        assert unit.pending_clarification is not None, f"expected a question in round {expected_round}"
        assert unit.clarification_round == expected_round
        unit = await resolve_clarification_and_finish(db, llm, unit, "An answer.")

    # Round limit reached: even though the model still wants to ask,
    # the question must not be surfaced again -- enrichment runs instead.
    assert unit.pending_clarification is None
    assert unit.clarification_round == MAX_CLARIFICATION_ROUNDS
    assert unit.status == "ready"


@pytest.mark.asyncio
async def test_multiple_questions_in_one_round_asked_in_sequence(db):
    """generate_questions() can return several candidates at once; they're
    drained one at a time into pending_clarification (never shown as a
    batch), and draining the queue costs no extra LLM call."""
    batch = {
        "questions": [
            {"question": "What is a Zorblex code?", "field": "Zorblex code", "relevance": "high"},
            {"question": "Which gym is this?", "field": "gym", "relevance": "medium"},
        ]
    }
    llm = FakeLLMProvider(question_response=[batch, _NO_QUESTIONS], enrichment_response=_GYM_SEMANTICS)

    capture = await create_capture(db, llm, text="My Zorblex code is 42 at the gym.", uploads=[])
    unit = capture.units[0]
    assert unit.pending_clarification["question"] == "What is a Zorblex code?"
    assert unit.clarification_queue == [{"question": "Which gym is this?", "field": "gym"}]
    calls_after_first_round = llm.calls.count("questions")

    unit = await resolve_clarification_and_finish(db, llm, unit, "It's my gym locker combination.")
    # Second question drained from the queue -- no new "questions" call yet.
    assert unit.pending_clarification == {"question": "Which gym is this?", "field": "gym", "round": 1}
    assert unit.clarification_queue == []
    assert unit.status == "awaiting_clarification"
    assert llm.calls.count("questions") == calls_after_first_round

    unit = await resolve_clarification_and_finish(db, llm, unit, "Downtown gym.")
    assert unit.pending_clarification is None
    assert len(unit.clarifications) == 2
    assert unit.status == "ready"


@pytest.mark.asyncio
async def test_answer_can_trigger_a_new_round_of_questions(db):
    """Iterative: if answering the first question reveals something else
    worth asking about, a new round is opened (bounded by
    MAX_CLARIFICATION_ROUNDS, same as any other round)."""
    llm = FakeLLMProvider(
        question_response=[
            _ZORBLEX_QUESTION,
            {"questions": [{"question": "Which gym is this?", "field": "gym", "relevance": "high"}]},
            _NO_QUESTIONS,
        ],
        enrichment_response=_GYM_SEMANTICS,
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.clarification_round == 1

    unit = await resolve_clarification_and_finish(db, llm, unit, "It's my gym locker combination.")
    assert unit.pending_clarification == {"question": "Which gym is this?", "field": "gym", "round": 2}
    assert unit.clarification_round == 2
    assert unit.status == "awaiting_clarification"

    unit = await resolve_clarification_and_finish(db, llm, unit, "Downtown gym.")
    assert unit.pending_clarification is None
    assert unit.status == "ready"
    assert len(unit.clarifications) == 2


@pytest.mark.asyncio
async def test_retrieval_succeeds_via_clarification_derived_alias(db):
    """The core alias-retrieval property (spec's AHV acceptance test,
    generalized): a term introduced only during clarification must become
    a working retrieval cue, exercised here with FakeLLMProvider's
    content-hashed embeddings so this is really testing the full-text/
    alias-in-text pathway deterministically (the semantic/vector version
    of this is covered against the real model in
    tests/test_integration_real_llm.py)."""
    llm = FakeLLMProvider(
        question_response=[_ZORBLEX_QUESTION, _NO_QUESTIONS],
        enrichment_response=_GYM_SEMANTICS,
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = await resolve_clarification_and_finish(db, llm, capture.units[0], "It's my gym locker combination.")
    assert unit.status == "ready"

    scored = await hybrid_search(db, llm, "gym locker combination")
    assert scored and scored[0].unit.id == unit.id


@pytest.mark.asyncio
async def test_unit_awaiting_clarification_is_not_searchable(db):
    """The gate itself: before the question is answered, the unit must
    not surface in recall at all -- not even via an exact keyword match
    on its own authoritative_content."""
    llm = FakeLLMProvider(question_response=_ZORBLEX_QUESTION)

    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    scored = await hybrid_search(db, llm, "Zorblex code")
    assert scored == [], "an unanswered unit must not be searchable"

    fetched = db.get(type(unit), unit.id)
    assert fetched is not None
    assert fetched.authoritative_content == "My Zorblex code is 42."
