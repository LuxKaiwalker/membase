import pytest

from app.ingestion.pipeline import MAX_CLARIFICATION_ROUNDS, resolve_clarification
from app.retrieval.pipeline import hybrid_search
from tests.fakes import FakeLLMProvider


@pytest.mark.asyncio
async def test_clarification_question_generated_and_surfaced(db):
    llm = FakeLLMProvider(
        enrichment_response={
            "meaning": "",
            "aliases": [],
            "people": [],
            "organizations": [],
            "locations": [],
            "topics": [],
            "definitions": [],
            "relations": [],
            "event_date": "",
            "clarification_question": "What is a Zorblex code?",
            "clarification_field": "Zorblex code",
        }
    )
    from app.ingestion.pipeline import create_capture

    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.pending_clarification == {"question": "What is a Zorblex code?", "field": "Zorblex code", "round": 0}
    # Gated: a unit with an open clarification question is durably saved
    # but not yet searchable (see docs/DECISIONS.md).
    assert unit.status == "awaiting_clarification"


@pytest.mark.asyncio
async def test_clarification_answer_is_persisted(db):
    from app.ingestion.pipeline import create_capture

    llm = FakeLLMProvider(
        enrichment_response=[
            {
                "meaning": "",
                "aliases": [],
                "people": [],
                "organizations": [],
                "locations": [],
                "topics": [],
                "definitions": [],
                "relations": [],
                "event_date": "",
                "clarification_question": "What is a Zorblex code?",
                "clarification_field": "Zorblex code",
            },
            {
                "meaning": "The user's gym locker combination is 42.",
                "aliases": ["Zorblex code", "gym locker combination"],
                "people": [],
                "organizations": [],
                "locations": [],
                "topics": ["gym"],
                "definitions": [{"term": "Zorblex code", "definition": "gym locker combination"}],
                "relations": [],
                "event_date": "",
                "clarification_question": "",
                "clarification_field": "",
            },
        ]
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    before_embedding = unit.embedding

    unit = await resolve_clarification(db, llm, unit, "It's my gym locker combination.")

    assert unit.pending_clarification is None
    assert len(unit.clarifications) == 1
    assert unit.clarifications[0]["question"] == "What is a Zorblex code?"
    assert unit.clarifications[0]["answer"] == "It's my gym locker combination."
    assert unit.semantic["aliases"] == ["Zorblex code", "gym locker combination"]
    assert "gym locker combination" in unit.retrieval_text
    assert unit.embedding is not None
    assert list(unit.embedding) != list(before_embedding)  # re-embedded, not left stale


@pytest.mark.asyncio
async def test_clarification_rounds_are_bounded(db):
    """The model keeps wanting to ask more questions, but
    MAX_CLARIFICATION_ROUNDS caps it -- see docs/DECISIONS.md."""
    from app.ingestion.pipeline import create_capture

    always_asks = {
        "meaning": "",
        "aliases": [],
        "people": [],
        "organizations": [],
        "locations": [],
        "topics": [],
        "definitions": [],
        "relations": [],
        "event_date": "",
        "clarification_question": "Another question?",
        "clarification_field": "x",
    }
    llm = FakeLLMProvider(enrichment_response=[always_asks] * 5)
    capture = await create_capture(db, llm, text="Ambiguous thing.", uploads=[])
    unit = capture.units[0]
    assert unit.clarification_round == 0
    assert unit.pending_clarification is not None

    unit = await resolve_clarification(db, llm, unit, "An answer.")
    assert unit.clarification_round == MAX_CLARIFICATION_ROUNDS
    # Round limit reached: even though the model still wants to ask,
    # the question must not be surfaced again.
    assert unit.pending_clarification is None


@pytest.mark.asyncio
async def test_retrieval_succeeds_via_clarification_derived_alias(db):
    """The core alias-retrieval property (spec's AHV acceptance test,
    generalized): a term introduced only during clarification must become
    a working retrieval cue, exercised here with FakeLLMProvider's
    content-hashed embeddings so this is really testing the full-text/
    alias-in-text pathway deterministically (the semantic/vector version
    of this is covered against the real model in
    tests/test_integration_real_llm.py)."""
    from app.ingestion.pipeline import create_capture

    llm = FakeLLMProvider(
        enrichment_response=[
            {
                "meaning": "",
                "aliases": [],
                "people": [],
                "organizations": [],
                "locations": [],
                "topics": [],
                "definitions": [],
                "relations": [],
                "event_date": "",
                "clarification_question": "What is a Zorblex code?",
                "clarification_field": "Zorblex code",
            },
            {
                "meaning": "The user's gym locker combination is 42.",
                "aliases": ["Zorblex code", "gym locker combination"],
                "people": [],
                "organizations": [],
                "locations": [],
                "topics": ["gym"],
                "definitions": [{"term": "Zorblex code", "definition": "gym locker combination"}],
                "relations": [],
                "event_date": "",
                "clarification_question": "",
                "clarification_field": "",
            },
        ]
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = await resolve_clarification(db, llm, capture.units[0], "It's my gym locker combination.")
    assert unit.status == "ready"

    scored = await hybrid_search(db, llm, "gym locker combination")
    assert scored and scored[0].unit.id == unit.id


@pytest.mark.asyncio
async def test_unit_awaiting_clarification_is_not_searchable(db):
    """The gate itself: before the question is answered, the unit must
    not surface in recall at all -- not even via an exact keyword match
    on its own authoritative_content."""
    from app.ingestion.pipeline import create_capture

    llm = FakeLLMProvider(
        enrichment_response={
            "meaning": "",
            "aliases": [],
            "people": [],
            "organizations": [],
            "locations": [],
            "topics": [],
            "definitions": [],
            "relations": [],
            "event_date": "",
            "clarification_question": "What is a Zorblex code?",
            "clarification_field": "Zorblex code",
        }
    )
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    scored = await hybrid_search(db, llm, "Zorblex code")
    assert scored == [], "an unanswered unit must not be searchable"

    fetched = db.get(type(unit), unit.id)
    assert fetched is not None
    assert fetched.authoritative_content == "My Zorblex code is 42."
