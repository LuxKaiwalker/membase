import pytest

from app.ingestion.pipeline import MAX_CLARIFICATION_ROUNDS, create_capture, resolve_clarification
from app.retrieval.pipeline import hybrid_search
from tests.fakes import FakeLLMProvider
from tests.helpers import resolve_clarification_and_finish

# Streamlining phase (see docs/DECISIONS.md "Sequential clarification"):
# there is no separate question-generation call and no pre-generated
# queue any more. One unified call (analyze_capture, see
# app/ingestion/analysis.py) decides segmentation + semantics +
# clarification need all at once; each clarification round costs exactly
# one further call (reanalyze_unit) that both re-derives semantics AND
# decides whether a *further* question is still warranted. This file
# replaces the old queue-based test_clarification.py.

_ZORBLEX_QUESTION = {"clarification_question": "What is a Zorblex code?", "clarification_field": "Zorblex code"}
_NO_QUESTION: dict = {}

_GYM_SEMANTICS_NO_QUESTION = {
    "aliases": ["Zorblex code", "gym locker combination"],
    "topics": ["gym"],
}


@pytest.mark.asyncio
async def test_clarification_question_generated_and_surfaced(db):
    """The unified analyze_capture() call decides both the semantics AND
    whether a question is needed, in one shot -- there's no separate
    question-generation pass any more."""
    llm = FakeLLMProvider(enrichment_response=_ZORBLEX_QUESTION)

    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.pending_clarification == {"question": "What is a Zorblex code?", "field": "Zorblex code", "round": 1}
    # Gated: a unit with an open clarification question is durably saved
    # but not yet searchable (see docs/DECISIONS.md).
    assert unit.status == "awaiting_clarification"
    assert unit.embedding is None
    assert llm.calls == ["analyze_capture"]


@pytest.mark.asyncio
async def test_clarification_answer_is_persisted(db):
    llm = FakeLLMProvider(enrichment_response=[_ZORBLEX_QUESTION, _GYM_SEMANTICS_NO_QUESTION])
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    # Embedding/retrieval text are deferred entirely while a question is
    # pending -- see docs/DECISIONS.md.
    assert unit.embedding is None

    unit = await resolve_clarification_and_finish(db, llm, unit, "It's my gym locker combination.")

    assert unit.pending_clarification is None
    assert len(unit.clarifications) == 1
    assert unit.clarifications[0]["question"] == "What is a Zorblex code?"
    assert unit.clarifications[0]["answer"] == "It's my gym locker combination."
    assert unit.clarifications[0]["outcome"] == "answered"
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
    llm = FakeLLMProvider(enrichment_response=[_ZORBLEX_QUESTION, _NO_QUESTION])
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
    always_asks = {"clarification_question": "Another question?", "clarification_field": "x"}
    llm = FakeLLMProvider(enrichment_response=always_asks)

    capture = await create_capture(db, llm, text="Ambiguous thing.", uploads=[])
    unit = capture.units[0]

    for expected_round in range(1, MAX_CLARIFICATION_ROUNDS + 1):
        assert unit.pending_clarification is not None, f"expected a question in round {expected_round}"
        assert unit.clarification_round == expected_round
        unit = await resolve_clarification_and_finish(db, llm, unit, "An answer.")

    # Round limit reached: even though the model still wants to ask,
    # the question must not be surfaced again -- finalization runs instead.
    assert unit.pending_clarification is None
    assert unit.clarification_round == MAX_CLARIFICATION_ROUNDS
    assert unit.status == "ready"


@pytest.mark.asyncio
async def test_declined_or_unknown_field_is_never_reasked(db):
    """A hard-coded guarantee, not just a prompt instruction: once the
    user has said "I don't know" (or declined) about a field, the system
    must never ask about that same field again, even if the model
    proposes it a second time (see docs/DECISIONS.md "Clarification
    outcomes")."""
    llm = FakeLLMProvider(enrichment_response=[_ZORBLEX_QUESTION, _ZORBLEX_QUESTION])
    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]

    unit = await resolve_clarification_and_finish(db, llm, unit, "I don't know.")

    assert unit.clarifications[0]["outcome"] == "unknown"
    # The model's second-round response still asks about "Zorblex code"
    # again, but that field was just marked unknown -- must be suppressed.
    assert unit.pending_clarification is None
    assert unit.status == "ready"


@pytest.mark.asyncio
async def test_answer_can_trigger_a_new_round_of_questions(db):
    """Iterative: if answering the first question reveals something else
    worth asking about, a new round is opened (bounded by
    MAX_CLARIFICATION_ROUNDS, same as any other round)."""
    llm = FakeLLMProvider(
        enrichment_response=[
            _ZORBLEX_QUESTION,
            {"clarification_question": "Which gym is this?", "clarification_field": "gym"},
            _GYM_SEMANTICS_NO_QUESTION,
        ]
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
    llm = FakeLLMProvider(enrichment_response=[_ZORBLEX_QUESTION, _GYM_SEMANTICS_NO_QUESTION])
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
    llm = FakeLLMProvider(enrichment_response=_ZORBLEX_QUESTION)

    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    scored = await hybrid_search(db, llm, "Zorblex code")
    assert scored == [], "an unanswered unit must not be searchable"

    fetched = db.get(type(unit), unit.id)
    assert fetched is not None
    assert fetched.authoritative_content == "My Zorblex code is 42."


@pytest.mark.asyncio
async def test_explicit_metadata_conflict_surfaces_as_clarification(db):
    """Explicit user-entered metadata always has higher authority than
    LLM-inferred metadata, but a genuine conflict (both sides non-empty
    and different) must be surfaced, never silently resolved either way
    (see docs/DECISIONS.md "Explicit metadata authority")."""
    llm = FakeLLMProvider(enrichment_response={"locations": ["Geneva"]})

    capture = await create_capture(
        db, llm, text="Notes from the office.", uploads=[], explicit_metadata={"location": "Zürich"}
    )
    unit = capture.units[0]

    assert unit.status == "awaiting_clarification"
    assert unit.pending_clarification["field"] == "metadata conflict"
    assert "Zürich" in unit.pending_clarification["question"]
    assert "geneva" in unit.pending_clarification["question"].lower()
    # Explicit value still wins for now -- both are retained so nothing
    # the model inferred is silently lost either.
    assert set(unit.semantic["locations"]) == {"Geneva", "Zürich"}


@pytest.mark.asyncio
async def test_explicit_metadata_with_no_conflict_is_applied_silently(db):
    """No disagreement -- explicit metadata is just folded into the
    semantic representation with no clarification needed."""
    llm = FakeLLMProvider(enrichment_response=_NO_QUESTION)

    capture = await create_capture(
        db, llm, text="Notes from the office.", uploads=[], explicit_metadata={"location": "Zürich"}
    )
    unit = capture.units[0]

    assert unit.status != "awaiting_clarification"
    assert unit.semantic["locations"] == ["Zürich"]
    assert unit.explicit_metadata == {"location": "Zürich"}


# --- Disable follow-up questions (see docs/DECISIONS.md "Disable
# follow-up questions"): allow_clarification=False must remove the
# clarification round-trip from the storage cycle entirely -- no unit
# may ever reach "awaiting_clarification", regardless of source.


@pytest.mark.asyncio
async def test_disabled_clarification_skips_a_model_proposed_question(db):
    llm = FakeLLMProvider(enrichment_response=_ZORBLEX_QUESTION)

    capture = await create_capture(
        db, llm, text="My Zorblex code is 42.", uploads=[], allow_clarification=False
    )
    unit = capture.units[0]

    assert unit.status != "awaiting_clarification"
    assert unit.pending_clarification is None
    assert unit.status == "processing"


@pytest.mark.asyncio
async def test_disabled_clarification_skips_an_explicit_metadata_conflict(db):
    llm = FakeLLMProvider(enrichment_response={"locations": ["Geneva"]})

    capture = await create_capture(
        db, llm, text="Notes from the office.", uploads=[],
        explicit_metadata={"location": "Zürich"}, allow_clarification=False,
    )
    unit = capture.units[0]

    assert unit.status != "awaiting_clarification"
    assert unit.pending_clarification is None
    # The explicit value still silently wins -- only the surfaced
    # question is suppressed, not the authority rule itself.
    assert set(unit.semantic["locations"]) == {"Geneva", "Zürich"}


@pytest.mark.asyncio
async def test_clarification_enabled_by_default(db):
    """allow_clarification defaults to True -- existing callers that
    don't pass it keep today's behavior unchanged."""
    llm = FakeLLMProvider(enrichment_response=_ZORBLEX_QUESTION)

    capture = await create_capture(db, llm, text="My Zorblex code is 42.", uploads=[])
    unit = capture.units[0]

    assert unit.status == "awaiting_clarification"
