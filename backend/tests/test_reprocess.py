from app.ingestion.pipeline import CURRENT_PIPELINE_VERSION, reprocess_unit
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish, create_capture_and_finish_unstructured, resolve_clarification_and_finish

# Regression coverage for the low-priority reprocessing pipeline (see
# docs/DECISIONS.md "Reprocessing pipeline" and scripts/reprocess_memories.py).
# A unit created by create_capture_and_finish() is immediately stamped
# with CURRENT_PIPELINE_VERSION (it was just processed by today's code) --
# these tests force pipeline_version back to 0 after creation to simulate
# a pre-existing unit left behind by an older pipeline, the actual
# scenario scripts/reprocess_memories.py's query selects for. They then
# exercise reprocess_unit() directly, the same function the script calls
# per unit.


async def test_reprocess_ready_unit_reenriches_and_stamps_version(db):
    # A 2-item list: one enrich_unit() response per call -- the first
    # consumed by the initial analysis, the second by reprocess_unit()
    # below (see fakes.py's _ResponseQueue: "a list's Nth item is the
    # Nth UNIT's response", i.e. one item per enrich_unit() invocation,
    # not per sub-call). Uses the *_unstructured helper so the membase
    # structurer's own chained pass -- which draws from this same
    # enrichment queue -- doesn't consume the second item first.
    llm = FakeLLMProvider(enrichment_response=[{"meaning": "first pass"}, {"meaning": "corrected after fix"}])
    capture = await create_capture_and_finish_unstructured(db, llm, text="A short memory.")
    unit = capture.units[0]
    assert unit.status == "ready"
    assert unit.semantic["meaning"] == "first pass"
    unit.pipeline_version = 0  # simulate a unit left behind by an older pipeline
    db.commit()

    updated = await reprocess_unit(db, llm, unit)

    assert updated.status == "ready"
    assert updated.pipeline_version == CURRENT_PIPELINE_VERSION
    assert updated.semantic["meaning"] == "corrected after fix"


async def test_reprocess_awaiting_clarification_resolves_when_fix_removes_ambiguity(db):
    llm = FakeLLMProvider(clarification_response=[
        {"clarification_question": "What do you mean?", "clarification_field": "x"},
        {"clarification_question": "", "clarification_field": ""},
    ])
    capture = await create_capture_and_finish(db, llm, text="A short memory.")
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"
    assert unit.pending_clarification["field"] == "x"

    updated = await reprocess_unit(db, llm, unit)

    assert updated.status == "ready"
    assert updated.pending_clarification is None
    assert updated.pipeline_version == CURRENT_PIPELINE_VERSION
    # Auto-resolving via reprocessing is not a user answer -- nothing
    # synthetic gets appended to the permanent Q&A history.
    assert updated.clarifications == []


async def test_reprocess_replaces_pending_question_in_place_without_bumping_round(db):
    llm = FakeLLMProvider(clarification_response=[
        {"clarification_question": "Which left eye values?", "clarification_field": "left_eye"},
        {"clarification_question": "Which right eye values?", "clarification_field": "right_eye"},
    ])
    capture = await create_capture_and_finish(db, llm, text="A short memory.")
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"
    assert unit.clarification_round == 1
    assert unit.pending_clarification["field"] == "left_eye"

    updated = await reprocess_unit(db, llm, unit)

    assert updated.status == "awaiting_clarification"
    # Still exactly one pending question -- never two.
    assert updated.pending_clarification["field"] == "right_eye"
    assert updated.clarification_round == 1
    assert updated.clarifications == []
    assert updated.pipeline_version == CURRENT_PIPELINE_VERSION


async def test_reprocess_never_reasks_a_declined_field(db):
    # Three generate_question() calls happen across this test, in order:
    # the initial analysis, resolve_clarification's follow-up round, and
    # finally reprocess_unit() below -- one queue item each (the queue
    # repeats its last item once exhausted, but all three are supplied
    # explicitly here so nothing relies on that fallback).
    llm = FakeLLMProvider(clarification_response=[
        {"clarification_question": "What is X?", "clarification_field": "x"},
        {"clarification_question": "What is Y?", "clarification_field": "y"},
        {"clarification_question": "What is X, really?", "clarification_field": "x"},
    ])
    capture = await create_capture_and_finish(db, llm, text="A short memory.")
    unit = capture.units[0]
    assert unit.pending_clarification["field"] == "x"

    # Answering "I don't know" marks field "x" declined/unknown and
    # opens a second round on field "y" (the queue's second item).
    unit = await resolve_clarification_and_finish(db, llm, unit, "I don't know")
    assert unit.status == "awaiting_clarification"
    assert unit.pending_clarification["field"] == "y"
    assert unit.clarifications[0]["field"] == "x"
    assert unit.clarifications[0]["outcome"] == "unknown"

    # Reprocessing proposes field "x" again (simulating a fix that makes
    # clarify re-surface the same field) -- the declined/unknown guard
    # must still filter it out, same as the live resolve_clarification
    # path already guarantees, so the unit resolves straight to ready
    # instead of re-asking something the user already couldn't answer.
    updated = await reprocess_unit(db, llm, unit)

    assert updated.status == "ready"
    assert updated.pending_clarification is None
    assert updated.pipeline_version == CURRENT_PIPELINE_VERSION


async def test_reprocess_skips_units_not_ready_or_awaiting_clarification(db):
    llm = FakeLLMProvider(fail="unavailable")  # any LLM call here would raise and fail the test
    capture = await create_capture_and_finish(db, FakeLLMProvider(), text="A short memory.")
    unit = capture.units[0]
    unit.status = "discarded"
    unit.pipeline_version = 0
    db.commit()

    updated = await reprocess_unit(db, llm, unit)

    assert updated.status == "discarded"
    assert updated.pipeline_version == 0  # untouched -- this pass is not this unit's job
