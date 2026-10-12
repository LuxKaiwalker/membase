import datetime as dt

from app.ingestion.pipeline import create_capture
from app.models import MemoryUnit
from app.retrieval.pipeline import ScoredUnit, format_evidence
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish

# Streamlining phase (see docs/DECISIONS.md "Contradiction detection"):
# a cheap, deterministic comparison of (subject, predicate)->object
# relation triples already produced by analysis, reusing the existing
# hybrid_search() call -- no extra LLM call, never deletes/supersedes,
# only an advisory marker surfaced explicitly at answer time.


async def test_conflicting_relations_are_marked_on_both_units(db):
    llm_a = FakeLLMProvider(enrichment_response={"relations": [{"subject": "rent", "predicate": "is", "object": "1200 dollars"}]})
    capture_a = await create_capture_and_finish(db, llm_a, text="My rent is 1200 dollars.", uploads=[])
    unit_a = capture_a.units[0]
    assert unit_a.potential_conflicts == []

    llm_b = FakeLLMProvider(enrichment_response={"relations": [{"subject": "rent", "predicate": "is", "object": "1400 dollars"}]})
    capture_b = await create_capture_and_finish(db, llm_b, text="My rent is 1400 dollars.", uploads=[])
    unit_b = capture_b.units[0]

    assert len(unit_b.potential_conflicts) == 1
    assert unit_b.potential_conflicts[0]["unit_id"] == str(unit_a.id)

    db.refresh(unit_a)
    assert len(unit_a.potential_conflicts) == 1
    assert unit_a.potential_conflicts[0]["unit_id"] == str(unit_b.id)


async def test_agreeing_relations_are_not_marked_as_conflicting(db, fake_llm):
    llm_a = FakeLLMProvider(enrichment_response={"relations": [{"subject": "rent", "predicate": "is", "object": "1200 dollars"}]})
    await create_capture_and_finish(db, llm_a, text="My rent is 1200 dollars.", uploads=[])

    llm_b = FakeLLMProvider(enrichment_response={"relations": [{"subject": "rent", "predicate": "is", "object": "1200 dollars"}]})
    capture_b = await create_capture_and_finish(db, llm_b, text="My rent is also 1200 dollars.", uploads=[])
    unit_b = capture_b.units[0]

    assert unit_b.potential_conflicts == []


async def test_conflicts_never_delete_or_change_either_units_content(db):
    """Advisory only: a detected conflict must never mutate authoritative
    content, status, or semantics of either side -- see docs/DECISIONS.md
    "Do not overbuild contradiction resolution"."""
    llm_a = FakeLLMProvider(enrichment_response={"relations": [{"subject": "rent", "predicate": "is", "object": "1200 dollars"}]})
    capture_a = await create_capture_and_finish(db, llm_a, text="My rent is 1200 dollars.", uploads=[])
    unit_a = capture_a.units[0]

    llm_b = FakeLLMProvider(enrichment_response={"relations": [{"subject": "rent", "predicate": "is", "object": "1400 dollars"}]})
    await create_capture_and_finish(db, llm_b, text="My rent is 1400 dollars.", uploads=[])

    db.refresh(unit_a)
    assert unit_a.status == "ready"
    assert unit_a.authoritative_content == "My rent is 1200 dollars."
    assert unit_a.semantic["relations"] == [{"subject": "rent", "predicate": "is", "object": "1200 dollars"}]


def test_conflict_note_surfaces_in_formatted_evidence():
    """Deterministic (non-LLM) formatting: a unit carrying a
    potential_conflicts marker must produce a labeled "[Conflict]" line
    in the evidence block fed to answer generation, so the prompt
    instruction to state the disagreement explicitly has something to
    act on (see docs/DECISIONS.md)."""
    unit = MemoryUnit(
        authoritative_content="My rent is 1200 dollars.",
        semantic={},
        potential_conflicts=[{"unit_id": "other-id", "note": "'rent is' -- here: '1200 dollars', in another memory: '1400 dollars'."}],
        created_at=dt.datetime.now(dt.UTC),
    )
    scored = ScoredUnit(unit=unit, vector_score=0.9, fts_score=1.0, fused_score=0.9)

    evidence = format_evidence([scored], [])

    assert "[Conflict]" in evidence
    assert "1400 dollars" in evidence


def test_unit_with_no_conflicts_has_no_conflict_line():
    unit = MemoryUnit(
        authoritative_content="My rent is 1200 dollars.",
        semantic={},
        potential_conflicts=[],
        created_at=dt.datetime.now(dt.UTC),
    )
    scored = ScoredUnit(unit=unit, vector_score=0.9, fts_score=1.0, fused_score=0.9)

    evidence = format_evidence([scored], [])

    assert "[Conflict]" not in evidence
