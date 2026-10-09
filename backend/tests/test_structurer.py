import pytest

from app.ingestion.pipeline import IngestionError, finish_structuring, retry_structuring
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish

# The "membase structurer" (see docs/DECISIONS.md "The 'membase
# structurer'"): a second, independent background pass that deepens an
# already-"ready" unit's representation. The one invariant every test
# here is really guarding: `status` (searchability) is NEVER touched by
# structuring, in success or failure.


@pytest.mark.asyncio
async def test_structuring_upgrades_semantics_without_touching_status(db):
    llm = FakeLLMProvider(enrichment_response={"topics": ["gym"]})
    capture = await create_capture_and_finish(db, llm, text="My gym locker combination is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.status == "ready"
    assert unit.structure_status == "unstructured"

    structure_llm = FakeLLMProvider(enrichment_response={"topics": ["locker combination", "gym"], "related_indices": []})
    await finish_structuring(unit.id, llm=structure_llm)
    db.refresh(unit)

    assert unit.status == "ready"  # never touched
    assert unit.structure_status == "structured"
    assert unit.structured_at is not None
    assert unit.semantic["topics"] == ["locker combination", "gym"]
    assert "locker combination" in unit.retrieval_text
    assert unit.embedding is not None


@pytest.mark.asyncio
async def test_structuring_skips_a_unit_that_is_not_ready(db, fake_llm):
    llm = FakeLLMProvider(enrichment_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"})
    capture = await create_capture_and_finish(db, llm, text="My Flimvex PIN is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    await finish_structuring(unit.id, llm=fake_llm)
    db.refresh(unit)

    assert unit.structure_status == "unstructured"  # never claimed


@pytest.mark.asyncio
async def test_structuring_is_a_noop_if_already_structured(db):
    llm = FakeLLMProvider(enrichment_response={"topics": ["gym"]})
    capture = await create_capture_and_finish(db, llm, text="My gym locker combination is 42.", uploads=[])
    unit = capture.units[0]

    structure_llm = FakeLLMProvider(enrichment_response={"topics": ["first pass"]})
    await finish_structuring(unit.id, llm=structure_llm)
    db.refresh(unit)
    assert unit.structure_status == "structured"

    second_llm = FakeLLMProvider(enrichment_response={"topics": ["should never be applied"]})
    await finish_structuring(unit.id, llm=second_llm)
    db.refresh(unit)
    assert unit.semantic["topics"] == ["first pass"]  # untouched by the second call


@pytest.mark.asyncio
async def test_structuring_failure_leaves_existing_representation_untouched(db):
    llm = FakeLLMProvider(enrichment_response={"topics": ["gym"]})
    capture = await create_capture_and_finish(db, llm, text="My gym locker combination is 42.", uploads=[])
    unit = capture.units[0]
    original_retrieval_text = unit.retrieval_text
    original_embedding = unit.embedding

    broken_llm = FakeLLMProvider(fail="unavailable")
    await finish_structuring(unit.id, llm=broken_llm)
    db.refresh(unit)

    assert unit.status == "ready"
    assert unit.structure_status == "failed"
    assert unit.retrieval_text == original_retrieval_text
    assert list(unit.embedding) == list(original_embedding)
    assert unit.semantic["topics"] == ["gym"]  # never overwritten with empty/degraded data


@pytest.mark.asyncio
async def test_related_candidates_are_recorded_as_advisory_links(db):
    llm_a = FakeLLMProvider(enrichment_response={"topics": ["robotics"]})
    capture_a = await create_capture_and_finish(db, llm_a, text="Anna recommended a robotics paper.", uploads=[])
    unit_a = capture_a.units[0]

    llm_b = FakeLLMProvider(enrichment_response={"topics": ["robotics"]})
    capture_b = await create_capture_and_finish(db, llm_b, text="I read the robotics paper Anna recommended.", uploads=[])
    unit_b = capture_b.units[0]

    structure_llm = FakeLLMProvider(enrichment_response={"topics": ["robotics"], "related_indices": [0]})
    await finish_structuring(unit_b.id, llm=structure_llm)
    db.refresh(unit_b)

    assert str(unit_a.id) in unit_b.related_unit_ids


@pytest.mark.asyncio
async def test_related_candidates_match_on_partial_overlap_not_every_term(db):
    """websearch_to_tsquery ANDs bare terms by default -- joining
    several about-terms with a plain space would require a related
    memory to share EVERY one of them. Two memories sharing only ONE
    term (same location, different topic/people) must still be found
    as candidates (see docs/DECISIONS.md)."""
    llm_a = FakeLLMProvider(enrichment_response={"topics": ["robotics conference"], "locations": ["Rome"]})
    capture_a = await create_capture_and_finish(db, llm_a, text="I attended a robotics conference in Rome.", uploads=[])
    unit_a = capture_a.units[0]

    llm_b = FakeLLMProvider(enrichment_response={"topics": ["pasta", "food"], "locations": ["Rome"], "people": ["Anna"]})
    capture_b = await create_capture_and_finish(db, llm_b, text="I had amazing pasta in Rome with Anna.", uploads=[])
    unit_b = capture_b.units[0]

    structure_llm = FakeLLMProvider(enrichment_response={"topics": ["pasta", "food"], "related_indices": [0]})
    await finish_structuring(unit_b.id, llm=structure_llm)
    db.refresh(unit_b)

    assert str(unit_a.id) in unit_b.related_unit_ids


@pytest.mark.asyncio
async def test_retry_structuring_rejects_a_unit_not_yet_ready(db, fake_llm):
    llm = FakeLLMProvider(enrichment_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"})
    capture = await create_capture_and_finish(db, llm, text="My Flimvex PIN is 42.", uploads=[])
    unit = capture.units[0]

    with pytest.raises(IngestionError):
        await retry_structuring(db, fake_llm, unit)


@pytest.mark.asyncio
async def test_retry_structuring_reclaims_a_failed_unit(db):
    llm = FakeLLMProvider(enrichment_response={"topics": ["gym"]})
    capture = await create_capture_and_finish(db, llm, text="My gym locker combination is 42.", uploads=[])
    unit = capture.units[0]

    broken_llm = FakeLLMProvider(fail="unavailable")
    await finish_structuring(unit.id, llm=broken_llm)
    db.refresh(unit)
    assert unit.structure_status == "failed"

    working_llm = FakeLLMProvider(enrichment_response={"topics": ["recovered"]})
    updated = await retry_structuring(db, working_llm, unit)

    assert updated.structure_status == "structured"
    assert updated.semantic["topics"] == ["recovered"]


@pytest.mark.asyncio
async def test_retry_structuring_leaves_a_fresh_in_flight_lease_alone(db):
    from app.ingestion.pipeline import _claim_structuring

    llm = FakeLLMProvider(enrichment_response={"topics": ["gym"]})
    capture = await create_capture_and_finish(db, llm, text="My gym locker combination is 42.", uploads=[])
    unit = capture.units[0]
    _claim_structuring(db, unit)
    db.refresh(unit)
    assert unit.structure_status == "structuring"

    result = await retry_structuring(db, llm, unit)
    assert result.structure_status == "structuring"  # left alone -- lease is fresh
