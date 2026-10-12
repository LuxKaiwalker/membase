
from app.ingestion.pipeline import create_capture, finish_unit_processing, retry_enrichment
from app.models import MemoryUnit
from tests.helpers import create_capture_and_finish


async def test_unexpected_enrichment_failure_marks_unit_failed_not_silently_ready(db, fake_llm, monkeypatch):
    """Two-phase commit guarantee (docs/ARCHITECTURE.md): authoritative_content
    is committed before derived processing runs, so an unexpected bug in
    derived processing must leave the unit in a visibly 'failed' state --
    never a silently-wrong 'ready'."""
    import app.ingestion.pipeline as pipeline_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated bug in derived processing")

    monkeypatch.setattr(pipeline_module, "build_retrieval_text", _boom)

    capture = await create_capture(db, fake_llm, text="Authoritative content must survive a bug", uploads=[])
    unit = capture.units[0]
    # Full enrichment (where the bug lives) hasn't run yet -- it's
    # scheduled as a background task, not run inline by create_capture()
    # anymore (see docs/DECISIONS.md). Drive it explicitly, the test
    # equivalent of "the background task runs".
    assert unit.status == "processing"

    await finish_unit_processing(unit.id, llm=fake_llm)
    db.expire_all()
    assert unit.status == "failed"
    # The authoritative content committed in phase 1, before enrichment
    # ran, must still be intact.
    assert unit.authoritative_content == "Authoritative content must survive a bug"

    monkeypatch.undo()
    recovered = await retry_enrichment(db, fake_llm, unit)
    if recovered.status == "processing":
        await finish_unit_processing(recovered.id, llm=fake_llm)
        db.expire_all()
        db.refresh(recovered)
    assert recovered.status == "ready"
    assert recovered.authoritative_content == "Authoritative content must survive a bug"


async def test_retry_on_an_already_ready_unit_is_a_noop(db, fake_llm):
    """Regression for a race the status guard in api/units.py can't see:
    that guard reads status before calling retry_enrichment(), so it's
    stale if finish_unit_processing() was still genuinely running and
    finished in between (it holds a row lock for its whole duration --
    see finish_unit_processing's with_for_update). Without re-checking
    status after acquiring that same lock, retry_enrichment would stomp
    a freshly-"ready" unit back to "processing" and schedule a wasted
    re-run the moment the lock freed. Simulated here directly (without
    reproducing the actual lock-timing race) by just calling
    retry_enrichment on a unit that is already "ready"."""
    capture = await create_capture_and_finish(db, fake_llm, text="Already finished enrichment.")
    unit = capture.units[0]
    assert unit.status == "ready"
    original_semantic = unit.semantic

    result = await retry_enrichment(db, fake_llm, unit)

    assert result.status == "ready"
    assert result.semantic == original_semantic


async def test_capture_and_units_survive_a_fresh_session_lookup(db, fake_llm):
    capture = await create_capture_and_finish(db, fake_llm, text="Durable across a restart")
    unit_id = capture.units[0].id

    fetched = db.get(MemoryUnit, unit_id)
    assert fetched is not None
    assert fetched.authoritative_content == "Durable across a restart"
    assert fetched.status == "ready"
