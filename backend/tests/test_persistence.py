import pytest

from app.ingestion.pipeline import create_capture, retry_enrichment
from app.models import MemoryUnit


@pytest.mark.asyncio
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
    assert unit.status == "failed"
    # The authoritative content committed in phase 1, before enrichment
    # ran, must still be intact.
    assert unit.authoritative_content == "Authoritative content must survive a bug"

    monkeypatch.undo()
    recovered = await retry_enrichment(db, fake_llm, unit)
    assert recovered.status == "ready"
    assert recovered.authoritative_content == "Authoritative content must survive a bug"


@pytest.mark.asyncio
async def test_capture_and_units_survive_a_fresh_session_lookup(db, fake_llm):
    capture = await create_capture(db, fake_llm, text="Durable across a restart", uploads=[])
    unit_id = capture.units[0].id

    fetched = db.get(MemoryUnit, unit_id)
    assert fetched is not None
    assert fetched.authoritative_content == "Durable across a restart"
    assert fetched.status == "ready"
