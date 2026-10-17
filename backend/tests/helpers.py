"""Shared test helpers for driving the now-fully-backgrounded ingestion
pipeline.

process_capture_fully() is the ONE function that does ALL of a
capture's LLM-dependent work in production (image captioning + structured
extraction, chunking, segmentation, clarification, enrichment,
embedding, and the membase structurer's deeper pass) as a single
background task (see app/ingestion/pipeline.py and docs/DECISIONS.md
"Restore the multi-call ingestion pipeline"). There's no BackgroundTasks
machinery outside the HTTP layer, so create_capture_and_finish() runs
create_capture() (Phase 0 only) and then awaits process_capture_fully()
directly -- the test equivalent of "wait for the background task to
complete".

Tests that only care about Phase-0 state (rejection, raw persistence,
`capture.units == []` immediately after creation) can keep calling
create_capture() directly; no need to pull this in.
"""

from app.ingestion.pipeline import (
    create_capture,
    finish_attachment_chunking,
    finish_unit_processing,
    process_capture_fully,
    resolve_clarification,
    should_chunk,
)


async def _finish_processing(db, llm, *units) -> None:
    for unit in units:
        if unit.status == "processing":
            await finish_unit_processing(unit.id, llm=llm)
    if units:
        db.expire_all()  # pick up what finish_unit_processing's own session committed


async def create_capture_and_finish(db, llm, *, text, uploads=None, explicit_metadata=None, allow_clarification=True):
    """create_capture() (Phase 0: raw text + attachments, zero LLM
    calls) + process_capture_fully() (everything else) + a non-image
    attachment's independently-scheduled chunking (see
    app/api/captures.py's create() -- process_capture_fully() only
    chunks an IMAGE attachment internally, since that's the one case
    whose chunking decision depends on state it just produced; a
    non-image attachment's chunking is scheduled separately, right
    alongside capture creation, in production too), synchronously -- the
    direct replacement for the old pre-restore
    `create_capture(db, llm, text=X, uploads=Y)` call pattern, which
    used to do all of this inline.
    """
    capture = await create_capture(
        db, text=text, uploads=uploads or [], explicit_metadata=explicit_metadata, allow_clarification=allow_clarification
    )
    await process_capture_fully(capture.id, llm=llm)
    db.expire_all()
    db.refresh(capture)
    for attachment in capture.attachments:
        if not attachment.content_type.startswith("image/") and should_chunk(attachment.extracted_text):
            await finish_attachment_chunking(attachment.id, llm=llm)
    db.expire_all()
    db.refresh(capture)
    return capture


async def create_capture_and_finish_unstructured(db, llm, *, text, uploads=None):
    """Like create_capture_and_finish(), but stops right after the fast
    pipeline (embedding) -- before the membase structurer's automatic
    chaining -- so a test can drive structuring itself, with a
    separately controlled LLM response (see tests/test_structurer.py)."""
    capture = await create_capture(db, text=text, uploads=uploads or [])
    await process_capture_fully(capture.id, llm=llm, auto_structure=False)
    db.expire_all()
    db.refresh(capture)
    return capture


async def resolve_clarification_and_finish(db, llm, unit, answer: str):
    """resolve_clarification() + synchronously finish background
    enrichment if that resolved the unit's last pending question."""
    updated = await resolve_clarification(db, llm, unit, answer)
    await _finish_processing(db, llm, updated)
    db.refresh(updated)
    return updated
