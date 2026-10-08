"""Shared test helpers for driving the now-backgrounded enrichment and
attachment-chunking steps.

Full enrichment and attachment chunk-indexing both run as scheduled
FastAPI BackgroundTasks in production (see docs/DECISIONS.md) -- a unit
reaches status="processing" the moment clarification is settled, and
only reaches "ready" once that background task runs; an attachment's
chunks only exist once its own background task runs. There's no
BackgroundTasks machinery outside the HTTP layer, so these helpers run
create_capture()/resolve_clarification() and then synchronously finish
whatever processing they scheduled -- the test equivalent of "wait for
the background tasks to complete".

Tests that only care about clarification-time state (an open question,
the gate, discard) and never need a unit to actually reach "ready" or
an attachment to actually have chunks can keep calling
create_capture()/resolve_clarification() directly; no need to pull
these in.
"""

from app.ingestion.pipeline import (
    create_capture,
    finish_attachment_chunking,
    finish_unit_processing,
    resolve_clarification,
)


async def _finish_processing(db, llm, *units) -> None:
    for unit in units:
        if unit.status == "processing":
            await finish_unit_processing(unit.id, llm=llm)
    if units:
        db.expire_all()  # pick up what finish_unit_processing's own session committed


async def _finish_chunking(db, llm, *attachments) -> None:
    for attachment in attachments:
        await finish_attachment_chunking(attachment.id, llm=llm)
    if attachments:
        db.expire_all()  # pick up what finish_attachment_chunking's own session committed


async def create_capture_and_finish(db, llm, *, text, uploads=None):
    """create_capture() + synchronously finish background enrichment for
    every unit that doesn't have a clarification question pending, and
    background chunk-indexing for every attachment."""
    capture = await create_capture(db, llm, text=text, uploads=uploads or [])
    await _finish_processing(db, llm, *capture.units)
    await _finish_chunking(db, llm, *capture.attachments)
    db.refresh(capture)
    return capture


async def resolve_clarification_and_finish(db, llm, unit, answer: str):
    """resolve_clarification() + synchronously finish background
    enrichment if that resolved the unit's last pending question."""
    updated = await resolve_clarification(db, llm, unit, answer)
    await _finish_processing(db, llm, updated)
    db.refresh(updated)
    return updated
