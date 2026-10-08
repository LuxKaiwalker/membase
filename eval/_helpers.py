"""Shared helper for eval scripts: driving the now-backgrounded
enrichment step to completion.

Mirrors backend/tests/helpers.py's create_capture_and_finish() -- eval
scripts call the real pipeline functions directly (create_capture(),
resolve_clarification()), not through the HTTP API, so there's no
FastAPI BackgroundTasks machinery to run finish_unit_processing()
automatically. Without this, every capture an eval script creates
stays in status="processing" forever and is invisible to hybrid_search()
(which only returns status="ready" units) -- see docs/DECISIONS.md.
"""

from app.ingestion.pipeline import finish_unit_processing


async def finish_processing(db, llm, *captures) -> None:
    """For each given Capture, run finish_unit_processing() for every
    unit still in status="processing" (i.e. not waiting on a pending
    clarification question) -- the eval-script equivalent of "wait for
    the background task to complete"."""
    for capture in captures:
        for unit in capture.units:
            if unit.status == "processing":
                await finish_unit_processing(unit.id, llm=llm)
    if captures:
        db.expire_all()
