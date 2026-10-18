"""Shared helpers for eval scripts: driving the now-backgrounded
ingestion pipeline to completion.

Mirrors backend/tests/helpers.py's create_capture_and_finish() -- eval
scripts call the real pipeline functions directly (create_capture(),
process_capture_fully(), resolve_clarification()), not through the HTTP
API, so there's no FastAPI BackgroundTasks machinery to run
process_capture_fully()/finish_unit_processing() automatically. Without
this, every capture an eval script creates stays in status="processing"
forever (and `capture.units` stays `[]`, since segmentation itself now
happens inside process_capture_fully() -- see docs/DECISIONS.md "Restore
the multi-call ingestion pipeline") and is invisible to hybrid_search()
(which only returns status="ready" units).
"""

from sqlalchemy import text as sql_text

from app.ingestion.pipeline import (
    create_capture,
    finish_attachment_chunking,
    finish_unit_processing,
    process_capture_fully,
    should_chunk,
)

# Every eval entry point TRUNCATEs its tables per case/run. That's safe
# against the isolated `membase_eval` database the scripts/*.sh wrappers
# point at (PGDATABASE/DATABASE_URL set before Python ever starts), but
# nothing stops someone from running e.g. `python3 eval/run_eval.py`
# directly inside `nix develop`, where DATABASE_URL defaults to the real
# dev database -- that would silently destroy real stored captures/
# memory units with no recovery path (.data/ isn't backed up anywhere
# else in this project). assert_eval_database() checks the live
# connection itself, not just an env var, so it can't be fooled by how
# the connection was established.
_EVAL_DATABASE_NAME = "membase_eval"


def assert_eval_database(db) -> None:
    name = db.execute(sql_text("SELECT current_database()")).scalar()
    if name != _EVAL_DATABASE_NAME:
        raise RuntimeError(
            f"Refusing to run: connected to database {name!r}, expected "
            f"{_EVAL_DATABASE_NAME!r}. Run this via its scripts/*.sh "
            "wrapper, not directly -- see docs/DECISIONS.md \"Eval database-"
            "isolation guard\" for why."
        )


async def create_capture_and_finish(db, llm, *, text, uploads=None, explicit_metadata=None, allow_clarification=True):
    """create_capture() (Phase 0: raw text + attachments, zero LLM calls)
    + process_capture_fully() (segmentation, clarification, enrichment,
    embedding) + a non-image attachment's independently-scheduled
    chunking, synchronously -- the eval-script equivalent of "wait for
    the background task to complete". Same pattern as
    backend/tests/helpers.py's create_capture_and_finish(); duplicated
    rather than imported since eval/ and backend/tests/ aren't on a
    shared import path.
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


async def finish_processing(db, llm, *captures) -> None:
    """For each given Capture, run finish_unit_processing() for every
    unit still in status="processing" -- used after resolve_clarification()
    answers a pending question, which can put a unit back into
    status="processing" until enrichment finishes. Captures returned by
    create_capture_and_finish() are already fully processed, so calling
    this on them again is a harmless no-op (no unit will be
    status="processing")."""
    for capture in captures:
        for unit in capture.units:
            if unit.status == "processing":
                await finish_unit_processing(unit.id, llm=llm)
    if captures:
        db.expire_all()
