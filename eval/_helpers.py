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

from sqlalchemy import text as sql_text

from app.ingestion.pipeline import finish_unit_processing

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
