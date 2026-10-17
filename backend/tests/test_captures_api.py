
from app.api.captures import create
from tests.fakes import FakeLLMProvider

# Streamlining phase (see docs/DECISIONS.md "Disable follow-up
# questions"): the API-level `disable_followup_questions` form field
# must thread through to create_capture()'s allow_clarification param.
# Calls the route function directly (FastAPI route functions are plain
# callables; db/llm are ordinary arguments when not going through a
# real request), same pattern as tests/test_recall_api.py.
#
# Units no longer exist when create() returns (see
# docs/DECISIONS.md "Restore the multi-call ingestion pipeline") -- all
# of a capture's analysis happens in the ONE background task create()
# schedules (process_capture_fully). _CollectingTasks records what was
# scheduled instead of dropping it (the old _NoopTasks pattern), so the
# test can run it explicitly afterward, the test equivalent of "wait for
# the background task".


class _CollectingTasks:
    def __init__(self):
        self.calls: list[tuple] = []

    def add_task(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))


async def _run_tasks(tasks, llm):
    for fn, args, kwargs in tasks.calls:
        kwargs = dict(kwargs)
        kwargs.setdefault("llm", llm)
        await fn(*args, **kwargs)


async def test_disable_followup_questions_form_field_suppresses_clarification(db):
    llm = FakeLLMProvider(clarification_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"})
    tasks = _CollectingTasks()

    capture = await create(
        tasks, text="My Flimvex PIN is 42.", files=[], metadata=None,
        disable_followup_questions=True, db=db,
    )
    await _run_tasks(tasks, llm)
    db.refresh(capture)

    assert capture.units[0].status != "awaiting_clarification"


async def test_followup_questions_enabled_by_default_via_api(db):
    llm = FakeLLMProvider(clarification_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"})
    tasks = _CollectingTasks()

    capture = await create(
        tasks, text="My Flimvex PIN is 42.", files=[], metadata=None,
        disable_followup_questions=False, db=db,
    )
    await _run_tasks(tasks, llm)
    db.refresh(capture)

    assert capture.units[0].status == "awaiting_clarification"
