import pytest

from app.api.captures import create
from tests.fakes import FakeLLMProvider

# Streamlining phase (see docs/DECISIONS.md "Disable follow-up
# questions"): the API-level `disable_followup_questions` form field
# must thread through to create_capture()'s allow_clarification param.
# Calls the route function directly (FastAPI route functions are plain
# callables; db/llm are ordinary arguments when not going through a
# real request), same pattern as tests/test_recall_api.py.


class _NoopTasks:
    def add_task(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_disable_followup_questions_form_field_suppresses_clarification(db):
    llm = FakeLLMProvider(enrichment_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"})

    capture = await create(
        _NoopTasks(), text="My Flimvex PIN is 42.", files=[], metadata=None,
        disable_followup_questions=True, db=db, llm=llm,
    )

    assert capture.units[0].status != "awaiting_clarification"


@pytest.mark.asyncio
async def test_followup_questions_enabled_by_default_via_api(db):
    llm = FakeLLMProvider(enrichment_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"})

    capture = await create(
        _NoopTasks(), text="My Flimvex PIN is 42.", files=[], metadata=None,
        disable_followup_questions=False, db=db, llm=llm,
    )

    assert capture.units[0].status == "awaiting_clarification"
