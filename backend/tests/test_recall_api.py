import pytest

from app.api.recall import recall
from app.models import RecallSession
from app.retrieval.pipeline import ABSTENTION_MESSAGE
from app.schemas import RecallRequest
from tests.helpers import create_capture_and_finish

# Streamlining phase (see docs/DECISIONS.md "RecallSession canonical"):
# the legacy one-shot POST /api/recall must not be a second, independent
# retrieval algorithm -- it runs a RecallSession's first automatic round
# exactly like POST /api/recall/sessions does, just reported in the flat
# legacy shape with no session id. These call the route function
# directly (FastAPI route functions are plain callables; db/llm are
# ordinary arguments when not going through a real request).


@pytest.mark.asyncio
async def test_recall_endpoint_answers_via_a_real_session(db, fake_llm):
    await create_capture_and_finish(db, fake_llm, text="My gym locker combination is 42.", uploads=[])

    response = await recall(RecallRequest(query="gym locker combination"), db=db, llm=fake_llm)

    assert response.abstained is False
    assert response.sources

    # Proof this went through the real session machinery, not a separate
    # code path: a session row was actually persisted.
    sessions = db.query(RecallSession).all()
    assert len(sessions) == 1
    assert sessions[0].status == "success"


@pytest.mark.asyncio
async def test_recall_endpoint_abstains_on_empty_store(db, fake_llm):
    response = await recall(RecallRequest(query="anything at all"), db=db, llm=fake_llm)

    assert response.abstained is True
    assert response.answer == ABSTENTION_MESSAGE
    assert response.sources == []
