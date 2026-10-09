import pytest
from fastapi import HTTPException

from app.api.recall import save_as_memory
from app.models import Capture
from app.retrieval.session import start_session
from app.schemas import SaveAsMemoryRequest

# Streamlining phase (see docs/DECISIONS.md "Save as memory"): recall-
# session state (cues, hypotheses, the final answer) is never promoted
# automatically into a MemoryUnit -- this is the one explicit action that
# lets a user promote something *they* said during recall into a new
# memory, and it must go through the ordinary create_capture() pipeline,
# never bypass it.


@pytest.mark.asyncio
async def test_saving_a_users_own_message_creates_a_normal_capture(db, fake_llm):
    session = await start_session(db, fake_llm, "something I can't quite recall")

    capture = await save_as_memory(
        session.id, SaveAsMemoryRequest(text="something I can't quite recall"), background_tasks=_NoopTasks(), db=db, llm=fake_llm
    )

    assert isinstance(capture, Capture)
    assert capture.raw_text == "something I can't quite recall"
    assert db.query(Capture).filter(Capture.id == capture.id).count() == 1


@pytest.mark.asyncio
async def test_saving_a_cue_the_user_supplied_mid_session_creates_a_normal_capture(db, fake_llm):
    """Session-round mechanics (how a cue ends up on session.cues) are
    already covered by test_recall_session.py -- this only needs a cue
    to exist, so it's set directly rather than re-exercising that flow."""
    session = await start_session(db, fake_llm, "who did I meet for coffee")
    session.cues = [
        {"text": "It was Markus, not Anna", "confidence": "certain", "polarity": "positive", "round": 1, "created_at": "2026-01-01T00:00:00Z"}
    ]
    db.commit()
    db.refresh(session)

    capture = await save_as_memory(
        session.id, SaveAsMemoryRequest(text="It was Markus, not Anna"), background_tasks=_NoopTasks(), db=db, llm=fake_llm
    )

    assert capture.raw_text == "It was Markus, not Anna"


@pytest.mark.asyncio
async def test_arbitrary_text_not_said_in_the_session_is_rejected(db, fake_llm):
    session = await start_session(db, fake_llm, "something I can't quite recall")

    with pytest.raises(HTTPException) as exc_info:
        await save_as_memory(
            session.id, SaveAsMemoryRequest(text="this was never said in this session"), background_tasks=_NoopTasks(), db=db, llm=fake_llm
        )
    assert exc_info.value.status_code == 400


class _NoopTasks:
    def add_task(self, *args, **kwargs):
        pass
