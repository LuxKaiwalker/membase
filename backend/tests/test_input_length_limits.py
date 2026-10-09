import pytest

from app.config import get_settings
from app.ingestion.pipeline import IngestionError, correct_unit, create_capture, resolve_clarification
from app.retrieval.session import RecallSessionError, continue_session, start_session
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish

# Deployment-readiness audit finding (see docs/DEPLOYMENT.md): unlike
# file uploads, every free-text input (capture text, corrections,
# clarification answers, recall queries/messages) had no server-side
# size limit at all -- an unauthenticated caller could post an
# arbitrarily large body. These tests lock in the fix.


@pytest.mark.asyncio
async def test_capture_text_over_limit_is_rejected(db, fake_llm):
    max_chars = get_settings().membase_max_text_chars
    with pytest.raises(IngestionError):
        await create_capture(db, fake_llm, text="x" * (max_chars + 1), uploads=[])


@pytest.mark.asyncio
async def test_capture_text_at_limit_is_accepted(db, fake_llm):
    max_chars = get_settings().membase_max_text_chars
    capture = await create_capture(db, fake_llm, text="x" * max_chars, uploads=[])
    assert capture.units


@pytest.mark.asyncio
async def test_correction_over_limit_is_rejected(db, fake_llm):
    capture = await create_capture_and_finish(db, fake_llm, text="A short memory.", uploads=[])
    unit = capture.units[0]
    max_chars = get_settings().membase_max_text_chars
    with pytest.raises(IngestionError):
        await correct_unit(db, fake_llm, unit, "x" * (max_chars + 1))


@pytest.mark.asyncio
async def test_clarification_answer_over_limit_is_rejected(db):
    llm = FakeLLMProvider(enrichment_response={"clarification_question": "What is Flimvex?", "clarification_field": "Flimvex"})
    capture = await create_capture(db, llm, text="My Flimvex PIN is 42.", uploads=[])
    unit = capture.units[0]
    max_chars = get_settings().membase_max_text_chars
    with pytest.raises(IngestionError):
        await resolve_clarification(db, llm, unit, "x" * (max_chars + 1))


@pytest.mark.asyncio
async def test_recall_query_over_limit_is_rejected(db, fake_llm):
    max_chars = get_settings().membase_max_text_chars
    with pytest.raises(RecallSessionError):
        await start_session(db, fake_llm, "x" * (max_chars + 1))


@pytest.mark.asyncio
async def test_recall_session_message_over_limit_is_rejected(db, fake_llm):
    session = await start_session(db, fake_llm, "something I can't quite recall")
    max_chars = get_settings().membase_max_text_chars
    with pytest.raises(RecallSessionError):
        await continue_session(db, fake_llm, session, "x" * (max_chars + 1))
