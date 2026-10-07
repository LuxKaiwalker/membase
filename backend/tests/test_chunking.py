import pytest

from app.ingestion.chunking import chunk_attachment
from app.ingestion.pipeline import add_attachment, create_capture
from app.models import AttachmentChunk
from app.retrieval.pipeline import hybrid_search_chunks
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish


def test_chunk_attachment_paginates_pdfs():
    pages = ["short page one", "x" * 3000]
    chunks = chunk_attachment(pages, None)
    assert chunks[0].page_number == 1
    assert chunks[0].text == "short page one"
    # The long second page must be split, and every piece still tagged page 2.
    page_two_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page_two_chunks) > 1
    assert all(c.page_number == 2 for c in page_two_chunks)


def test_chunk_attachment_splits_plain_text_without_pages():
    chunks = chunk_attachment(None, "y" * 3000)
    assert len(chunks) > 1
    assert all(c.page_number is None for c in chunks)


def test_chunk_attachment_empty_input_yields_no_chunks():
    assert chunk_attachment(None, None) == []
    assert chunk_attachment(None, "   ") == []


@pytest.mark.asyncio
async def test_chunks_are_created_and_embedded_on_ingestion(db, fake_llm):
    capture = await create_capture(
        db, fake_llm, text="Notes attached.", uploads=[("notes.txt", b"the secret launch date is March 3rd")]
    )
    chunks = db.query(AttachmentChunk).filter(AttachmentChunk.capture_id == capture.id).all()
    assert len(chunks) == 1
    assert chunks[0].embedding is not None
    assert "launch date" in chunks[0].text


@pytest.mark.asyncio
async def test_chunks_are_searchable_via_hybrid_search_chunks(db, fake_llm):
    # Chunks only clear the same status gate memory_units do (see
    # test_chunk_search_is_gated_until_a_sibling_unit_is_ready below) --
    # drive the unit to "ready" for this to be searchable at all.
    capture = await create_capture_and_finish(
        db, fake_llm, text="Notes attached.", uploads=[("notes.txt", b"the secret launch code is 9981")]
    )
    scored = await hybrid_search_chunks(db, fake_llm, "launch code")
    assert scored
    assert scored[0].chunk.attachment.capture_id == capture.id


@pytest.mark.asyncio
async def test_chunk_search_is_gated_until_a_sibling_unit_is_ready(db):
    """Attachment chunks have no status column of their own -- they're
    indexed as soon as the attachment is uploaded, before segmentation or
    clarification even runs (see _index_attachment_chunks). Without an
    explicit gate, a chunk stayed searchable even while its only sibling
    MemoryUnit was still awaiting_clarification -- content the user
    hasn't finished confirming leaking into retrieval. hybrid_search_chunks
    must require at least one ready sibling unit in the same capture."""
    asks_question = {
        "questions": [{"question": "What is a Zorblex code?", "field": "Zorblex code", "relevance": "high"}]
    }
    llm = FakeLLMProvider(question_response=asks_question)
    capture = await create_capture(
        db, llm, text="My Zorblex code is 42.", uploads=[("notes.txt", b"the secret launch code is 9981")]
    )
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"

    scored = await hybrid_search_chunks(db, llm, "launch code")
    assert scored == [], "a chunk must not be searchable while every sibling unit is still gated"


@pytest.mark.asyncio
async def test_posthoc_attachment_chunks_are_also_indexed(db, fake_llm):
    capture = await create_capture(db, fake_llm, text="Meeting notes.", uploads=[])
    capture = await add_attachment(db, fake_llm, capture, "later.txt", b"the budget is 50000 dollars")
    chunks = db.query(AttachmentChunk).filter(AttachmentChunk.capture_id == capture.id).all()
    assert len(chunks) == 1
    assert chunks[0].embedding is not None
