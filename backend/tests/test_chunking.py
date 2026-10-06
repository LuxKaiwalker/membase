import pytest

from app.ingestion.chunking import chunk_attachment
from app.ingestion.pipeline import add_attachment, create_capture
from app.models import AttachmentChunk
from app.retrieval.pipeline import hybrid_search_chunks


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
    capture = await create_capture(
        db, fake_llm, text="Notes attached.", uploads=[("notes.txt", b"the secret launch code is 9981")]
    )
    scored = await hybrid_search_chunks(db, fake_llm, "launch code")
    assert scored
    assert scored[0].chunk.attachment.capture_id == capture.id


@pytest.mark.asyncio
async def test_posthoc_attachment_chunks_are_also_indexed(db, fake_llm):
    capture = await create_capture(db, fake_llm, text="Meeting notes.", uploads=[])
    capture = await add_attachment(db, fake_llm, capture, "later.txt", b"the budget is 50000 dollars")
    chunks = db.query(AttachmentChunk).filter(AttachmentChunk.capture_id == capture.id).all()
    assert len(chunks) == 1
    assert chunks[0].embedding is not None
