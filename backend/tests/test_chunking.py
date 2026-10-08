import pytest

from app.ingestion.chunking import chunk_attachment
from app.ingestion.pipeline import add_attachment, create_capture
from app.models import Attachment, AttachmentChunk
from app.retrieval.pipeline import hybrid_search_chunks
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish
from tests.test_files import _png_bytes


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


def test_chunk_attachment_max_chunks_bounds_pages():
    pages = [f"page {i}" for i in range(10)]
    chunks = chunk_attachment(pages, None, max_chunks=3)
    assert len(chunks) == 3


def test_chunk_attachment_max_chunks_bounds_plain_text():
    chunks = chunk_attachment(None, "z" * 10_000, max_chunks=2)
    assert len(chunks) == 2


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


@pytest.mark.asyncio
async def test_uploaded_image_is_captioned_and_becomes_searchable(db):
    """No OCR/vision model exists separately -- the generation model's
    native multimodal understanding (see docs/DECISIONS.md "Gemma 4
    migration") is what makes an uploaded image searchable at all."""
    llm = FakeLLMProvider(image_caption_response="a red square on a white background")
    capture = await create_capture_and_finish(
        db, llm, text="Photo from the trip.", uploads=[("photo.png", _png_bytes())]
    )

    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.extraction_status == "ok"
    assert attachment.extracted_text == "a red square on a white background"
    assert len(llm.images_received) == 1
    assert llm.images_received[0] == [_png_bytes()]

    scored = await hybrid_search_chunks(db, llm, "red square")
    assert scored and "red square" in scored[0].chunk.text


@pytest.mark.asyncio
async def test_image_captioning_failure_leaves_image_stored_but_uncaptioned(db):
    """Best-effort, same tolerance as every other derived LLM step in
    this pipeline: a captioning failure must not lose the original image
    or block the rest of ingestion."""
    llm = FakeLLMProvider(fail="unavailable")
    capture = await create_capture(db, llm, text="Photo.", uploads=[("photo.png", _png_bytes())])
    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.extraction_status == "unsupported"
    assert attachment.extracted_text is None


@pytest.mark.asyncio
async def test_image_captioning_is_capped_per_capture(db, monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "membase_max_images_captioned_per_capture", 1)
    llm = FakeLLMProvider(image_caption_response="a photo")
    capture = await create_capture(
        db,
        llm,
        text="Two photos.",
        uploads=[("one.png", _png_bytes()), ("two.png", _png_bytes())],
    )
    attachments = db.query(Attachment).filter(Attachment.capture_id == capture.id).order_by(Attachment.original_filename).all()
    statuses = {a.original_filename: a.extraction_status for a in attachments}
    assert statuses == {"one.png": "ok", "two.png": "unsupported"}
    assert len(llm.images_received) == 1


@pytest.mark.asyncio
async def test_image_caption_response_accepts_a_list_for_per_image_captions(db):
    """Regression: unlike every other FakeLLMProvider *_response param,
    image_caption_response previously only wrapped a plain string into
    the {"caption": ...} shape _ResponseQueue needs -- passing a list
    (the documented convention every other param supports, for a
    different response per call) left raw strings in the queue, and
    next()["caption"] would crash with a TypeError on a plain string."""
    llm = FakeLLMProvider(image_caption_response=["first caption", "second caption"])
    capture = await create_capture(
        db,
        llm,
        text="Two photos.",
        uploads=[("one.png", _png_bytes()), ("two.png", _png_bytes())],
    )
    attachments = db.query(Attachment).filter(Attachment.capture_id == capture.id).order_by(Attachment.original_filename).all()
    captions = {a.original_filename: a.extracted_text for a in attachments}
    assert captions == {"one.png": "first caption", "two.png": "second caption"}
