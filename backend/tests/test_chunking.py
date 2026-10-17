import pytest

from app.ingestion.chunking import chunk_attachment
from app.ingestion.pipeline import (
    IngestionError,
    add_attachment,
    add_attachments,
    create_capture,
    finish_attachment_chunking,
    process_capture_fully,
)
from app.models import Attachment, AttachmentChunk
from app.retrieval.pipeline import hybrid_search, hybrid_search_chunks
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


@pytest.fixture
def no_min_chunk_chars(monkeypatch):
    """Chunk-level indexing is lazy/conditional now (see
    docs/DECISIONS.md "Lazy attachment chunking") -- short attachments
    below membase_min_chars_for_chunking are deliberately never chunked.
    Tests in this file that exercise chunk creation/search mechanics
    (not the threshold itself) use short fixtures predating that
    threshold, so they disable it rather than rewriting every fixture."""
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "membase_min_chars_for_chunking", 0)


async def test_chunks_are_created_and_embedded_on_ingestion(db, fake_llm, no_min_chunk_chars):
    """Chunk-indexing is backgrounded (see docs/DECISIONS.md "offloading
    to background processing") -- create_capture_and_finish() drives it
    to completion, the test equivalent of "wait for that background
    task"."""
    capture = await create_capture_and_finish(
        db, fake_llm, text="Notes attached.", uploads=[("notes.txt", b"the secret launch date is March 3rd")]
    )
    chunks = db.query(AttachmentChunk).filter(AttachmentChunk.capture_id == capture.id).all()
    assert len(chunks) == 1
    assert chunks[0].embedding is not None
    assert "launch date" in chunks[0].text


async def test_chunks_are_searchable_via_hybrid_search_chunks(db, fake_llm, no_min_chunk_chars):
    # Chunks only clear the same status gate memory_units do (see
    # test_chunk_search_is_gated_until_a_sibling_unit_is_ready below) --
    # drive the unit to "ready" for this to be searchable at all.
    capture = await create_capture_and_finish(
        db, fake_llm, text="Notes attached.", uploads=[("notes.txt", b"the secret launch code is 9981")]
    )
    scored = await hybrid_search_chunks(db, fake_llm, "launch code")
    assert scored
    assert scored[0].chunk.attachment.capture_id == capture.id


async def test_chunk_search_is_gated_until_a_sibling_unit_is_ready(db, no_min_chunk_chars):
    """Attachment chunks have no status column of their own -- they're
    indexed as soon as the attachment is uploaded, before segmentation or
    clarification even runs (see _index_attachment_chunks). Without an
    explicit gate, a chunk stayed searchable even while its only sibling
    MemoryUnit was still awaiting_clarification -- content the user
    hasn't finished confirming leaking into retrieval. hybrid_search_chunks
    must require at least one ready sibling unit in the same capture."""
    asks_question = {"clarification_question": "What is a Zorblex code?", "clarification_field": "Zorblex code"}
    llm = FakeLLMProvider(clarification_response=asks_question)
    capture = await create_capture_and_finish(
        db, llm, text="My Zorblex code is 42.", uploads=[("notes.txt", b"the secret launch code is 9981")]
    )
    unit = capture.units[0]
    assert unit.status == "awaiting_clarification"
    # Chunking is backgrounded (see docs/DECISIONS.md) and has no
    # dependency on the unit's clarification state -- drive it to
    # completion explicitly so this test genuinely exercises the gate
    # (a chunk that exists but is excluded), not just "no chunks were
    # ever created, so naturally nothing is found."
    for attachment in capture.attachments:
        await finish_attachment_chunking(attachment.id, llm=llm)
    assert db.query(AttachmentChunk).filter(AttachmentChunk.capture_id == capture.id).count() > 0

    scored = await hybrid_search_chunks(db, llm, "launch code")
    assert scored == [], "a chunk must not be searchable while every sibling unit is still gated"


async def test_posthoc_attachment_chunks_are_also_indexed(db, fake_llm, no_min_chunk_chars):
    capture = await create_capture(db, text="Meeting notes.", uploads=[])
    capture = await add_attachment(db, fake_llm, capture, "later.txt", b"the budget is 50000 dollars")
    chunks = db.query(AttachmentChunk).filter(AttachmentChunk.capture_id == capture.id).all()
    assert len(chunks) == 1
    assert chunks[0].embedding is not None


async def test_multiple_files_can_be_attached_in_one_call(db, fake_llm):
    capture = await create_capture_and_finish(db, fake_llm, text="Meeting notes.", uploads=[])

    capture = await add_attachments(
        db, fake_llm, capture,
        [("a.txt", b"notes about the budget"), ("b.txt", b"notes about the schedule"), ("c.txt", b"notes about staffing")],
    )

    assert {a.original_filename for a in capture.attachments} == {"a.txt", "b.txt", "c.txt"}
    assert db.query(Attachment).filter(Attachment.capture_id == capture.id).count() == 3


async def test_multiple_files_only_re_embed_sibling_units_once(db, fake_llm):
    """Attaching N files must not re-embed every ready sibling unit N
    times -- see docs/DECISIONS.md "Multi-file posthoc attachment":
    sibling re-embedding happens once for the whole batch, not once per
    file."""
    capture = await create_capture_and_finish(db, fake_llm, text="Meeting notes.", uploads=[])
    embeds_before = len(fake_llm.embed_calls)

    await add_attachments(
        db, fake_llm, capture,
        [("a.txt", b"notes about the budget"), ("b.txt", b"notes about the schedule"), ("c.txt", b"notes about staffing")],
    )

    # One re-embed of the sibling unit's retrieval_text for the whole
    # batch, not three.
    assert len(fake_llm.embed_calls) - embeds_before == 1


async def test_one_bad_file_in_a_batch_attaches_none_of_them(db, fake_llm):
    """Validation is atomic across the batch (see docs/DECISIONS.md) --
    a rejected file must not leave the other, otherwise-valid files of
    the same request attached."""
    capture = await create_capture(db, text="Meeting notes.", uploads=[])

    with pytest.raises(IngestionError):
        await add_attachments(
            db, fake_llm, capture,
            [("good.txt", b"a perfectly fine file"), ("bad.txt", b"")],  # empty file is rejected
        )

    assert db.query(Attachment).filter(Attachment.capture_id == capture.id).count() == 0


async def test_short_attachment_is_not_chunked_but_still_searchable(db, fake_llm):
    """Default threshold, no monkeypatch: a short attachment gets no
    AttachmentChunk rows at all -- the parent unit's full-text/embedding
    already covers it (see docs/DECISIONS.md "Lazy attachment
    chunking")."""
    capture = await create_capture_and_finish(
        db, fake_llm, text="Meeting notes.", uploads=[("notes.txt", b"the budget is 50000 dollars")]
    )
    chunks = db.query(AttachmentChunk).filter(AttachmentChunk.capture_id == capture.id).all()
    assert chunks == []

    scored = await hybrid_search(db, fake_llm, "50000 dollars")
    assert scored and scored[0].unit.capture_id == capture.id


async def test_uploaded_image_is_captioned_and_becomes_searchable(db):
    """No OCR/vision model exists separately -- the generation model's
    native multimodal understanding is what makes an uploaded image
    searchable at all. Captioning is a fully independent background task
    now (see docs/DECISIONS.md "Decouple image captioning"); a short
    caption like this one stays under the chunking threshold, so it's
    the parent unit's search, not chunk search, that must find it."""
    llm = FakeLLMProvider(image_caption_response="a red square on a white background")
    capture = await create_capture_and_finish(
        db, llm, text="Photo from the trip.", uploads=[("photo.png", _png_bytes())]
    )

    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.extraction_status == "ok"
    assert attachment.extracted_text == "a red square on a white background"
    assert len(llm.images_received) == 1
    assert llm.images_received[0] == [_png_bytes()]

    scored = await hybrid_search(db, llm, "red square")
    assert scored and scored[0].unit.capture_id == capture.id


async def test_image_captioning_failure_leaves_image_stored_but_uncaptioned(db):
    """Best-effort, same tolerance as every other derived LLM step in
    this pipeline: create_capture() (Phase 0) never even calls the LLM
    for the image, and a subsequent captioning failure inside
    process_capture_fully() must not block the rest of the capture's
    formation -- the unit still reaches "ready", just without this
    image's content folded in."""
    llm = FakeLLMProvider(fail="unavailable")
    capture = await create_capture(db, text="Photo.", uploads=[("photo.png", _png_bytes())])
    attachment = db.query(Attachment).filter(Attachment.capture_id == capture.id).one()
    assert attachment.extraction_status == "unsupported"
    assert attachment.extracted_text is None

    await process_capture_fully(capture.id, llm=llm)
    db.refresh(attachment)
    assert attachment.extraction_status == "unsupported"
    assert attachment.extracted_text is None
    db.refresh(capture)
    assert capture.units[0].status == "ready"


async def test_image_caption_response_accepts_a_list_for_per_image_captions(db):
    """Regression: unlike every other FakeLLMProvider *_response param,
    image_caption_response previously only wrapped a plain string into
    the {"caption": ...} shape _ResponseQueue needs -- passing a list
    (the documented convention every other param supports, for a
    different response per call) left raw strings in the queue, and
    next()["caption"] would crash with a TypeError on a plain string.
    Each image is captioned in attachment order (see
    process_capture_fully), matching upload order here."""
    llm = FakeLLMProvider(image_caption_response=["first caption", "second caption"])
    capture = await create_capture_and_finish(
        db, llm, text="Two photos.",
        uploads=[("one.png", _png_bytes()), ("two.png", _png_bytes())],
    )

    attachments = db.query(Attachment).filter(Attachment.capture_id == capture.id).order_by(Attachment.original_filename).all()
    captions = {a.original_filename: a.extracted_text for a in attachments}
    assert captions == {"one.png": "first caption", "two.png": "second caption"}
