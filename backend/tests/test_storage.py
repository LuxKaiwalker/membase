import pytest

from app.ingestion.pipeline import IngestionError, create_capture
from app.models import Capture
from tests.helpers import create_capture_and_finish


async def test_create_capture_persists(db, fake_llm):
    capture = await create_capture_and_finish(db, fake_llm, text="Test memory")
    assert capture.id is not None
    assert len(capture.units) == 1
    assert capture.units[0].authoritative_content == "Test memory"

    # Look it up fresh by id rather than relying on the in-memory object --
    # the closest a unit test gets to "survives a restart" without actually
    # restarting the process.
    fetched = db.get(Capture, capture.id)
    assert fetched is not None
    assert fetched.raw_text == "Test memory"
    assert fetched.units[0].status == "ready"


async def test_create_capture_rejects_empty_text(db, fake_llm):
    with pytest.raises(IngestionError):
        await create_capture(db, fake_llm, text="   ", uploads=[])


async def test_attachment_persists_on_capture(db, fake_llm):
    capture = await create_capture_and_finish(
        db, fake_llm, text="Has an attachment", uploads=[("notes.txt", b"hello world")]
    )
    assert len(capture.attachments) == 1
    assert capture.attachments[0].extraction_status == "ok"
    assert capture.attachments[0].extracted_text == "hello world"
    assert "hello world" in capture.units[0].searchable_text


async def test_create_capture_rejects_unsupported_file(db, fake_llm):
    with pytest.raises(IngestionError):
        await create_capture(
            db,
            fake_llm,
            text="bad file",
            uploads=[("virus.exe", b"MZ\x90\x00\x03\x00\x00\x00")],
        )


async def test_create_capture_rejects_empty_file(db, fake_llm):
    with pytest.raises(IngestionError):
        await create_capture(db, fake_llm, text="empty file", uploads=[("empty.txt", b"")])
