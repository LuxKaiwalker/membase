import pytest

from app.ingestion.pipeline import IngestionError, create_memory
from app.models import Memory


@pytest.mark.asyncio
async def test_create_memory_persists(db, fake_llm):
    memory = await create_memory(db, fake_llm, caption="Test memory", uploads=[])
    assert memory.id is not None

    # Look it up fresh by id rather than relying on the in-memory object --
    # the closest a unit test gets to "survives a restart" without actually
    # restarting the process.
    fetched = db.get(Memory, memory.id)
    assert fetched is not None
    assert fetched.caption == "Test memory"


@pytest.mark.asyncio
async def test_create_memory_rejects_empty_caption(db, fake_llm):
    with pytest.raises(IngestionError):
        await create_memory(db, fake_llm, caption="   ", uploads=[])


@pytest.mark.asyncio
async def test_attachment_persists(db, fake_llm):
    memory = await create_memory(
        db, fake_llm, caption="Has an attachment", uploads=[("notes.txt", b"hello world")]
    )
    assert len(memory.attachments) == 1
    assert memory.attachments[0].extraction_status == "ok"
    assert memory.attachments[0].extracted_text == "hello world"
    assert "hello world" in memory.searchable_text


@pytest.mark.asyncio
async def test_create_memory_rejects_unsupported_file(db, fake_llm):
    with pytest.raises(IngestionError):
        await create_memory(
            db,
            fake_llm,
            caption="bad file",
            uploads=[("virus.exe", b"MZ\x90\x00\x03\x00\x00\x00")],
        )


@pytest.mark.asyncio
async def test_create_memory_rejects_empty_file(db, fake_llm):
    with pytest.raises(IngestionError):
        await create_memory(db, fake_llm, caption="empty file", uploads=[("empty.txt", b"")])
