import pytest

from app.ingestion.pipeline import correct_unit, create_capture, delete_capture
from app.models import Capture
from tests.fakes import FakeLLMProvider


@pytest.mark.asyncio
async def test_delete_capture_removes_row_and_file(db, fake_llm, tmp_path, monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "uploads_dir", tmp_path)
    capture = await create_capture(db, fake_llm, text="To be deleted", uploads=[("note.txt", b"content")])
    capture_dir = tmp_path / str(capture.id)
    assert capture_dir.exists()

    delete_capture(db, capture)

    assert db.get(Capture, capture.id) is None
    assert not capture_dir.exists()


@pytest.mark.asyncio
async def test_correction_updates_semantic_representation(db):
    llm = FakeLLMProvider(
        enrichment_response=[
            {
                "meaning": "The user's gym locker combination is 42.",
                "aliases": ["gym locker combination"],
                "people": [],
                "organizations": [],
                "locations": [],
                "topics": [],
                "definitions": [],
                "relations": [],
                "event_date": "",
            },
            {
                "meaning": "The user's bike lock PIN is 42.",
                "aliases": ["bike lock PIN"],
                "people": [],
                "organizations": [],
                "locations": [],
                "topics": [],
                "definitions": [],
                "relations": [],
                "event_date": "",
            },
        ]
    )
    capture = await create_capture(db, llm, text="My code is 42.", uploads=[])
    unit = capture.units[0]
    assert unit.semantic["aliases"] == ["gym locker combination"]

    corrected = await correct_unit(db, llm, unit, "Actually it's my bike lock PIN, not a gym locker.")
    assert corrected.semantic["aliases"] == ["bike lock PIN"]
    assert corrected.clarifications[-1]["source"] == "user_correction"
    assert corrected.clarifications[-1]["answer"] == "Actually it's my bike lock PIN, not a gym locker."
    # A correction doesn't consume the system-clarification budget.
    assert corrected.clarification_round == 0


@pytest.mark.asyncio
async def test_correction_rejects_empty_text(db, fake_llm):
    capture = await create_capture(db, fake_llm, text="Something.", uploads=[])
    from app.ingestion.pipeline import IngestionError

    with pytest.raises(IngestionError):
        await correct_unit(db, fake_llm, capture.units[0], "   ")
