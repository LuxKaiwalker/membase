import pytest

from app.ingestion.pipeline import correct_unit, create_capture, delete_capture
from app.models import Capture
from tests.fakes import FakeLLMProvider
from tests.helpers import create_capture_and_finish


async def test_delete_capture_removes_row_and_file(db, fake_llm, tmp_path, monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "uploads_dir", tmp_path)
    capture = await create_capture(db, fake_llm, text="To be deleted", uploads=[("note.txt", b"content")])
    capture_dir = tmp_path / str(capture.id)
    assert capture_dir.exists()

    delete_capture(db, capture)

    assert db.get(Capture, capture.id) is None
    assert not capture_dir.exists()


async def test_correction_updates_semantic_representation(db):
    # Two separate LLM instances (one for the initial analyze_capture
    # call, one for correct_unit's reanalyze_unit call) keep this test's
    # two distinct "before"/"after" semantics cleanly separated, rather
    # than relying on list-queue ordering within a single provider.
    gym_semantics = {
        "meaning": "The user's gym locker combination is 42.",
        "aliases": ["gym locker combination"],
        "people": [],
        "organizations": [],
        "locations": [],
        "topics": [],
        "definitions": [],
        "relations": [],
        "event_date": "",
    }
    bike_semantics = {
        "meaning": "The user's bike lock PIN is 42.",
        "aliases": ["bike lock PIN"],
        "people": [],
        "organizations": [],
        "locations": [],
        "topics": [],
        "definitions": [],
        "relations": [],
        "event_date": "",
    }

    llm = FakeLLMProvider(enrichment_response=gym_semantics)
    capture = await create_capture_and_finish(db, llm, text="My code is 42.")
    unit = capture.units[0]
    assert unit.semantic["aliases"] == ["gym locker combination"]

    llm2 = FakeLLMProvider(enrichment_response=bike_semantics)
    corrected = await correct_unit(db, llm2, unit, "Actually it's my bike lock PIN, not a gym locker.")
    assert corrected.semantic["aliases"] == ["bike lock PIN"]
    assert corrected.clarifications[-1]["source"] == "user_correction"
    assert corrected.clarifications[-1]["answer"] == "Actually it's my bike lock PIN, not a gym locker."
    # A correction doesn't consume the system-clarification budget.
    assert corrected.clarification_round == 0


async def test_correction_rejects_empty_text(db, fake_llm):
    capture = await create_capture(db, fake_llm, text="Something.", uploads=[])
    from app.ingestion.pipeline import IngestionError

    with pytest.raises(IngestionError):
        await correct_unit(db, fake_llm, capture.units[0], "   ")
