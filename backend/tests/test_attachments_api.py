import uuid

import pytest
from fastapi import HTTPException

from app.api.attachments import get_attachment_file
from app.models import Attachment, Capture

# app/api/attachments.py's GET /{attachment_id}/file is the one route in
# this app that serves raw file bytes back to a caller (see
# docs/DECISIONS.md "Attachment view/download") -- had zero test coverage
# before this file, found during a full-repo audit.


@pytest.fixture(autouse=True)
def _uploads_dir(tmp_path, monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "uploads_dir", tmp_path)
    return tmp_path


def _make_attachment(db, tmp_path, *, filename="note.txt", content=b"hello world", content_type="text/plain"):
    capture = Capture(raw_text="a capture with an attachment")
    db.add(capture)
    db.flush()

    # stored_path is always server-generated (see app/ingestion/files.py)
    # -- never derived from the user-supplied `filename`/original_filename,
    # so it's safe to write to disk even when a test's `filename` is
    # deliberately attacker-shaped (e.g. "../../etc/passwd").
    stored_path = f"{capture.id}/{uuid.uuid4().hex}_stored"
    (tmp_path / str(capture.id)).mkdir(parents=True, exist_ok=True)
    (tmp_path / stored_path).write_bytes(content)

    attachment = Attachment(
        capture_id=capture.id,
        original_filename=filename,
        stored_path=stored_path,
        content_type=content_type,
        size_bytes=len(content),
        extraction_status="skipped",
    )
    db.add(attachment)
    db.commit()
    return attachment


def test_get_attachment_file_returns_bytes_with_original_filename(db, tmp_path):
    attachment = _make_attachment(db, tmp_path, filename="note.txt", content=b"hello world")

    response = get_attachment_file(attachment.id, db=db)

    assert response.media_type == "text/plain"
    assert response.path == tmp_path / attachment.stored_path
    assert response.filename == "note.txt"


def test_get_attachment_file_strips_directory_components_from_filename(db, tmp_path):
    attachment = _make_attachment(db, tmp_path, filename="../../etc/passwd")

    response = get_attachment_file(attachment.id, db=db)

    # original_filename is user-supplied -- the response's filename must
    # never leak directory components, even though stored_path (used for
    # the actual on-disk lookup) is always server-generated and safe.
    assert response.filename == "passwd"


def test_get_attachment_file_404s_for_unknown_id(db):
    with pytest.raises(HTTPException) as exc_info:
        get_attachment_file(uuid.uuid4(), db=db)

    assert exc_info.value.status_code == 404


def test_get_attachment_file_404s_when_file_missing_on_disk(db, tmp_path):
    attachment = _make_attachment(db, tmp_path)
    (tmp_path / attachment.stored_path).unlink()

    with pytest.raises(HTTPException) as exc_info:
        get_attachment_file(attachment.id, db=db)

    assert exc_info.value.status_code == 404
