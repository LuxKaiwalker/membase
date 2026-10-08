import io
import uuid

import pytest
from PIL import Image

from app.ingestion.files import UnsupportedFileError, process_upload

# process_upload() had no dedicated tests at all before this file -- found
# during a full-repo audit while chasing down a real bug (see
# test_process_upload_rejects_truncated_image_cleanly below), added here
# since there's no obvious existing place for pure-filesystem-level
# ingestion tests like these.


@pytest.fixture(autouse=True)
def _uploads_dir(tmp_path, monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "uploads_dir", tmp_path)
    return tmp_path


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="PNG")
    return buf.getvalue()


def test_process_upload_rejects_unsupported_mime():
    with pytest.raises(UnsupportedFileError):
        process_upload("archive.zip", b"PK\x03\x04garbage", uuid.uuid4())


def test_process_upload_rejects_empty_file():
    with pytest.raises(UnsupportedFileError):
        process_upload("empty.txt", b"", uuid.uuid4())


def test_process_upload_rejects_oversized_file(monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "membase_max_upload_mb", 0)
    with pytest.raises(UnsupportedFileError):
        process_upload("note.txt", b"just a few bytes", uuid.uuid4())


def test_process_upload_accepts_valid_image():
    result = process_upload("photo.png", _png_bytes(), uuid.uuid4())
    assert result.content_type == "image/png"
    # No local vision/OCR model wired in for the MVP -- by design.
    assert result.extraction_status == "unsupported"
    assert result.extracted_text is None


def test_process_upload_rejects_truncated_image_cleanly():
    """Regression: a recognized-but-corrupt image (e.g. a PNG cut off
    mid-stream) makes PIL's img.verify() raise a bare OSError ("Truncated
    File Read"), not UnidentifiedImageError -- confirmed by direct
    testing. process_upload previously only caught UnidentifiedImageError,
    so this case propagated an unhandled OSError (-> 500) instead of the
    clean UnsupportedFileError (-> 400) every other rejected upload gets."""
    truncated = _png_bytes()[:-20]
    with pytest.raises(UnsupportedFileError):
        process_upload("broken.png", truncated, uuid.uuid4())


def test_process_upload_sniffs_markdown_by_extension():
    # text/markdown has no reliable magic signature and is frequently
    # sniffed as text/plain -- process_upload trusts the .md extension.
    result = process_upload("notes.md", b"# Heading\n\nSome text.", uuid.uuid4())
    assert result.content_type == "text/markdown"
    assert result.extraction_status == "ok"


def test_process_upload_strips_directory_components_from_filename(tmp_path):
    result = process_upload("../../etc/passwd.txt", b"not actually passwd", uuid.uuid4())
    stored = tmp_path / result.stored_path
    assert stored.exists()
    assert stored.is_relative_to(tmp_path)
