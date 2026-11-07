import io
import uuid
import wave

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


def _wav_bytes(seconds: float = 0.1, rate: int = 16000) -> bytes:
    """A real, structurally valid PCM WAV, built with the standard
    library -- no ffmpeg or audio dependency needed in the test suite.
    libmagic sniffs this as "audio/x-wav" (not "audio/wav"), which is
    exactly why _ALLOWED_MIMES carries both spellings."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(rate * seconds))
    return buf.getvalue()


# Minimal container headers -- enough for libmagic to sniff the type, which
# is all process_upload() looks at for these (only WAV gets a structural
# check; see _validate_wav's docstring for why).
_MP3_BYTES = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 40
_FLAC_BYTES = b"fLaC\x00\x00\x00\x22" + b"\x00" * 100
_OGG_BYTES = (
    b"OggS\x00\x02" + b"\x00" * 20 + b"\x01\x13"
    + b"OpusHead\x01\x01\x00\x0f\x80\xbb\x00\x00\x00\x00\x00" + b"\x00" * 40
)


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


# --- audio uploads ------------------------------------------------------
# Transcription itself is the generation model's job and happens later, in
# the background (see tests/test_audio.py); everything here is about
# accepting, validating and never losing the original recording.


def test_process_upload_accepts_wav_audio(tmp_path):
    data = _wav_bytes()
    result = process_upload("memo.wav", data, uuid.uuid4())
    # libmagic reports the x- spelling; both are in _ALLOWED_MIMES.
    assert result.content_type == "audio/x-wav"
    # Nothing is extracted locally -- "unsupported" is the marker the
    # background transcription pass looks for, exactly as for an image.
    assert result.extraction_status == "unsupported"
    assert result.extracted_text is None
    # The original must survive byte-for-byte: it's the only copy.
    assert (tmp_path / result.stored_path).read_bytes() == data


@pytest.mark.parametrize(
    ("filename", "data", "expected_mime"),
    [
        ("memo.mp3", _MP3_BYTES, "audio/mpeg"),
        ("memo.flac", _FLAC_BYTES, "audio/flac"),
        ("memo.ogg", _OGG_BYTES, "audio/ogg"),
    ],
)
def test_process_upload_accepts_other_audio_containers(filename, data, expected_mime):
    """OGG is accepted and stored even though this Ollama version cannot
    decode it (see TRANSCRIBABLE_AUDIO_MIMES in app/ingestion/pipeline.py)
    -- refusing the upload would lose the recording, which the "don't lose
    the original" rule forbids."""
    result = process_upload(filename, data, uuid.uuid4())
    assert result.content_type == expected_mime
    assert result.extraction_status == "unsupported"


def test_process_upload_rejects_oversized_audio(monkeypatch):
    from app import config as config_module

    monkeypatch.setattr(config_module.get_settings(), "membase_max_upload_mb", 0)
    with pytest.raises(UnsupportedFileError):
        process_upload("memo.wav", _wav_bytes(), uuid.uuid4())


def test_process_upload_rejects_truncated_wav_cleanly():
    """The audio counterpart of the truncated-PNG bug above, and the
    reason _validate_wav() doesn't stop at wave.open(): a WAV whose body
    was cut off still opens happily and reports its ORIGINAL frame count,
    so only comparing the declared payload size against the real file
    length catches it. Without this it would be stored as a valid
    recording and only fail later, at transcription."""
    truncated = _wav_bytes()[:200]
    with pytest.raises(UnsupportedFileError):
        process_upload("broken.wav", truncated, uuid.uuid4())


def test_process_upload_rejects_wav_with_no_fmt_or_data_chunk():
    """A bare 44-byte RIFF/WAVE header with nothing after it still sniffs
    as audio/x-wav, so the mime gate alone lets it through -- wave.open()
    is what rejects it ("fmt chunk and/or data chunk missing")."""
    with pytest.raises(UnsupportedFileError):
        process_upload("headeronly.wav", _wav_bytes()[:44], uuid.uuid4())


def test_process_upload_rejects_wav_with_mangled_fmt_chunk():
    valid = _wav_bytes()
    with pytest.raises(UnsupportedFileError):
        process_upload("mangled.wav", valid[:12] + b"XXXX" + valid[16:], uuid.uuid4())


def test_process_upload_rejects_riff_garbage_at_the_mime_gate():
    """Not even audio: "RIFF" plus noise sniffs as
    application/octet-stream, so this is refused before _validate_wav()
    is ever reached."""
    with pytest.raises(UnsupportedFileError):
        process_upload("noise.wav", b"RIFF" + b"\x00" * 40 + b"garbage" * 20, uuid.uuid4())
