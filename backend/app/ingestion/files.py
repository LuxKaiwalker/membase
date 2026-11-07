import io
import logging
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import magic
from pypdf import PdfReader
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


class UnsupportedFileError(ValueError):
    """Raised for files we refuse to store at all (wrong type, too large,
    corrupt). Distinct from extraction failures, which we tolerate."""


# MVP file support, per spec section 7. Anything else is rejected outright
# rather than silently accepted-but-ignored.
#
# The audio entries carry BOTH the canonical IANA name and the spelling
# libmagic actually emits, because they differ and libmagic is what this
# module sniffs with: measured on this machine's libmagic, a PCM WAV comes
# back as "audio/x-wav" (not "audio/wav") and an AAC/MP4 file as
# "audio/x-m4a" (not "audio/mp4"). Keying only on the canonical names would
# have rejected every real WAV upload. The canonical spellings are kept
# alongside so a libmagic version that does report them keeps working.
#
# "video/webm" is here on purpose: a WebM file carrying only an Opus audio
# track (exactly what a browser's MediaRecorder produces) is sniffed as
# "video/webm", and there is no magic-level way to tell it from a WebM with
# a video track. Accepting it is the only way "audio/webm" works at all;
# the cost is that a genuine video file is stored and downloadable rather
# than rejected. It is never transcribed either way -- see
# TRANSCRIBABLE_AUDIO_MIMES in app/ingestion/pipeline.py.
_ALLOWED_MIMES = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "text",
    "text/x-markdown": "text",
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/mpeg": "audio",
    "audio/mp4": "audio",
    "audio/x-m4a": "audio",
    "audio/ogg": "audio",
    "audio/flac": "audio",
    "audio/x-flac": "audio",
    "audio/webm": "audio",
    "video/webm": "audio",
}


@dataclass
class ProcessedFile:
    stored_path: str
    content_type: str
    size_bytes: int
    extracted_text: str | None
    extraction_status: str  # "ok" | "unsupported" | "failed"
    pages: list[str] | None = None  # per-page text for PDFs; None for everything else


def _detect_kind(data: bytes, filename: str) -> tuple[str, str]:
    mime = magic.from_buffer(data, mime=True)
    # `.md` files are frequently sniffed as text/plain; trust the extension
    # for markdown specifically since there's no reliable magic signature.
    if mime == "text/plain" and filename.lower().endswith(".md"):
        mime = "text/markdown"
    kind = _ALLOWED_MIMES.get(mime)
    if kind is None:
        raise UnsupportedFileError(f"Unsupported file type: {mime!r} ({filename})")
    return mime, kind


def extract_pdf_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _validate_image(data: bytes) -> None:
    with Image.open(io.BytesIO(data)) as img:
        img.verify()


def _validate_wav(data: bytes, filename: str) -> None:
    """Reject a corrupt/truncated WAV the same way _validate_image()
    rejects a corrupt image -- cleanly, at upload time, as a 400.

    Only WAV gets a structural check, and deliberately so: `wave` is in
    the standard library, while MP3/FLAC/OGG/M4A would each need a real
    decoder, and adding an audio-decoding dependency to reject the rare
    corrupt upload is not a trade worth making (a container we can't parse
    here simply stays stored-but-untranscribed, which is the same outcome
    the pipeline already degrades to).

    The two-part check mirrors the truncated-PNG bug this module already
    carries a comment about: `wave.open()` alone raises for a non-RIFF
    file, a header-only file, or a mangled fmt chunk, but a WAV whose
    *body* was cut off still opens happily and only reveals itself by
    promising more audio in its header than the file actually contains --
    verified directly against a truncated WAV, which opened fine and
    reported 80016 frames with only 922 bytes of payload present. The
    size comparison is arithmetic on the declared frame count rather than
    a readframes() call, so a header claiming an absurd length can't make
    this try to allocate it.
    """
    try:
        with wave.open(io.BytesIO(data)) as w:
            declared_bytes = w.getnframes() * w.getsampwidth() * w.getnchannels()
    except (wave.Error, EOFError) as exc:
        raise UnsupportedFileError(f"File claims to be a WAV but isn't: {filename}") from exc
    if declared_bytes > len(data):
        raise UnsupportedFileError(f"WAV file is truncated or corrupt: {filename}")


def process_upload(filename: str, data: bytes, memory_id: uuid.UUID) -> ProcessedFile:
    """Validate, persist, and best-effort extract text from an uploaded file.

    Raises UnsupportedFileError for files we refuse to store. Extraction
    failures for an otherwise-valid file are *not* raised -- the original
    file is still stored and `extraction_status` records what happened, per
    the "don't lose the original" rule in the spec.
    """
    settings = get_settings()
    max_bytes = settings.membase_max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise UnsupportedFileError(
            f"File exceeds max upload size of {settings.membase_max_upload_mb}MB"
        )
    if not data:
        raise UnsupportedFileError("Empty file")

    mime, kind = _detect_kind(data, filename)

    if kind == "image":
        try:
            _validate_image(data)
        except OSError as exc:
            # Covers both an unrecognizable format (UnidentifiedImageError,
            # itself an OSError subclass) and a recognized-but-corrupt/
            # truncated file (a plain OSError, e.g. "Truncated File Read"
            # from a PNG cut off mid-stream) -- confirmed by direct testing
            # that PIL raises the latter as a bare OSError, not
            # UnidentifiedImageError, so catching only the subclass let a
            # truncated image 500 instead of cleanly 400ing.
            raise UnsupportedFileError(f"File claims to be an image but isn't: {filename}") from exc

    if kind == "audio" and mime in ("audio/wav", "audio/x-wav"):
        _validate_wav(data, filename)

    dest_dir = settings.uploads_dir / str(memory_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
    dest_path = dest_dir / safe_name
    dest_path.write_bytes(data)

    extracted_text: str | None = None
    pages: list[str] | None = None
    extraction_status = "unsupported"
    if kind == "pdf":
        try:
            pages = extract_pdf_pages(data)
            extracted_text = "\n".join(pages).strip()
            extraction_status = "ok" if extracted_text else "unsupported"
        except Exception:
            logger.exception("PDF text extraction failed for %s", filename)
            extraction_status = "failed"
    elif kind == "text":
        try:
            extracted_text = data.decode("utf-8")
            extraction_status = "ok"
        except UnicodeDecodeError:
            extraction_status = "failed"
    elif kind in ("image", "audio"):
        # Nothing to extract *locally* for either: an image's caption and an
        # audio file's transcript both come from the generation model, in the
        # background, well after this function has returned (see
        # _caption_one_image / _transcribe_one_audio in
        # app/ingestion/pipeline.py). "unsupported" is the marker those
        # passes look for, and the value that stays behind for good if the
        # model can't handle the file -- the original is on disk and
        # downloadable via GET /api/attachments/{id}/file either way.
        extraction_status = "unsupported"

    return ProcessedFile(
        stored_path=str(dest_path.relative_to(settings.uploads_dir)),
        content_type=mime,
        size_bytes=len(data),
        extracted_text=extracted_text,
        extraction_status=extraction_status,
        pages=pages,
    )
