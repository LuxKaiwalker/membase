import io
import logging
import uuid
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
_ALLOWED_MIMES = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "text",
    "text/x-markdown": "text",
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
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


def _extract_pdf_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _validate_image(data: bytes) -> None:
    with Image.open(io.BytesIO(data)) as img:
        img.verify()


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
            pages = _extract_pdf_pages(data)
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
    elif kind == "image":
        # No local vision/OCR model is wired in for the MVP -- documented
        # limitation (spec section 7 explicitly allows this).
        extraction_status = "unsupported"

    return ProcessedFile(
        stored_path=str(dest_path.relative_to(settings.uploads_dir)),
        content_type=mime,
        size_bytes=len(data),
        extracted_text=extracted_text,
        extraction_status=extraction_status,
        pages=pages,
    )
