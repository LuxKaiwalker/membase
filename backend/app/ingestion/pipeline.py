import logging

from sqlalchemy.orm import Session

from app.ingestion.files import UnsupportedFileError, process_upload
from app.ingestion.metadata import extract_metadata
from app.llm.base import LLMError, LLMProvider
from app.models import Attachment, Memory

logger = logging.getLogger(__name__)


class IngestionError(ValueError):
    """Input-validation failure -- caller should turn this into a 4xx."""


def _embedding_input(caption: str, summary: str, topics: list[str]) -> str:
    # Deliberately caption + derived summary/topics only, *not* raw
    # attachment text: keeps embedding input short and semantically dense.
    # Full attachment text still contributes to full-text search via
    # searchable_text. See docs/DECISIONS.md.
    parts = [caption]
    if summary:
        parts.append(summary)
    if topics:
        parts.append("Topics: " + ", ".join(topics))
    return "\n".join(parts)


def _derived_text(derived: dict) -> str:
    # Folds extracted metadata into the full-text index too, so an indirect
    # cue ("that person from ETH") matches even if "ETH" never appears
    # verbatim in the caption -- it only needs to appear in extracted
    # location/people/topics/summary.
    bits = []
    for key in ("people", "topics"):
        values = derived.get(key)
        if values:
            bits.append(", ".join(values))
    for key in ("location", "summary"):
        value = derived.get(key)
        if value:
            bits.append(value)
    return " / ".join(bits)


def _build_searchable_text(
    caption: str, attachment_text: str, clarification_text: str, derived: dict | None = None
) -> str:
    parts = [caption]
    if attachment_text:
        parts.append(attachment_text)
    if clarification_text:
        parts.append(clarification_text)
    if derived:
        derived_text = _derived_text(derived)
        if derived_text:
            parts.append(derived_text)
    return "\n\n".join(parts)


async def _enrich(llm: LLMProvider, memory: Memory, attachment_text: str) -> None:
    """Best-effort metadata + embedding enrichment. Never raises: a memory
    must persist with just its authoritative caption even if the local LLM
    is down (spec section 22, "LLM failure" requirements)."""
    metadata = await extract_metadata(llm, memory.caption, attachment_text)

    if metadata is None:
        memory.derived = {"enrichment_status": "unavailable"}
        summary, topics = "", []
    else:
        derived = {
            "people": metadata["people"],
            "location": metadata["location"],
            "topics": metadata["topics"],
            "event_date": metadata["event_date"],
            "summary": metadata["summary"],
            "enrichment_status": "ok",
        }
        memory.derived = derived
        summary, topics = metadata["summary"], metadata["topics"]
        if metadata["clarification_question"]:
            memory.pending_clarification = {
                "question": metadata["clarification_question"],
                "field": metadata["clarification_field"] or "clarification",
            }

    memory.searchable_text = _build_searchable_text(
        memory.caption, attachment_text, "", memory.derived
    )

    try:
        embedding = await llm.embed([_embedding_input(memory.caption, summary, topics)])
        memory.embedding = embedding[0]
        memory.embedding_model = llm.embedding_model_name
        memory.embedding_dim = llm.embedding_dim
    except LLMError:
        logger.warning("Embedding failed during ingestion; memory kept, vector search unavailable for it")


async def create_memory(
    db: Session,
    llm: LLMProvider,
    *,
    caption: str,
    uploads: list[tuple[str, bytes]],
) -> Memory:
    caption = caption.strip()
    if not caption:
        raise IngestionError("caption must not be empty")

    memory = Memory(caption=caption, derived={}, searchable_text=caption)
    db.add(memory)
    db.flush()  # assigns memory.id without committing, so attachment FKs/paths are valid

    attachment_texts: list[str] = []
    for filename, data in uploads:
        try:
            processed = process_upload(filename, data, memory.id)
        except UnsupportedFileError as exc:
            db.rollback()
            raise IngestionError(str(exc)) from exc
        db.add(
            Attachment(
                memory_id=memory.id,
                original_filename=filename,
                stored_path=processed.stored_path,
                content_type=processed.content_type,
                size_bytes=processed.size_bytes,
                extracted_text=processed.extracted_text,
                extraction_status=processed.extraction_status,
            )
        )
        if processed.extracted_text:
            attachment_texts.append(processed.extracted_text)

    await _enrich(llm, memory, "\n\n".join(attachment_texts))

    db.commit()
    db.refresh(memory)
    return memory


async def resolve_clarification(
    db: Session, llm: LLMProvider, memory: Memory, answer: str
) -> Memory:
    answer = answer.strip()
    if not answer:
        raise IngestionError("clarification answer must not be empty")

    pending = memory.pending_clarification or {}
    field = pending.get("field") or "clarification"

    derived = dict(memory.derived or {})
    clarifications = dict(derived.get("clarification_answers") or {})
    clarifications[field] = answer
    derived["clarification_answers"] = clarifications
    # A date clarification is common enough to special-case into event_date
    # directly, so it's usable as a structured filter rather than only free text.
    if "date" in field.lower():
        derived["event_date"] = answer
    memory.derived = derived
    memory.pending_clarification = None

    attachment_text = "\n\n".join(
        a.extracted_text for a in memory.attachments if a.extracted_text
    )
    clarification_text = f"{pending.get('question', '')} {answer}".strip()
    memory.searchable_text = _build_searchable_text(
        memory.caption, attachment_text, clarification_text, derived
    )

    try:
        embedding = await llm.embed(
            [_embedding_input(memory.caption, derived.get("summary", ""), derived.get("topics", []))]
        )
        memory.embedding = embedding[0]
        memory.embedding_model = llm.embedding_model_name
        memory.embedding_dim = llm.embedding_dim
    except LLMError:
        logger.warning("Re-embedding after clarification failed; previous embedding left in place")

    db.commit()
    db.refresh(memory)
    return memory


async def add_attachment(db: Session, memory: Memory, filename: str, data: bytes) -> Memory:
    """Attach a file to an already-persisted memory.

    Scope decision (see docs/TODO.md): this only extends the full-text
    index with the new attachment's extracted text. It does not re-run LLM
    metadata extraction or re-embed -- doing so for every post-hoc
    attachment add would mean an unbounded number of LLM calls over a
    memory's lifetime for an MVP feature that's secondary to the core
    create -> retrieve -> answer loop.
    """
    try:
        processed = process_upload(filename, data, memory.id)
    except UnsupportedFileError as exc:
        raise IngestionError(str(exc)) from exc

    db.add(
        Attachment(
            memory_id=memory.id,
            original_filename=filename,
            stored_path=processed.stored_path,
            content_type=processed.content_type,
            size_bytes=processed.size_bytes,
            extracted_text=processed.extracted_text,
            extraction_status=processed.extraction_status,
        )
    )
    db.flush()

    attachment_text = "\n\n".join(a.extracted_text for a in memory.attachments if a.extracted_text)
    memory.searchable_text = _build_searchable_text(memory.caption, attachment_text, "", memory.derived)

    db.commit()
    db.refresh(memory)
    return memory
