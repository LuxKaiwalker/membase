import datetime as dt
import logging

from sqlalchemy.orm import Session

from app.ingestion.enrichment import build_retrieval_text, enrich
from app.ingestion.files import UnsupportedFileError, process_upload
from app.ingestion.segmentation import segment_capture
from app.llm.base import LLMError, LLMProvider
from app.models import Attachment, Capture, MemoryUnit

logger = logging.getLogger(__name__)

# 0 or 1 clarification question per unit. See docs/DECISIONS.md: bounded
# deliberately low so the system never turns into an interrogation.
MAX_CLARIFICATION_ROUNDS = 1


class IngestionError(ValueError):
    """Input-validation failure -- caller should turn this into a 4xx."""


def _build_searchable_text(authoritative_content: str, attachment_text: str, retrieval_text: str) -> str:
    parts = [authoritative_content]
    if attachment_text:
        parts.append(attachment_text)
    if retrieval_text:
        parts.append(retrieval_text)
    return "\n\n".join(parts)


def _attachment_text_for(capture: Capture) -> str:
    return "\n\n".join(a.extracted_text for a in capture.attachments if a.extracted_text)


async def _enrich_and_index(db: Session, llm: LLMProvider, unit: MemoryUnit, attachment_text: str) -> None:
    """Run semantic enrichment, build retrieval_text/searchable_text, embed,
    and set unit.status -- the "derived processing" half of the two-phase
    commit described in app/ingestion/pipeline.py:create_capture.

    Never raises outward: an *unexpected* exception here is caught and
    recorded as status="failed" (retryable via retry_enrichment), rather
    than propagating and losing the authoritative_content that was already
    committed before this ran. LLM-specific failures (down/timeout/
    malformed) are handled one level down in enrich()/embed() and don't
    even reach this except clause -- they degrade status to "ready" with
    reduced enrichment instead, same tolerance as before.
    """
    try:
        allow_clarification = unit.clarification_round < MAX_CLARIFICATION_ROUNDS
        semantic = await enrich(
            llm, unit.authoritative_content, unit.clarifications, allow_clarification=allow_clarification
        )
        unit.semantic = semantic

        if semantic.get("clarification_question") and allow_clarification:
            unit.pending_clarification = {
                "question": semantic["clarification_question"],
                "field": semantic.get("clarification_field") or "clarification",
                "round": unit.clarification_round,
            }
        else:
            unit.pending_clarification = None

        unit.retrieval_text = build_retrieval_text(unit.authoritative_content, unit.clarifications, semantic)
        unit.searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, unit.retrieval_text)

        try:
            embedding = await llm.embed([unit.retrieval_text])
            unit.embedding = embedding[0]
            unit.embedding_model = llm.embedding_model_name
            unit.embedding_dim = llm.embedding_dim
        except LLMError:
            logger.warning("Embedding failed for unit %s; leaving unindexed for vector search", unit.id)

        unit.status = "ready"
    except Exception:
        logger.exception("Unexpected failure enriching unit %s", unit.id)
        unit.status = "failed"
    db.commit()


async def create_capture(db: Session, llm: LLMProvider, *, text: str, uploads: list[tuple[str, bytes]]) -> Capture:
    text = text.strip()
    if not text:
        raise IngestionError("text must not be empty")

    capture = Capture(raw_text=text)
    db.add(capture)
    db.flush()

    for filename, data in uploads:
        try:
            processed = process_upload(filename, data, capture.id)
        except UnsupportedFileError as exc:
            db.rollback()
            raise IngestionError(str(exc)) from exc
        db.add(
            Attachment(
                capture_id=capture.id,
                original_filename=filename,
                stored_path=processed.stored_path,
                content_type=processed.content_type,
                size_bytes=processed.size_bytes,
                extracted_text=processed.extracted_text,
                extraction_status=processed.extraction_status,
            )
        )
    db.commit()  # Phase 0: capture + attachments durably saved, no LLM call yet

    proposed_units = await segment_capture(llm, text)
    units = [
        MemoryUnit(capture_id=capture.id, authoritative_content=u.content, status="draft")
        for u in proposed_units
    ]
    db.add_all(units)
    db.commit()  # Phase 1: every unit's authoritative_content durably saved

    attachment_text = _attachment_text_for(capture)
    for unit in units:
        await _enrich_and_index(db, llm, unit, attachment_text)

    db.refresh(capture)
    return capture


async def resolve_clarification(db: Session, llm: LLMProvider, unit: MemoryUnit, answer: str) -> MemoryUnit:
    answer = answer.strip()
    if not answer:
        raise IngestionError("clarification answer must not be empty")
    if unit.pending_clarification is None:
        raise IngestionError("unit has no pending clarification")

    pending = unit.pending_clarification
    clarifications = [*unit.clarifications, {
        "question": pending["question"],
        "answer": answer,
        "round": unit.clarification_round,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
    }]
    unit.clarifications = clarifications
    unit.clarification_round += 1
    unit.pending_clarification = None

    await _enrich_and_index(db, llm, unit, _attachment_text_for(unit.capture))
    db.refresh(unit)
    return unit


async def retry_enrichment(db: Session, llm: LLMProvider, unit: MemoryUnit) -> MemoryUnit:
    """Retry derived processing for a unit stuck in draft/failed (e.g. the
    LLM was down during ingestion). authoritative_content/clarifications
    are untouched; only the derived half re-runs."""
    await _enrich_and_index(db, llm, unit, _attachment_text_for(unit.capture))
    db.refresh(unit)
    return unit


async def add_attachment(db: Session, capture: Capture, filename: str, data: bytes) -> Capture:
    """Attach a file to an already-persisted capture.

    Scope decision (docs/TODO.md): extends every sibling unit's full-text
    index with the new attachment's text, but does not re-run semantic
    enrichment or re-embed those units -- doing so for every post-hoc
    attachment would mean an unbounded number of LLM calls over a
    capture's lifetime for an MVP feature secondary to the core loop.
    """
    try:
        processed = process_upload(filename, data, capture.id)
    except UnsupportedFileError as exc:
        raise IngestionError(str(exc)) from exc

    db.add(
        Attachment(
            capture_id=capture.id,
            original_filename=filename,
            stored_path=processed.stored_path,
            content_type=processed.content_type,
            size_bytes=processed.size_bytes,
            extracted_text=processed.extracted_text,
            extraction_status=processed.extraction_status,
        )
    )
    db.flush()

    attachment_text = _attachment_text_for(capture)
    for unit in capture.units:
        unit.searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, unit.retrieval_text)

    db.commit()
    db.refresh(capture)
    return capture
