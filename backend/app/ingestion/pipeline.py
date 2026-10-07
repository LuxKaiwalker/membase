import datetime as dt
import logging
import shutil
import uuid

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_sessionmaker
from app.ingestion.chunking import chunk_attachment
from app.ingestion.clarify import generate_questions
from app.ingestion.enrichment import build_retrieval_text, enrich
from app.ingestion.files import ProcessedFile, UnsupportedFileError, process_upload
from app.ingestion.segmentation import segment_capture
from app.llm import get_llm_provider
from app.llm.base import LLMError, LLMProvider
from app.models import Attachment, AttachmentChunk, Capture, MemoryUnit

logger = logging.getLogger(__name__)

# Bounded, but no longer a single yes/no: each round can surface several
# relevant questions at once (asked one at a time), and an answer that
# reveals new ambiguity can trigger another round. Both bounds exist for
# the same reason the original single-question cap did -- stop the
# *system* from interrogating the user -- not to limit a user-initiated
# correct_unit(), which doesn't touch either budget. See docs/DECISIONS.md.
MAX_CLARIFICATION_ROUNDS = 3
MAX_QUESTIONS_PER_ROUND = 3

# Marker delimiting an attachment excerpt appended to a unit's
# retrieval_text by add_attachment() -- used to strip a stale excerpt
# before appending a fresh one, so repeated attachment adds don't
# accumulate duplicate/outdated excerpts.
_ATTACHMENT_EXCERPT_MARKER = "\n\nAttachment excerpt:\n"


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


async def _index_attachment_chunks(db: Session, llm: LLMProvider, attachment: Attachment, processed: ProcessedFile) -> None:
    """Deterministically chunk an attachment's extracted text (spec
    section 27) and best-effort embed each chunk. Never raises: a chunk
    embedding failure just leaves that chunk full-text-only searchable,
    same tolerance as everywhere else LLM calls happen in this pipeline.
    """
    chunks = chunk_attachment(processed.pages, processed.extracted_text)
    if not chunks:
        return
    try:
        embeddings = await llm.embed([c.text for c in chunks])
    except LLMError:
        logger.warning("Chunk embedding failed for attachment %s; chunks stay full-text-only", attachment.id)
        embeddings = [None] * len(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        db.add(
            AttachmentChunk(
                attachment_id=attachment.id,
                capture_id=attachment.capture_id,
                chunk_index=i,
                page_number=chunk.page_number,
                text=chunk.text,
                embedding=embedding,
                embedding_model=llm.embedding_model_name if embedding is not None else None,
                embedding_dim=llm.embedding_dim if embedding is not None else None,
            )
        )
    db.commit()


async def _run_full_enrichment(db: Session, llm: LLMProvider, unit: MemoryUnit, attachment_text: str) -> None:
    """Run full semantic enrichment, build retrieval_text/searchable_text,
    embed, and flip the unit to "ready". Only ever called once no
    clarification question is pending and the round budget has either
    resolved cleanly or been exhausted -- see _advance_unit(). Running
    this *before* clarification is settled would be wasted work: the
    semantic representation would just be redone once an answer comes in,
    since enrich() takes `clarifications` as context (see
    docs/DECISIONS.md for why these were split into two calls).

    Never raises outward: an *unexpected* exception here is caught and
    recorded as status="failed" (retryable via retry_enrichment), rather
    than propagating and losing the authoritative_content that was already
    committed before this ran. LLM-specific failures (down/timeout/
    malformed) are handled one level down in enrich()/embed() and don't
    even reach this except clause -- they degrade to "ready" with reduced
    enrichment instead (embedding failure specifically does not block
    "ready": see docs/DECISIONS.md, full-text search must not depend on
    the LLM being up).
    """
    try:
        semantic = await enrich(llm, unit.authoritative_content, unit.clarifications)
        unit.semantic = semantic
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


async def _advance_unit(db: Session, llm: LLMProvider, unit: MemoryUnit, attachment_text: str) -> None:
    """The single entry point for "what happens to this unit next",
    called after a fresh draft, after a clarification answer, or on
    retry. Drains any already-generated candidate questions first (fast
    -- no LLM call); only once the queue is empty does it decide whether
    another bounded round of clarify.generate_questions() is warranted
    (the *iterative* part: an answer can reveal new ambiguity worth a
    fresh question) or whether to run full enrichment and finish.

    This ordering is the whole point of the clarification/enrichment
    split (see docs/DECISIONS.md): a unit reaches "awaiting_clarification"
    -- and the user sees a question -- after one narrow, fast LLM call,
    never after the slower full-enrichment call.
    """
    if unit.clarification_queue:
        next_question, *rest = unit.clarification_queue
        unit.pending_clarification = {**next_question, "round": unit.clarification_round}
        unit.clarification_queue = rest
        unit.status = "awaiting_clarification"
        db.commit()
        return

    if unit.clarification_round < MAX_CLARIFICATION_ROUNDS:
        try:
            questions = await generate_questions(
                llm, unit.authoritative_content, unit.clarifications, max_questions=MAX_QUESTIONS_PER_ROUND
            )
        except Exception:
            # generate_questions() itself never raises (it catches
            # LLMError internally) -- this is a genuinely unexpected bug,
            # same class of failure _run_full_enrichment guards against.
            # Without this, an unexpected exception here would propagate
            # uncaught, leaving the unit's status wherever it was before
            # (not "failed"), so retry_enrichment's "draft"/"failed"-only
            # guard could never recover it.
            logger.exception("Unexpected failure generating clarification questions for unit %s", unit.id)
            unit.status = "failed"
            db.commit()
            return
        if questions:
            # Only a round that actually surfaces a question counts --
            # an ordinary unit that never needed clarification keeps
            # clarification_round at 0 forever, not "1 check used up".
            unit.clarification_round += 1
            unit.pending_clarification = {**questions[0], "round": unit.clarification_round}
            unit.clarification_queue = questions[1:]
            unit.status = "awaiting_clarification"
            db.commit()
            return

    # No clarification pending (or the round budget is exhausted): hand
    # full enrichment off to run in the background rather than blocking
    # this request -- see docs/DECISIONS.md. The API layer schedules
    # finish_unit_processing() once it sees this status, so the HTTP
    # response returns immediately and a second capture's clarification
    # flow is never blocked by the first one's enrichment.
    unit.status = "processing"
    db.commit()


async def finish_unit_processing(unit_id: uuid.UUID, llm: LLMProvider | None = None) -> None:
    """Runs full enrichment for a unit that reached status="processing"
    -- the background-task entry point scheduled by the API layer (see
    api/captures.py, api/units.py) once _advance_unit() sets that status,
    so it always runs after the triggering HTTP request's own DB session
    has already closed. Opens its own session instead of reusing anything
    from the request. `llm` defaults to the production singleton; tests
    pass a FakeLLMProvider explicitly so this never hits the real Ollama
    service.
    """
    db = get_sessionmaker()()
    try:
        if llm is None:
            llm = get_llm_provider()
        unit = db.get(MemoryUnit, unit_id)
        if unit is None or unit.status != "processing":
            # Deleted, discarded, or already finished by a concurrent
            # retry in the time between scheduling and running -- nothing
            # to do.
            return
        attachment_text = _attachment_text_for(unit.capture)
        await _run_full_enrichment(db, llm, unit, attachment_text)
    finally:
        db.close()


async def create_capture(db: Session, llm: LLMProvider, *, text: str, uploads: list[tuple[str, bytes]]) -> Capture:
    text = text.strip()
    if not text:
        raise IngestionError("text must not be empty")

    capture = Capture(raw_text=text)
    db.add(capture)
    db.flush()

    new_attachments: list[tuple[Attachment, ProcessedFile]] = []
    for filename, data in uploads:
        try:
            processed = process_upload(filename, data, capture.id)
        except UnsupportedFileError as exc:
            db.rollback()
            raise IngestionError(str(exc)) from exc
        attachment = Attachment(
            capture_id=capture.id,
            original_filename=filename,
            stored_path=processed.stored_path,
            content_type=processed.content_type,
            size_bytes=processed.size_bytes,
            extracted_text=processed.extracted_text,
            extraction_status=processed.extraction_status,
        )
        db.add(attachment)
        new_attachments.append((attachment, processed))
    db.commit()  # Phase 0: capture + attachments durably saved, no LLM call yet

    for attachment, processed in new_attachments:
        await _index_attachment_chunks(db, llm, attachment, processed)

    proposed_units = await segment_capture(llm, text)
    units = [
        MemoryUnit(capture_id=capture.id, authoritative_content=u.content, status="draft")
        for u in proposed_units
    ]
    db.add_all(units)
    db.commit()  # Phase 1: every unit's authoritative_content durably saved

    attachment_text = _attachment_text_for(capture)
    for unit in units:
        await _advance_unit(db, llm, unit, attachment_text)

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
    unit.pending_clarification = None
    # clarification_round itself advances in _advance_unit(), only when a
    # *new* round of questions is actually generated -- not here, so that
    # every question drained from the same batch's queue keeps the round
    # number it was asked under.

    await _advance_unit(db, llm, unit, _attachment_text_for(unit.capture))
    db.refresh(unit)
    return unit


async def correct_unit(db: Session, llm: LLMProvider, unit: MemoryUnit, correction: str) -> MemoryUnit:
    """User-initiated correction (spec section 30): "Correct this memory"
    free-text input, re-running enrichment -- the same underlying
    mechanism as a clarification answer (append to `clarifications`,
    re-derive `semantic`/retrieval_text/embedding), but user-triggered at
    any time a unit is ready, not system-prompted. Explicitly does not
    consume the bounded system-clarification budget (clarification_round
    is untouched): that bound exists to stop the *system* from
    interrogating the user, not to limit how often a user can proactively
    fix something.
    """
    correction = correction.strip()
    if not correction:
        raise IngestionError("correction must not be empty")
    if unit.status not in ("ready", "awaiting_clarification"):
        raise IngestionError(f"cannot correct a unit in status '{unit.status}'")

    clarifications = [*unit.clarifications, {
        "question": "(user correction)",
        "answer": correction,
        "round": unit.clarification_round,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": "user_correction",
    }]
    unit.clarifications = clarifications
    unit.pending_clarification = None

    # Straight to full enrichment, bypassing _advance_unit()'s clarification
    # check entirely -- a correction is the user proactively fixing
    # something, not grounds for the system to open a new round of its
    # own questions.
    await _run_full_enrichment(db, llm, unit, _attachment_text_for(unit.capture))
    db.refresh(unit)
    return unit


async def retry_enrichment(db: Session, llm: LLMProvider, unit: MemoryUnit) -> MemoryUnit:
    """Retry derived processing for a unit stuck in draft/failed (e.g. the
    LLM was down during ingestion). authoritative_content/clarifications
    are untouched; resumes from wherever it got stuck -- _advance_unit()
    drains any already-generated queue, otherwise re-evaluates from the
    current clarification_round."""
    await _advance_unit(db, llm, unit, _attachment_text_for(unit.capture))
    db.refresh(unit)
    return unit


async def add_attachment(db: Session, llm: LLMProvider, capture: Capture, filename: str, data: bytes) -> Capture:
    """Attach a file to an already-persisted capture.

    The new attachment's text must become visible to *both* full-text and
    semantic (vector) retrieval, not just full-text -- re-running full
    semantic enrichment for every sibling unit on every attachment add
    would be an unbounded number of LLM calls over a capture's lifetime,
    so this takes a lighter path: rebuild searchable_text (same as
    before) and re-embed from a retrieval_text that now also carries a
    short attachment excerpt, without re-deriving aliases/definitions.
    Embedding failure here is tolerated the same way as during ingestion
    (unit stays "ready", just not newly re-indexed semantically).
    """
    try:
        processed = process_upload(filename, data, capture.id)
    except UnsupportedFileError as exc:
        raise IngestionError(str(exc)) from exc

    attachment = Attachment(
        capture_id=capture.id,
        original_filename=filename,
        stored_path=processed.stored_path,
        content_type=processed.content_type,
        size_bytes=processed.size_bytes,
        extracted_text=processed.extracted_text,
        extraction_status=processed.extraction_status,
    )
    db.add(attachment)
    db.commit()

    await _index_attachment_chunks(db, llm, attachment, processed)

    attachment_text = _attachment_text_for(capture)
    for unit in capture.units:
        if unit.status != "ready":
            continue
        # Bug fix: this used to compute an excerpt-augmented text only
        # for the embed() call below, without ever assigning it back to
        # unit.retrieval_text -- so the stored embedding reflected the
        # attachment excerpt but the "text this embedding represents"
        # column didn't, and a later unrelated re-embed (e.g.
        # scripts/reembed.py after a model change) would silently
        # regress the embedding back to not knowing about the
        # attachment. Strip any previously-appended excerpt before
        # reappending so repeated attachment adds don't accumulate
        # stale excerpts from earlier calls.
        base_retrieval_text = unit.retrieval_text.split(_ATTACHMENT_EXCERPT_MARKER, 1)[0]
        excerpt = attachment_text[:1000]
        retrieval_text = f"{base_retrieval_text}{_ATTACHMENT_EXCERPT_MARKER}{excerpt}" if excerpt else base_retrieval_text
        unit.retrieval_text = retrieval_text
        unit.searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, unit.retrieval_text)
        try:
            embedding = await llm.embed([retrieval_text])
            unit.embedding = embedding[0]
            unit.embedding_model = llm.embedding_model_name
            unit.embedding_dim = llm.embedding_dim
        except LLMError:
            logger.warning("Re-embedding after attachment add failed for unit %s", unit.id)

    db.commit()
    db.refresh(capture)
    return capture


def delete_capture(db: Session, capture: Capture) -> None:
    """Explicit user deletion (spec section 31): removes the capture and
    everything derived from it (units, attachments, chunks via DB
    cascade) plus the original files on disk. No automatic/relevance-based
    deletion exists anywhere in this codebase -- this is the only way a
    memory ever goes away, and it's always a direct user action.
    """
    capture_dir = get_settings().uploads_dir / str(capture.id)
    db.delete(capture)
    db.commit()
    shutil.rmtree(capture_dir, ignore_errors=True)


async def discard_unit(db: Session, unit: MemoryUnit) -> MemoryUnit:
    """Explicit user cancellation of a unit awaiting clarification (spec
    section 3: "the user may cancel/discard the draft"). Deliberately a
    status flip, not a hard delete -- see docs/DECISIONS.md for why this
    repo doesn't need a separate hard-delete code path for this case."""
    unit.status = "discarded"
    unit.pending_clarification = None
    db.commit()
    db.refresh(unit)
    return unit
