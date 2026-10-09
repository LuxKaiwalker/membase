import datetime as dt
import logging
import re
import shutil
import uuid

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_sessionmaker
from app.ingestion.analysis import (
    Semantics,
    UnitAnalysis,
    analyze_capture,
    build_retrieval_text,
    reanalyze_unit,
)
from app.ingestion.chunking import chunk_attachment
from app.ingestion.files import ProcessedFile, UnsupportedFileError, extract_pdf_pages, process_upload
from app.ingestion import structurer
from app.llm import get_llm_provider
from app.llm.base import LLMError, LLMProvider
from app.models import Attachment, AttachmentChunk, Capture, MemoryUnit

logger = logging.getLogger(__name__)

# Bounded rounds of *system*-initiated clarification -- stops the system
# from interrogating the user, not a limit on user-initiated correct_unit().
# See docs/DECISIONS.md "Sequential clarification".
MAX_CLARIFICATION_ROUNDS = 3

# Marker delimiting an attachment excerpt appended to a unit's
# retrieval_text by add_attachment() -- used to strip a stale excerpt
# before appending a fresh one, so repeated attachment adds don't
# accumulate duplicate/outdated excerpts.
_ATTACHMENT_EXCERPT_MARKER = "\n\nAttachment excerpt:\n"

# Deterministic, no-LLM classification of a clarification answer's
# outcome (see docs/DECISIONS.md "Clarification outcomes" -- spec
# explicitly calls for ANSWERED/UNKNOWN/DECLINED, not just answer/
# failure). Keyword heuristics, not a model call: recognizing "I don't
# know" is exactly the kind of decision ordinary code should make.
_UNKNOWN_PATTERNS = (
    r"\bi don'?t know\b", r"\bnot sure\b", r"\bno idea\b", r"\bunsure\b", r"\bunknown\b",
)
_DECLINED_PATTERNS = (
    r"\bdon'?t want to (say|answer|share)\b", r"\brather not\b", r"\bprefer not to\b",
    r"\bskip (this|that)\b", r"\bnone of your business\b",
)


def _classify_outcome(answer: str) -> str:
    lowered = answer.lower()
    if any(re.search(p, lowered) for p in _DECLINED_PATTERNS):
        return "declined"
    if any(re.search(p, lowered) for p in _UNKNOWN_PATTERNS):
        return "unknown"
    return "answered"


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
    """Full, uncapped combined attachment text -- used for the full-text
    search index (_build_searchable_text), which has no model context
    budget to protect. Never pass this directly into an LLM prompt; see
    _attachment_text_for_analysis() below."""
    return "\n\n".join(a.extracted_text for a in capture.attachments if a.extracted_text)


# A generation call's context window is shared between its input prompt
# and its output (see docs/DECISIONS.md "Chunk long captures before
# analysis" -- the same 4096-token budget that bug was about). Unlike
# that fix, which was about OUTPUT size (many units needing full
# semantics), an uncapped attachment_text is an INPUT-side version of
# the same risk: a large PDF, or several attachments concatenated with
# no limit, crowds out the budget the model needs for its own response,
# on every analyze_capture()/reanalyze_unit() call a capture's siblings
# ever make. Capped independently of should_chunk()'s indexing
# threshold -- that constant governs a different concern (whether
# AttachmentChunk rows are worth creating), not how much raw attachment
# text is safe to paste into one generation prompt.
MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS = 1500


def _attachment_text_for_analysis(capture: Capture) -> str:
    """Bounded combined attachment text for an LLM prompt -- see
    MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS above. Still gives the model
    something to resolve a vague reference against (e.g. an image
    caption, or a document's opening section); a full large document's
    *content* remains separately searchable via AttachmentChunk
    regardless of what's included here."""
    return _attachment_text_for(capture)[:MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS]


# --- cheap prior-memory lookup (see docs/DECISIONS.md "Prior memory
# context") -------------------------------------------------------------
# Deliberately NOT a semantic/vector search over the whole store -- a
# plain lexical lookup against already-ready units' retrieval_text for
# distinctive (capitalized, or multi-word proper-noun-shaped) terms in
# the new text. Context only, injected into the analysis prompt; never
# authoritative over what the user is currently saying (see
# docs/DECISIONS.md "Prior memory is context, not authority").
_CANDIDATE_TERM_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,2}\b")
_PRIOR_CONTEXT_SNIPPET_CHARS = 300
_PRIOR_CONTEXT_LIMIT = 3


def _candidate_terms(raw_text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _CANDIDATE_TERM_RE.finditer(raw_text):
        seen.setdefault(m.group(0), None)
    return list(seen)


def _find_related_context(db: Session, raw_text: str) -> list[str]:
    """Pure SQL lexical lookup, no LLM call -- bounded to a small number
    of short snippets so this never turns ingestion into a retrieval
    session (see docs/DECISIONS.md)."""
    terms = _candidate_terms(raw_text)
    if not terms:
        return []
    snippets: list[str] = []
    seen_units: set[uuid.UUID] = set()
    for term in terms:
        if len(snippets) >= _PRIOR_CONTEXT_LIMIT:
            break
        rows = (
            db.query(MemoryUnit)
            .filter(MemoryUnit.status == "ready", MemoryUnit.retrieval_text.ilike(f"%{term}%"))
            .limit(_PRIOR_CONTEXT_LIMIT)
            .all()
        )
        for unit in rows:
            if unit.id in seen_units or len(snippets) >= _PRIOR_CONTEXT_LIMIT:
                continue
            seen_units.add(unit.id)
            snippets.append(unit.retrieval_text[:_PRIOR_CONTEXT_SNIPPET_CHARS])
    return snippets


# --- explicit metadata authority (see docs/DECISIONS.md) ---------------
# User-provided fields always win over whatever the LLM separately
# inferred for the same field. A genuine disagreement (both sides
# non-empty and different) is surfaced as a clarification rather than
# silently resolved either way.
_METADATA_TO_SEMANTIC_FIELD = {
    "location": "locations",
    "date": "event_date",
    "people": "people",
    "organizations": "organizations",
    "topics": "topics",
}


def _apply_explicit_metadata(semantic: Semantics, explicit_metadata: dict) -> str | None:
    """Mutates `semantic` in place to apply explicit-field authority.
    Returns a clarification question string if a genuine conflict was
    found (both sides non-empty and different), else None.
    """
    if not explicit_metadata:
        return None
    for meta_field, semantic_field in _METADATA_TO_SEMANTIC_FIELD.items():
        explicit_value = explicit_metadata.get(meta_field)
        if not explicit_value:
            continue
        current = semantic.get(semantic_field)
        if semantic_field == "event_date":
            inferred_list = [current] if isinstance(current, str) and current else []
        else:
            inferred_list = current if isinstance(current, list) else []
        inferred_norm = {v.strip().lower() for v in inferred_list if isinstance(v, str)}
        explicit_values = explicit_value if isinstance(explicit_value, list) else [explicit_value]
        explicit_norm = {str(v).strip().lower() for v in explicit_values if v}

        conflicting = inferred_norm - explicit_norm
        if conflicting:
            inferred_display = ", ".join(sorted(conflicting))
            explicit_display = ", ".join(str(v) for v in explicit_values)
            question = (
                f"You entered {meta_field}={explicit_display!r}, but the text seems to say "
                f"{meta_field}={inferred_display!r}. Which should be associated with this memory?"
            )
            # Explicit value still wins for now (authority rule) -- the
            # question offers the user a chance to correct it, but never
            # silently defers to the model's guess in the meantime.
            if semantic_field == "event_date":
                semantic[semantic_field] = str(explicit_values[0])
            else:
                semantic[semantic_field] = list({*inferred_list, *[str(v) for v in explicit_values]})
            return question

        if semantic_field == "event_date":
            semantic[semantic_field] = str(explicit_values[0])
        else:
            semantic[semantic_field] = list({*inferred_list, *[str(v) for v in explicit_values]})
    return None


_IMAGE_CAPTION_PROMPT = (
    "Describe this image factually and in detail, as if writing a caption "
    "for someone who can't see it: people, objects, scene, setting, colors, "
    "and anything notable. Transcribe any visible text verbatim. This "
    "description becomes the image's only searchable representation, so "
    "include anything someone might later search for."
)


async def _describe_image(llm: LLMProvider, data: bytes) -> str | None:
    """Caption an image via the generation model's native multimodal
    understanding. Best-effort: an LLMError leaves the image stored
    (already committed before this runs) but without a textual
    description, rather than blocking anything.
    """
    try:
        description = await llm.generate(_IMAGE_CAPTION_PROMPT, images=[data], temperature=0.2)
    except LLMError:
        logger.warning("Image captioning failed; image stored but not text-searchable")
        return None
    description = description.strip()
    return description or None


def is_uncaptioned_image(attachment: Attachment) -> bool:
    return attachment.content_type.startswith("image/") and attachment.extraction_status == "unsupported"


async def finish_image_captioning(attachment_id: uuid.UUID, llm: LLMProvider | None = None) -> None:
    """Background-task entry point: captions one image attachment after
    the triggering request has already returned (see docs/DECISIONS.md
    "Decouple image captioning"). Deliberately NOT in the synchronous
    ingestion path any more -- a text memory's formation (segmentation +
    clarification) never waits on this. The trade-off, accepted
    explicitly: the *first* analysis pass for a capture with images runs
    without their captions as context, so a vague reference an image
    would have resolved may get asked about anyway.

    By the time this runs, any sibling unit may already be "ready" (its
    embedding was built with no knowledge of this caption) -- the same
    light re-indexing path add_attachment() uses for a posthoc attachment
    is reused here so the caption still becomes searchable, either via
    chunking (if long enough) or folded into sibling units' retrieval
    text (if not).
    """
    db = get_sessionmaker()()
    try:
        if llm is None:
            llm = get_llm_provider()
        attachment = db.get(Attachment, attachment_id)
        if attachment is None or not is_uncaptioned_image(attachment):
            return
        data = (get_settings().uploads_dir / attachment.stored_path).read_bytes()
        description = await _describe_image(llm, data)
        if not description:
            return
        attachment.extracted_text = description
        attachment.extraction_status = "ok"
        db.commit()

        if should_chunk(description):
            processed = ProcessedFile(
                stored_path=attachment.stored_path,
                content_type=attachment.content_type,
                size_bytes=attachment.size_bytes,
                extracted_text=description,
                extraction_status="ok",
                pages=None,
            )
            await _index_attachment_chunks(db, llm, attachment, processed)
        await _refresh_ready_units_with_attachment_excerpt(db, llm, attachment.capture)
    finally:
        db.close()


async def finish_attachment_chunking(attachment_id: uuid.UUID, llm: LLMProvider | None = None) -> None:
    """Background-task entry point for chunk-level indexing, scheduled
    after an attachment is created (and, for an image, independently of
    -- not chained after -- finish_image_captioning; a short caption
    rarely needs chunking at all, see should_chunk below).

    Re-derives `pages`/`extracted_text` from the already-persisted
    Attachment row (and, for a PDF, a cheap local re-extraction -- no
    LLM call) rather than requiring the caller to thread the original
    in-memory ProcessedFile across the background-task boundary.
    """
    db = get_sessionmaker()()
    try:
        if llm is None:
            llm = get_llm_provider()
        attachment = db.get(Attachment, attachment_id)
        if attachment is None:
            return
        if not should_chunk(attachment.extracted_text):
            return
        pages = None
        if attachment.content_type == "application/pdf":
            try:
                data = (get_settings().uploads_dir / attachment.stored_path).read_bytes()
                pages = extract_pdf_pages(data)
            except Exception:
                logger.exception(
                    "Could not re-extract PDF pages for attachment %s; chunking without page numbers",
                    attachment.id,
                )
        processed = ProcessedFile(
            stored_path=attachment.stored_path,
            content_type=attachment.content_type,
            size_bytes=attachment.size_bytes,
            extracted_text=attachment.extracted_text,
            extraction_status=attachment.extraction_status,
            pages=pages,
        )
        await _index_attachment_chunks(db, llm, attachment, processed)
    finally:
        db.close()


def should_chunk(extracted_text: str | None) -> bool:
    """Small text/PDF: the parent unit's full-text search over
    extracted_text already covers it -- chunk-level embedding adds
    nothing a single excerpt doesn't (see docs/DECISIONS.md "Lazy
    attachment chunking"). Large documents still get fully chunked."""
    return bool(extracted_text) and len(extracted_text) >= get_settings().membase_min_chars_for_chunking


async def _index_attachment_chunks(db: Session, llm: LLMProvider, attachment: Attachment, processed: ProcessedFile) -> None:
    """Deterministically chunk an attachment's extracted text and
    best-effort embed each chunk. Never raises: a chunk embedding
    failure just leaves that chunk full-text-only searchable.

    Idempotent: clears any chunks already indexed for this attachment
    first, so this is safe to call again (e.g. a manual retry after a
    dropped background task -- see docs/DECISIONS.md "Attachment
    chunking recovery") without accumulating duplicates.
    """
    db.query(AttachmentChunk).filter(AttachmentChunk.attachment_id == attachment.id).delete()

    max_chunks = get_settings().membase_max_chunks_per_attachment
    chunks = chunk_attachment(processed.pages, processed.extracted_text, max_chunks=max_chunks)
    if not chunks:
        db.commit()
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


async def retry_attachment_chunking(db: Session, llm: LLMProvider, attachment: Attachment) -> None:
    """Manual recovery path if a server restart dropped the background
    chunking task (see docs/DECISIONS.md "Attachment chunking
    recovery") -- no separate pending/processing status needed, since
    _index_attachment_chunks is idempotent and chunk existence is
    itself the signal."""
    await _index_attachment_chunks(db, llm, attachment, ProcessedFile(
        stored_path=attachment.stored_path,
        content_type=attachment.content_type,
        size_bytes=attachment.size_bytes,
        extracted_text=attachment.extracted_text,
        extraction_status=attachment.extraction_status,
        pages=extract_pdf_pages((get_settings().uploads_dir / attachment.stored_path).read_bytes())
        if attachment.content_type == "application/pdf" else None,
    ))


# --- contradiction detection (see docs/DECISIONS.md) --------------------
# Cheap, deterministic comparison of subject/predicate/object relations
# already extracted by analysis -- no extra LLM call. Not a general
# truth-maintenance system: a match only ever adds an advisory marker to
# both units, never deletes/supersedes either.

def _relation_key_map(semantic: dict) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for r in (semantic or {}).get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        subject = str(r.get("subject", "")).strip().lower()
        predicate = str(r.get("predicate", "")).strip().lower()
        obj = str(r.get("object", "")).strip()
        if subject and predicate and obj:
            out[(subject, predicate)] = obj
    return out


async def _detect_conflicts(db: Session, llm: LLMProvider, unit: MemoryUnit) -> None:
    my_rels = _relation_key_map(unit.semantic)
    if not my_rels or not unit.retrieval_text:
        return
    from app.retrieval.pipeline import hybrid_search  # local import: avoids a hard ingestion<->retrieval coupling at module load time

    # Search by what the relation is ABOUT (subject + predicate), not the
    # unit's full retrieval_text -- the full text also contains the
    # (possibly differing) object, and a genuine numeric/factual conflict
    # by definition won't appear in the other memory's text, which would
    # otherwise make a full-text query over it systematically fail to
    # find exactly the case this exists to catch.
    query = " ".join({term for key in my_rels for term in key})
    try:
        candidates = await hybrid_search(
            db, llm, query, limit=get_settings().membase_conflict_check_candidates
        )
    except LLMError:
        return

    for cand in candidates:
        if cand.unit.id == unit.id:
            continue
        cand_rels = _relation_key_map(cand.unit.semantic)
        for key, my_obj in my_rels.items():
            cand_obj = cand_rels.get(key)
            if cand_obj is None or cand_obj.strip().lower() == my_obj.strip().lower():
                continue
            subject, predicate = key
            note = f"'{subject} {predicate}' -- here: '{my_obj}', in another memory: '{cand_obj}'."
            if not any(c.get("unit_id") == str(cand.unit.id) for c in unit.potential_conflicts):
                unit.potential_conflicts = [*unit.potential_conflicts, {"unit_id": str(cand.unit.id), "note": note}]
            if not any(c.get("unit_id") == str(unit.id) for c in cand.unit.potential_conflicts):
                cand.unit.potential_conflicts = [*cand.unit.potential_conflicts, {"unit_id": str(unit.id), "note": note}]
                db.add(cand.unit)
    db.commit()


# --- processing lease (see docs/DECISIONS.md "processing lease") -------
# Replaces holding a database row lock for an entire LLM call's
# duration: claim the unit with a short transaction (status=processing +
# a fresh token + a timestamp), do the LLM work with no transaction
# open at all, then commit the result with an UPDATE ... WHERE token
# matches. A token mismatch means someone else already finished (or is
# finishing) this unit -- the attempt is simply dropped, never stomped.

def _claim_processing(db: Session, unit: MemoryUnit) -> str:
    token = uuid.uuid4().hex
    unit.status = "processing"
    unit.processing_token = token
    unit.processing_started_at = dt.datetime.now(dt.UTC)
    db.commit()
    return token


def _commit_if_token_matches(db: Session, unit_id: uuid.UUID, token: str, values: dict) -> bool:
    """Returns True if this call's token still matched (i.e. its result
    was actually committed), False if another attempt already claimed
    this unit first."""
    result = db.execute(
        update(MemoryUnit)
        .where(MemoryUnit.id == unit_id, MemoryUnit.processing_token == token)
        .values(**values, processing_token=None, processing_started_at=None)
    )
    db.commit()
    return result.rowcount > 0


async def _finalize_unit(db: Session, llm: LLMProvider, unit: MemoryUnit, token: str, attachment_text: str) -> None:
    """The background step for a unit that has no clarification pending:
    build retrieval_text/searchable_text from the semantic fields the
    analysis call already produced (no further generation call needed),
    embed once, detect conflicts, and commit -- gated on `token` still
    matching so a concurrent retry can never stomp a result that's
    already landed (see the processing-lease docstrings above).
    """
    try:
        retrieval_text = build_retrieval_text(unit.authoritative_content, unit.clarifications, unit.semantic)
        searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, retrieval_text)
        values: dict = {"retrieval_text": retrieval_text, "searchable_text": searchable_text, "status": "ready"}
        try:
            embedding = await llm.embed([retrieval_text])
            values["embedding"] = embedding[0]
            values["embedding_model"] = llm.embedding_model_name
            values["embedding_dim"] = llm.embedding_dim
        except LLMError:
            logger.warning("Embedding failed for unit %s; leaving unindexed for vector search", unit.id)
    except Exception:
        logger.exception("Unexpected failure finalizing unit %s", unit.id)
        _commit_if_token_matches(db, unit.id, token, {"status": "failed"})
        return

    committed = _commit_if_token_matches(db, unit.id, token, values)
    if committed and values.get("embedding") is not None:
        db.refresh(unit)
        await _detect_conflicts(db, llm, unit)


async def finish_unit_processing(unit_id: uuid.UUID, llm: LLMProvider | None = None) -> None:
    """Background-task entry point scheduled by the API layer once a
    unit reaches status="processing" with no clarification pending.
    Opens its own session; `llm` defaults to the production singleton,
    tests pass a FakeLLMProvider explicitly.
    """
    db = get_sessionmaker()()
    try:
        if llm is None:
            llm = get_llm_provider()
        unit = db.get(MemoryUnit, unit_id)
        if unit is None or unit.status != "processing" or not unit.processing_token:
            return
        token = unit.processing_token
        attachment_text = _attachment_text_for(unit.capture)
        await _finalize_unit(db, llm, unit, token, attachment_text)
    finally:
        db.close()


# --- membase structurer (see docs/DECISIONS.md "The 'membase
# structurer'") -- a second, independent background pass that deepens
# an already-"ready" unit's representation. Uses the exact same
# processing-lease pattern as _claim_processing/_commit_if_token_matches
# above, with its own token/timestamp pair (structure_token/
# structure_started_at) so the two leases can never collide. Critically,
# this NEVER touches `status` -- a unit is exactly as searchable before,
# during, and after structuring.

def _claim_structuring(db: Session, unit: MemoryUnit) -> str:
    token = uuid.uuid4().hex
    unit.structure_status = "structuring"
    unit.structure_token = token
    unit.structure_started_at = dt.datetime.now(dt.UTC)
    db.commit()
    return token


def _commit_structuring_if_token_matches(db: Session, unit_id: uuid.UUID, token: str, values: dict) -> bool:
    result = db.execute(
        update(MemoryUnit)
        .where(MemoryUnit.id == unit_id, MemoryUnit.structure_token == token)
        .values(**values, structure_token=None, structure_started_at=None)
    )
    db.commit()
    return result.rowcount > 0


async def _finalize_structuring(db: Session, llm: LLMProvider, unit: MemoryUnit, token: str, attachment_text: str) -> None:
    """The structuring step itself: find candidate related units via
    the existing hybrid_search() (same reuse pattern as
    _detect_conflicts -- no new retrieval primitive), ask the model for
    richer semantics + which candidates are genuinely related, then
    atomically swap retrieval_text/searchable_text/embedding/semantic in
    place -- gated on `token` so a stale/duplicate attempt can never
    stomp a result that already landed. A failure here leaves the
    unit's existing (fast-path) representation completely untouched;
    structuring only ever upgrades, never degrades.
    """
    from app.retrieval.pipeline import hybrid_search  # local import: avoids a hard ingestion<->retrieval coupling at module load time

    # Search by what the unit is ABOUT (its own topics/people/orgs/
    # locations/aliases), not its full retrieval_text -- the same class
    # of bug fixed in _detect_conflicts applies here too:
    # websearch_to_tsquery ANDs bare terms together by default, so even
    # a short query over several about-terms would require a related
    # memory to share EVERY one of them, when sharing just one (e.g. the
    # same location, different topic) is already a meaningful signal.
    # Joined with explicit "OR" instead -- a real related memory only
    # needs to match *some* of what this one is about, not all of it.
    # Capped at a handful of terms so the query itself stays bounded.
    semantic = unit.semantic or {}
    about_terms = [
        *semantic.get("topics", []), *semantic.get("people", []), *semantic.get("organizations", []),
        *semantic.get("locations", []), *semantic.get("aliases", []),
    ][:6]
    query = " OR ".join(about_terms) if about_terms else unit.authoritative_content
    try:
        candidate_scored = await hybrid_search(db, llm, query, limit=structurer.MAX_RELATED_CANDIDATES + 1)
    except LLMError:
        candidate_scored = []
    candidates = [(str(s.unit.id), s.unit.retrieval_text[:300]) for s in candidate_scored if s.unit.id != unit.id]

    result = await structurer.structure_unit(llm, unit.authoritative_content, unit.clarifications, attachment_text, candidates)
    if result.semantic.get("enrichment_status") == "unavailable":
        _commit_structuring_if_token_matches(db, unit.id, token, {"structure_status": "failed"})
        return

    semantic = dict(result.semantic)
    _apply_explicit_metadata(semantic, unit.explicit_metadata)
    retrieval_text = build_retrieval_text(unit.authoritative_content, unit.clarifications, semantic)
    searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, retrieval_text)
    related_unit_ids = [candidates[i][0] for i in result.related_indices]

    values: dict = {
        "semantic": semantic,
        "retrieval_text": retrieval_text,
        "searchable_text": searchable_text,
        "related_unit_ids": related_unit_ids,
        "structure_status": "structured",
        "structured_at": dt.datetime.now(dt.UTC),
    }
    try:
        embedding = await llm.embed([retrieval_text])
        values["embedding"] = embedding[0]
        values["embedding_model"] = llm.embedding_model_name
        values["embedding_dim"] = llm.embedding_dim
    except LLMError:
        logger.warning("Structuring embed failed for unit %s; keeping prior embedding", unit.id)

    _commit_structuring_if_token_matches(db, unit.id, token, values)


async def finish_structuring(unit_id: uuid.UUID, llm: LLMProvider | None = None) -> None:
    """Background-task entry point -- scheduled by the API layer right
    after finish_unit_processing for the same unit (Starlette's
    BackgroundTasks run sequentially in registration order within one
    request, so by the time this runs the unit is already "ready" with
    real semantic/retrieval_text to build on; see docs/DECISIONS.md).
    Only ever acts on a unit that is actually "ready" -- structuring a
    unit still awaiting clarification, or one that never finished the
    fast pipeline, would have nothing solid to deepen.
    """
    db = get_sessionmaker()()
    try:
        if llm is None:
            llm = get_llm_provider()
        unit = db.get(MemoryUnit, unit_id)
        if unit is None or unit.status != "ready" or unit.structure_status != "unstructured":
            return
        token = _claim_structuring(db, unit)
        attachment_text = _attachment_text_for_analysis(unit.capture)
        await _finalize_structuring(db, llm, unit, token, attachment_text)
    finally:
        db.close()


async def retry_structuring(db: Session, llm: LLMProvider, unit: MemoryUnit) -> MemoryUnit:
    """Recovery path for a unit stuck in structure_status="structuring"
    (a server restart dropped the background task) or "failed". Mirrors
    retry_enrichment()'s staleness check for the fast pipeline's lease --
    see docs/DECISIONS.md.
    """
    if unit.status != "ready":
        raise IngestionError("unit must be 'ready' before it can be structured")

    if unit.structure_status == "structuring":
        stale_after = get_settings().membase_processing_lease_stale_seconds
        started = unit.structure_started_at
        age = (dt.datetime.now(dt.UTC) - started).total_seconds() if started else float("inf")
        if age < stale_after:
            return unit  # still plausibly in flight; nothing to reclaim yet

    if unit.structure_status not in ("unstructured", "structuring", "failed"):
        return unit

    token = _claim_structuring(db, unit)
    attachment_text = _attachment_text_for_analysis(unit.capture)
    await _finalize_structuring(db, llm, unit, token, attachment_text)
    db.refresh(unit)
    return unit


def _unit_from_analysis(
    capture_id: uuid.UUID, analysis: UnitAnalysis, explicit_metadata: dict, allow_clarification: bool = True
) -> MemoryUnit:
    semantic = dict(analysis.semantic)
    metadata_question = _apply_explicit_metadata(semantic, explicit_metadata)
    # When clarification is disabled (see docs/DECISIONS.md "Disable
    # follow-up questions"), any question -- model-proposed OR a genuine
    # explicit-metadata conflict -- is suppressed uniformly: the whole
    # point of the setting is that a unit never waits on the user, not
    # just that the model's own guess is skipped. The explicit value
    # still silently wins either way (_apply_explicit_metadata already
    # applied it above); only the surfaced QUESTION is dropped.
    question = (metadata_question or analysis.clarification_question) if allow_clarification else None
    field = "metadata conflict" if metadata_question else analysis.clarification_field

    unit = MemoryUnit(
        capture_id=capture_id,
        authoritative_content=analysis.content,
        status="draft",
        semantic=semantic,
        explicit_metadata=explicit_metadata,
    )
    if question:
        unit.pending_clarification = {"question": question, "field": field, "round": 1}
        unit.clarification_round = 1
        unit.status = "awaiting_clarification"
    return unit


async def create_capture(
    db: Session,
    llm: LLMProvider,
    *,
    text: str,
    uploads: list[tuple[str, bytes]],
    explicit_metadata: dict | None = None,
    allow_clarification: bool = True,
) -> Capture:
    """`allow_clarification=False` (see docs/DECISIONS.md "Disable
    follow-up questions") removes the clarification round-trip from
    this capture's storage cycle entirely: no resulting unit can ever
    reach `status="awaiting_clarification"`, regardless of what the
    model proposes or whether explicit metadata conflicts with
    inference. Every unit goes straight to `"processing"` -- the
    analysis call itself still runs (segmentation/semantics are not
    optional), but the unbounded, human-paced wait for an answer is
    what's eliminated.
    """
    text = text.strip()
    if not text:
        raise IngestionError("text must not be empty")
    max_chars = get_settings().membase_max_text_chars
    if len(text) > max_chars:
        raise IngestionError(f"text exceeds max length of {max_chars} characters")
    explicit_metadata = explicit_metadata or {}

    capture = Capture(raw_text=text)
    db.add(capture)
    db.flush()

    new_attachments: list[Attachment] = []
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
        new_attachments.append(attachment)
    db.commit()  # Phase 0: capture + attachments durably saved, no LLM call yet

    # Image captioning is NOT run here -- it's scheduled as a background
    # task by the API layer alongside chunking, decoupled from text
    # memory formation entirely (see docs/DECISIONS.md "Decouple image
    # captioning"). Non-image attachment text (plain extraction, no LLM)
    # is already available and *is* used below.
    attachment_text = _attachment_text_for_analysis(capture)
    prior_context = _find_related_context(db, text)

    analyses = await analyze_capture(llm, text, attachment_text, prior_context)
    units = [_unit_from_analysis(capture.id, a, explicit_metadata, allow_clarification) for a in analyses]
    db.add_all(units)
    db.commit()  # Phase 1: every unit's authoritative_content + analysis durably saved

    for unit in units:
        if unit.status != "awaiting_clarification":
            _claim_processing(db, unit)

    db.refresh(capture)
    return capture


async def resolve_clarification(db: Session, llm: LLMProvider, unit: MemoryUnit, answer: str | None) -> MemoryUnit:
    """Answering is optional (soft client-side guard only, see
    docs/DECISIONS.md) -- a blank/omitted answer skips this question:
    nothing is added to `clarifications`. A non-blank answer is
    classified (answered/unknown/declined, see _classify_outcome) and
    recorded either way, then the unit is re-analyzed exactly once
    (sequential clarification -- never a pre-generated queue): the
    single reanalyze_unit() call both re-derives semantic fields with
    the new answer as context AND decides whether a further question is
    still warranted.
    """
    answer = (answer or "").strip()
    max_chars = get_settings().membase_max_text_chars
    if len(answer) > max_chars:
        raise IngestionError(f"answer exceeds max length of {max_chars} characters")
    if unit.pending_clarification is None:
        raise IngestionError("unit has no pending clarification")

    pending = unit.pending_clarification
    declined_or_unknown_fields = {
        c["field"] for c in unit.clarifications if c.get("outcome") in ("declined", "unknown")
    }
    if answer:
        outcome = _classify_outcome(answer)
        clarifications = [*unit.clarifications, {
            "question": pending["question"],
            "answer": answer,
            "field": pending.get("field", "clarification"),
            "round": unit.clarification_round,
            "outcome": outcome,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
        }]
        unit.clarifications = clarifications
        if outcome in ("declined", "unknown"):
            declined_or_unknown_fields.add(pending.get("field", "clarification"))
    unit.pending_clarification = None

    attachment_text = _attachment_text_for_analysis(unit.capture)
    prior_context = _find_related_context(db, unit.authoritative_content)

    if unit.clarification_round < MAX_CLARIFICATION_ROUNDS:
        analysis = await reanalyze_unit(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
        semantic = dict(analysis.semantic)
        metadata_question = _apply_explicit_metadata(semantic, unit.explicit_metadata)
        unit.semantic = semantic
        question = metadata_question or analysis.clarification_question
        field = "metadata conflict" if metadata_question else analysis.clarification_field
        # Never re-ask about a field the user already declined or
        # couldn't answer -- a hard guarantee in code, not just a prompt
        # instruction (see docs/DECISIONS.md), since the whole point of
        # tracking the outcome is to make re-asking impossible, not just
        # discouraged.
        if question and field not in declined_or_unknown_fields:
            unit.clarification_round += 1
            unit.pending_clarification = {"question": question, "field": field, "round": unit.clarification_round}
            unit.status = "awaiting_clarification"
            db.commit()
            db.refresh(unit)
            return unit

    # No further question (or the round budget is exhausted): finalize
    # in the background, same as a unit that never needed clarification.
    _claim_processing(db, unit)
    db.refresh(unit)
    return unit


async def correct_unit(db: Session, llm: LLMProvider, unit: MemoryUnit, correction: str) -> MemoryUnit:
    """User-initiated correction: free-text input, re-running analysis --
    the same underlying mechanism as a clarification answer (append to
    `clarifications`, re-derive `semantic`/retrieval_text/embedding), but
    user-triggered at any time a unit is ready, not system-prompted.
    Explicitly does not consume the bounded system-clarification budget.
    Stays synchronous: the unit is already "ready" going in, so there's
    no "user is blocked waiting to see a question" pressure to relieve.
    """
    correction = correction.strip()
    if not correction:
        raise IngestionError("correction must not be empty")
    max_chars = get_settings().membase_max_text_chars
    if len(correction) > max_chars:
        raise IngestionError(f"correction exceeds max length of {max_chars} characters")
    if unit.status not in ("ready", "awaiting_clarification"):
        raise IngestionError(f"cannot correct a unit in status '{unit.status}'")

    clarifications = [*unit.clarifications, {
        "question": "(user correction)",
        "answer": correction,
        "field": "correction",
        "round": unit.clarification_round,
        "outcome": "answered",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "source": "user_correction",
    }]
    unit.clarifications = clarifications
    unit.pending_clarification = None

    attachment_text = _attachment_text_for_analysis(unit.capture)
    prior_context = _find_related_context(db, unit.authoritative_content)
    analysis = await reanalyze_unit(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
    semantic = dict(analysis.semantic)
    _apply_explicit_metadata(semantic, unit.explicit_metadata)
    unit.semantic = semantic
    unit.retrieval_text = build_retrieval_text(unit.authoritative_content, unit.clarifications, semantic)
    unit.searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, unit.retrieval_text)
    try:
        embedding = await llm.embed([unit.retrieval_text])
        unit.embedding = embedding[0]
        unit.embedding_model = llm.embedding_model_name
        unit.embedding_dim = llm.embedding_dim
    except LLMError:
        logger.warning("Embedding failed during correction for unit %s", unit.id)
    unit.status = "ready"
    db.commit()
    db.refresh(unit)
    await _detect_conflicts(db, llm, unit)
    return unit


async def retry_enrichment(db: Session, llm: LLMProvider, unit: MemoryUnit) -> MemoryUnit:
    """Recovery path for a unit stuck in draft/failed, or left in
    "processing" by a background task dropped by a server restart.
    Reclaims the processing lease only if it looks genuinely abandoned
    (no matching commit within membase_processing_lease_stale_seconds) --
    otherwise returns the unit as-is, since a *live* background task is
    presumably still going to finish it (see docs/DECISIONS.md
    "processing lease": this is the lease-based replacement for the row
    lock finish_unit_processing used to hold for the whole LLM call).
    """
    if unit.status == "processing":
        stale_after = get_settings().membase_processing_lease_stale_seconds
        started = unit.processing_started_at
        age = (dt.datetime.now(dt.UTC) - started).total_seconds() if started else float("inf")
        if age < stale_after:
            return unit  # still plausibly in flight; nothing to reclaim yet
        _claim_processing(db, unit)
        db.refresh(unit)
        return unit

    if unit.status not in ("draft", "failed"):
        return unit

    _claim_processing(db, unit)
    db.refresh(unit)
    return unit


async def _refresh_ready_units_with_attachment_excerpt(db: Session, llm: LLMProvider, capture: Capture) -> None:
    """Fold a short excerpt of the capture's combined attachment text into
    every already-"ready" sibling unit's retrieval/searchable text and
    re-embed -- without re-running full analysis (an unbounded number of
    attachment adds would otherwise mean an unbounded number of analysis
    calls over a capture's lifetime). Shared by add_attachment() (a
    posthoc file) and finish_image_captioning() (a caption that arrived
    after the unit was already embedded).
    """
    attachment_text = _attachment_text_for(capture)
    for unit in capture.units:
        if unit.status != "ready":
            continue
        base_retrieval_text = unit.retrieval_text.split(_ATTACHMENT_EXCERPT_MARKER, 1)[0]
        excerpt = attachment_text[:1000]
        retrieval_text = f"{base_retrieval_text}{_ATTACHMENT_EXCERPT_MARKER}{excerpt}" if excerpt else base_retrieval_text
        unit.retrieval_text = retrieval_text
        unit.searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, retrieval_text)
        try:
            embedding = await llm.embed([retrieval_text])
            unit.embedding = embedding[0]
            unit.embedding_model = llm.embedding_model_name
            unit.embedding_dim = llm.embedding_dim
        except LLMError:
            logger.warning("Re-embedding after attachment add failed for unit %s", unit.id)
    db.commit()


async def add_attachments(
    db: Session, llm: LLMProvider, capture: Capture, files: list[tuple[str, bytes]]
) -> Capture:
    """Attach one or more files to an already-persisted capture in one
    call (see docs/DECISIONS.md "Multi-file posthoc attachment").

    Validation and storage are atomic across the whole batch -- one
    unsupported/oversized file rolls back the entire call, exactly like
    create_capture()'s own multi-upload handling, rather than leaving a
    confusing mix of "some of the files you just tried to attach actually
    got attached, one didn't" for the caller to untangle. Best-effort
    derived processing (captioning, chunking) is still per-file after
    that, same tolerance as everywhere else in this pipeline: one file's
    captioning failure never blocks another's.

    The new attachments' text must become visible to *both* full-text
    and semantic (vector) retrieval -- re-running full analysis for
    every sibling unit on every attachment add would be an unbounded
    number of LLM calls over a capture's lifetime, so this takes a
    lighter path: rebuild searchable_text and re-embed from a
    retrieval_text that now also carries a short combined-attachment
    excerpt, without re-deriving aliases/definitions. Sibling units are
    re-embedded exactly ONCE after the whole batch, covering every newly
    attached file's text together -- not once per file, which would
    otherwise mean N redundant embedding calls for N files. Embedding
    failure here is tolerated the same way as during ingestion.
    """
    staged: list[tuple[Attachment, ProcessedFile]] = []
    for filename, data in files:
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
        staged.append((attachment, processed))
    db.commit()  # every file validated and durably stored, no LLM call yet

    for attachment, processed in staged:
        if is_uncaptioned_image(attachment):
            data_bytes = (get_settings().uploads_dir / processed.stored_path).read_bytes()
            description = await _describe_image(llm, data_bytes)
            if description:
                attachment.extracted_text = description
                attachment.extraction_status = "ok"
                processed.extracted_text = description
                db.commit()
        if should_chunk(processed.extracted_text):
            await _index_attachment_chunks(db, llm, attachment, processed)

    await _refresh_ready_units_with_attachment_excerpt(db, llm, capture)
    db.refresh(capture)
    return capture


async def add_attachment(db: Session, llm: LLMProvider, capture: Capture, filename: str, data: bytes) -> Capture:
    """Attach a single file -- thin wrapper around add_attachments() for
    the one-file case."""
    return await add_attachments(db, llm, capture, [(filename, data)])


def delete_capture(db: Session, capture: Capture) -> None:
    """Explicit user deletion: removes the capture and everything derived
    from it (units, attachments, chunks via DB cascade) plus the
    original files on disk. No automatic/relevance-based deletion exists
    anywhere in this codebase -- this is the only way a memory ever goes
    away, and it's always a direct user action.
    """
    capture_dir = get_settings().uploads_dir / str(capture.id)
    db.delete(capture)
    db.commit()
    shutil.rmtree(capture_dir, ignore_errors=True)


async def discard_unit(db: Session, unit: MemoryUnit) -> MemoryUnit:
    """Explicit user cancellation of a unit awaiting clarification.
    Deliberately a status flip, not a hard delete."""
    unit.status = "discarded"
    unit.pending_clarification = None
    db.commit()
    db.refresh(unit)
    return unit
