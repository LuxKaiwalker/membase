import datetime as dt
import logging
import re
import shutil
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_sessionmaker
from app.ingestion.analysis import (
    Semantics,
    UnitAnalysis,
    build_retrieval_text,
    empty_semantics,
)
from app.ingestion.clarify import generate_question
from app.ingestion.enrichment import enrich_unit
from app.ingestion.segmentation import segment_capture
from app.ingestion.chunking import chunk_attachment
from app.ingestion.files import ProcessedFile, UnsupportedFileError, extract_pdf_pages, process_upload
from app.ingestion import structurer
from app.llm import get_llm_provider
from app.llm.base import LLMError, LLMProvider
# The graph projection is deliberately imported at module level, unlike
# app/retrieval/pipeline.py's hybrid_search (imported locally inside
# _finalize_structuring/_detect_conflicts to avoid an ingestion<->retrieval
# cycle). app/memory/ depends only on app/models + app/config, so there is no
# cycle to avoid here.
from app.memory.graph import project_unit_safely
from app.models import Attachment, AttachmentChunk, Capture, MemoryUnit

logger = logging.getLogger(__name__)

# Bounded rounds of *system*-initiated clarification -- stops the system
# from interrogating the user, not a limit on user-initiated correct_unit().
# See docs/DECISIONS.md "Sequential clarification".
MAX_CLARIFICATION_ROUNDS = 3

# Bump this whenever a change to attachment-text assembly, clarify, or
# enrichment would produce a materially different result for units
# already processed by the old code -- see docs/DECISIONS.md
# "Reprocessing pipeline" and scripts/reprocess_memories.py, the
# low-priority background pass that finds every MemoryUnit with
# pipeline_version < CURRENT_PIPELINE_VERSION and re-derives it. Every
# unit that successfully finishes derivation (_finalize_unit,
# correct_unit, reprocess_unit) is stamped with this value. 1: fixes
# _attachment_text_for_analysis() silently dropping a later
# attachment's text entirely when several images' combined text
# exceeded the old flat per-capture slice (see docs/user-todo.md).
# 2: enrichment now extracts `semantic.visual_concepts` (see
# app/ingestion/enrichment.py). Every photo stored before this ran has an
# empty list there, and so contributes no "object" entities to the
# association graph -- reprocessing is what makes "that photo with a lake
# and mountains behind it" reach memories captured before the field existed.
# 3: the attachment-text budget is now shared max-min fair across
# attachments (_allocate_part_budgets) instead of split evenly, so a
# capture whose attachments are unevenly sized no longer wastes the
# small one's unused share. Only changes anything for captures with 2+
# attachments of differing lengths; for a single attachment, or parts
# that all exceed an equal share, the assembled text is byte-identical
# and reprocessing is a no-op.
CURRENT_PIPELINE_VERSION = 3

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


def _attachment_parts(capture: Capture) -> list[str]:
    """Per-attachment text blocks, each already labeled ("Image N:" for
    images), before any joining or length-capping -- shared by
    _attachment_text_for() (uncapped join) and
    _attachment_text_for_analysis() (per-part capped join) so both stay
    in sync on how attachments are labeled/ordered.

    Each image attachment's text is labeled "Image N:" so a later
    enrichment call sees multiple photos as distinct, orderable sources
    (e.g. OD values from one photo and OS values from another) rather
    than one undifferentiated blob -- see docs/DECISIONS.md.

    An audio attachment's transcript is labeled the same way, for two
    reasons: the model must not read a transcript as the user's own typed
    words, and `visual_concepts` enrichment keys explicitly off the
    "Image N:" label (see app/ingestion/enrichment.py), so an unlabeled
    transcript full of concrete nouns would otherwise be mined for things
    "visible in the image" of a memory that has no image at all.
    """
    parts: list[str] = []
    image_n = 0
    audio_n = 0
    for a in capture.attachments:
        if not a.extracted_text:
            continue
        if a.content_type.startswith("image/"):
            image_n += 1
            parts.append(f"Image {image_n}:\n{a.extracted_text}")
        elif is_audio(a):
            audio_n += 1
            parts.append(f"Audio recording {audio_n} (transcript):\n{a.extracted_text}")
        else:
            parts.append(a.extracted_text)
    return parts


def _attachment_text_for(capture: Capture) -> str:
    """Full, uncapped combined attachment text -- used for the full-text
    search index (_build_searchable_text), which has no model context
    budget to protect. Never pass this directly into an LLM prompt; see
    _attachment_text_for_analysis() below.
    """
    return "\n\n".join(_attachment_parts(capture))


# A generation call's context window is shared between its input prompt
# and its output (see docs/DECISIONS.md "Chunk long captures before
# analysis" -- the same 4096-token budget that bug was about). Unlike
# that fix, which was about OUTPUT size (many units needing full
# semantics), an uncapped attachment_text is an INPUT-side version of
# the same risk: a large PDF, or several attachments concatenated with
# no limit, crowds out the budget the model needs for its own response,
# on every clarify/enrich call a capture's units ever make. Capped
# independently of should_chunk()'s indexing
# threshold -- that constant governs a different concern (whether
# AttachmentChunk rows are worth creating), not how much raw attachment
# text is safe to paste into one generation prompt.
MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS = 1500


# The marker _caption_and_extract_image() joins a free-text caption and
# its "Structured data:" block with (see that function). When a single
# attachment's own share of the budget isn't enough for both, the exact
# transcribed values in the structured block are what clarify/enrich
# actually need (see _extract_image_structured_data()'s docstring on why
# it's kept separate from the prose caption in the first place) -- so
# truncation below shortens the caption first and keeps the structured
# block whole, rather than cutting wherever the flat character count
# happens to land.
_STRUCTURED_DATA_MARKER = "\n\nStructured data:\n"


def _allocate_part_budgets(demands: list[int], total: int) -> list[int]:
    """Split `total` characters across parts, max-min fair: a part that
    needs less than an equal share keeps only what it needs and releases
    the surplus to the parts that want more, repeatedly, until every
    part still competing wants at least an equal share of what's left.

    An even split (the previous rule, `total // len(parts)`) wastes the
    difference whenever attachments are unevenly sized -- a 60-char
    "a blurry photo of a wall" caption next to a long PDF excerpt took
    the same 750-char share as the PDF and left ~690 chars of the budget
    unused while the PDF was cut short. Redistribution is pure
    arithmetic over text the caller already has, so it costs nothing and
    never grows the prompt past `total`.

    It does NOT help when total demand exceeds the budget: two parts
    wanting 1299 and 866 chars out of 1500 still get 750 each, because
    there is no surplus anywhere to move. Fitting that case would mean
    raising MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS, which trades against
    the output budget this cap exists to protect -- see docs/TODO.md.
    """
    budgets = [0] * len(demands)
    pending = list(range(len(demands)))
    pool = total
    while pending:
        share = pool // len(pending)
        if share < 1:
            break
        under = [i for i in pending if demands[i] <= share]
        if not under:
            # Everyone left wants more than an equal share: split what
            # remains evenly, handing the integer-division remainder to
            # the first few so the whole budget is actually used.
            remainder = pool - share * len(pending)
            for n, i in enumerate(pending):
                budgets[i] = share + (1 if n < remainder else 0)
            pending = []
            break
        for i in under:
            budgets[i] = demands[i]
            pool -= demands[i]
        satisfied = set(under)
        pending = [i for i in pending if i not in satisfied]
    # Every part gets at least one character. A part allocated zero
    # disappears from the prompt entirely, and an attachment silently
    # absent is the exact bug this whole code path exists to prevent --
    # better to overshoot a 1500-char cap by a few characters on a
    # capture with hundreds of attachments than to drop one without
    # saying so.
    return [max(b, 1) for b in budgets]


def _truncate_attachment_part(part: str, budget: int) -> str:
    if len(part) <= budget:
        return part
    marker_at = part.find(_STRUCTURED_DATA_MARKER)
    if marker_at == -1:
        return part[:budget]
    structured_block = part[marker_at:]
    if len(structured_block) >= budget:
        # The structured block alone doesn't fit either -- nothing left
        # to prioritize, fall back to a flat slice.
        return part[:budget]
    caption_block = part[:marker_at]
    return caption_block[: budget - len(structured_block)] + structured_block


def _attachment_text_for_analysis(capture: Capture) -> str:
    """Bounded combined attachment text for an LLM prompt -- see
    MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS above. Still gives the model
    something to resolve a vague reference against (e.g. an image
    caption, or a document's opening section); a full large document's
    *content* remains separately searchable via AttachmentChunk
    regardless of what's included here.

    The budget is shared across attachments (_allocate_part_budgets)
    and each one is truncated on its own boundary -- a flat slice of the
    whole concatenated blob instead spends the entire budget on the
    first attachment(s) and silently drops later ones, which previously
    made the model report a later image's description as "cut off" when
    it had actually just never been included at all (e.g. a two-photo
    eyeglasses prescription where the second photo's SPH/CYL/AXS/PSM
    values never made it into the prompt -- see docs/user-todo.md).
    """
    parts = _attachment_parts(capture)
    if not parts:
        return ""
    budgets = _allocate_part_budgets(
        [len(p) for p in parts], MAX_ATTACHMENT_TEXT_FOR_ANALYSIS_CHARS
    )
    return "\n\n".join(
        _truncate_attachment_part(part, budget) for part, budget in zip(parts, budgets)
    )


# --- cheap prior-memory lookup (see docs/DECISIONS.md "Prior memory
# context") -------------------------------------------------------------
# Deliberately NOT a semantic/vector search over the whole store -- a
# plain lexical lookup against already-ready units' retrieval_text for
# distinctive (capitalized, or multi-word proper-noun-shaped) terms in
# the new text. Context only, injected into the analysis prompt; never
# authoritative over what the user is currently saying (see
# docs/DECISIONS.md "Prior memory is context, not authority").
# [^\W\d_]: any Unicode letter (word char minus digits/underscore) --
# NOT [a-zA-Z], which is ASCII-only and would silently find zero
# candidate terms in capitalized names with accented characters
# ("Zürich", "Müller", "François"). The per-word uppercase-start check
# that [A-Z] used to enforce inline now happens in _candidate_terms
# below, since there's no portable Unicode "uppercase letter" character
# class in stdlib `re` (only the third-party `regex` package has one).
_CANDIDATE_TERM_RE = re.compile(r"\b[^\W\d_]{3,}(?:\s+[^\W\d_]{3,}){0,2}\b")
_PRIOR_CONTEXT_SNIPPET_CHARS = 300
_PRIOR_CONTEXT_LIMIT = 3


def _candidate_terms(raw_text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _CANDIDATE_TERM_RE.finditer(raw_text):
        term = m.group(0)
        if all(word[:1].isupper() for word in term.split()):
            seen.setdefault(term, None)
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


_IMAGE_STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "has_structured_data": {"type": "boolean"},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "value": {"type": "string"}},
                "required": ["label", "value"],
            },
        },
    },
    "required": ["has_structured_data", "fields"],
}

_IMAGE_STRUCTURED_PROMPT = (
    "Look for any form, table, label, prescription, measurement, or other "
    "data with named fields and values in this image (e.g. a glasses/contact "
    "lens prescription's OD/OS sphere/cylinder/axis values, a nutrition "
    "label, a receipt's line items, a gauge reading). If you find any, set "
    "has_structured_data=true and list each as a {label, value} pair, "
    "transcribing numbers and units exactly as shown -- do not round, infer, "
    "or normalize them. If nothing like that is present, set "
    "has_structured_data=false and return an empty list."
)


async def _extract_image_structured_data(llm: LLMProvider, data: bytes) -> str:
    """Returns a "Structured data:\\n..." text block if the image contains
    any labeled field/value data (forms, tables, prescriptions,
    measurements), or "" if none was found or the call failed.

    Kept separate from _describe_image()'s free-text caption -- a
    dedicated schema-constrained call is what actually gets exact
    numeric values transcribed into a form later enrichment can promote
    into `semantic.relations` (see app/ingestion/enrichment.py), instead
    of them staying buried, possibly garbled, in a general-purpose prose
    caption (see docs/DECISIONS.md).
    """
    try:
        raw = await llm.generate_structured(
            _IMAGE_STRUCTURED_PROMPT, schema=_IMAGE_STRUCTURED_SCHEMA, images=[data]
        )
    except LLMError:
        return ""
    if not raw.get("has_structured_data"):
        return ""
    fields = raw.get("fields")
    if not isinstance(fields, list):
        return ""
    lines = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        label, value = f.get("label"), f.get("value")
        if isinstance(label, str) and label.strip() and isinstance(value, str) and value.strip():
            lines.append(f"- {label.strip()}: {value.strip()}")
    return "Structured data:\n" + "\n".join(lines) if lines else ""


async def _caption_and_extract_image(llm: LLMProvider, data: bytes) -> str | None:
    """The free-text caption plus, when present, a structured label/value
    block -- this combined text is what gets written into
    Attachment.extracted_text, both for a capture's own images
    (_caption_one_image, called from process_capture_fully) and for a
    posthoc attachment (add_attachments())."""
    description = await _describe_image(llm, data)
    if description is None:
        return None
    structured = await _extract_image_structured_data(llm, data)
    return f"{description}\n\n{structured}" if structured else description


def is_uncaptioned_image(attachment: Attachment) -> bool:
    return attachment.content_type.startswith("image/") and attachment.extraction_status == "unsupported"


# --- audio transcription (see docs/DECISIONS.md) ------------------------
# Empirically established against THIS machine's stack (Ollama 0.40.3 +
# gemma4:12b, which advertises capabilities ["completion","vision","audio",
# "tools","thinking"]) by POSTing real synthesized-speech files to
# /api/chat directly, not by reading docs. Two findings drive the code
# below, and both are the kind that are invisible until measured:
#
# 1. Audio must be passed in the message's EXISTING `images` array -- the
#    same base64 field OllamaProvider._encode_images() already fills for
#    photos. `{"role":"user","content":...,"images":[<b64 wav>]}` returned
#    HTTP 200 with a real transcript. A separate `"audio":[<b64 wav>]`
#    field on the message is NOT supported by this Ollama version, and --
#    the dangerous part -- it does not error: the request returns HTTP 200
#    while the audio is silently DISCARDED. The model then answers from the
#    text prompt alone (prompt_eval_count collapsed from 164 to 36, i.e. no
#    media tokens at all, and the reply was "NO SPEECH"). So no provider
#    change is needed, and anyone "fixing" this to use an `audio` field
#    would get a plausible-looking but entirely hallucinated transcript.
#
# 2. Only WAV, MP3 and FLAC actually decode. OGG (both Opus and Vorbis),
#    M4A/AAC and WebM all came back HTTP 400 "Failed to load image or audio
#    file". Those are still accepted and stored (see _ALLOWED_MIMES in
#    app/ingestion/files.py -- never lose the original), but transcribing
#    them is a guaranteed-failing LLM call, so they are skipped outright
#    rather than attempted and logged. They keep extraction_status
#    "unsupported" and remain downloadable via
#    GET /api/attachments/{id}/file.
#
# Transcription *accuracy* is a separate matter and was NOT established:
# the only speech available here was espeak-ng TTS, on which gemma4
# demonstrably hears the audio (digits and "Thursday" came back exactly,
# and "four o'clock" was misheard as the phonetically-adjacent "Oracle" --
# a mishearing is itself proof the acoustic signal is being decoded) but
# also rewrites whole clauses it can't make out. Worse, a non-speech input
# (a pure 440Hz sine tone) produced a fluent, wholly invented paragraph
# instead of reporting silence. Hence the prompt below insists on a
# verbatim transcript and on NO_SPEECH for anything without speech, and
# _transcribe_audio() treats that sentinel as "nothing to store" -- a
# confabulated transcript indexed as a memory is worse than no transcript.
TRANSCRIBABLE_AUDIO_MIMES = frozenset(
    {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac", "audio/x-flac"}
)

# The sentinel the prompt asks for when there is no speech. Checked for
# exactly because of finding (2) above: left to its own devices this model
# invents dialogue for non-speech audio.
_NO_SPEECH_SENTINEL = "NO_SPEECH"

_AUDIO_TRANSCRIBE_PROMPT = (
    "Transcribe the speech in this audio recording verbatim, as plain text. "
    "Write only what is actually said -- do not summarize, translate, "
    "correct, or continue it, and do not add commentary. If a passage is "
    "unintelligible, write [inaudible] rather than guessing at the words. "
    f"If the recording contains no speech at all, reply with exactly "
    f"{_NO_SPEECH_SENTINEL} and nothing else. This transcript becomes the "
    "recording's only searchable representation, so accuracy matters more "
    "than fluency."
)


async def _transcribe_audio(llm: LLMProvider, data: bytes) -> str | None:
    """Transcribe an audio file via the generation model's native
    multimodal understanding -- the exact shape of _describe_image()
    above, including passing the bytes as `images=[data]` (finding 1
    above: that is the field Ollama reads audio from).

    Deliberately no second model: this GPU has 11GB VRAM and gemma4:12b
    alone measures ~8.7GB, so a dedicated speech model (Whisper or
    otherwise) has nowhere to live -- see CLAUDE.md "Environment facts".

    Best-effort in exactly the same way as image captioning: an LLMError
    -- which is also how an undecodable container surfaces, since
    Ollama's HTTP 400 becomes an LLMUnavailableError in
    OllamaProvider._post -- leaves the recording stored (already
    committed before this runs) but without a transcript, rather than
    failing the capture.
    """
    try:
        transcript = await llm.generate(_AUDIO_TRANSCRIBE_PROMPT, images=[data], temperature=0.0)
    except LLMError:
        logger.warning("Audio transcription failed; recording stored but not text-searchable")
        return None
    transcript = transcript.strip()
    # Matched loosely on purpose: asked for "NO_SPEECH", the live model was
    # observed replying "NO SPEECH" (space, no underscore), and a sentinel
    # that misses by one character would store the sentinel itself as a
    # memory's searchable transcript.
    if not transcript or transcript.strip(" .\"'").upper().replace(" ", "_") == _NO_SPEECH_SENTINEL:
        return None
    return transcript


def is_untranscribed_audio(attachment: Attachment) -> bool:
    """Audio we have not transcribed yet AND can actually transcribe. The
    mime gate is what keeps an OGG/M4A/WebM upload from costing a
    guaranteed-400 LLM call on every pass -- see finding (2) above."""
    return (
        attachment.extraction_status == "unsupported"
        and attachment.content_type in TRANSCRIBABLE_AUDIO_MIMES
    )


def is_audio(attachment: Attachment) -> bool:
    """Includes "video/webm", which is how libmagic reports an audio-only
    WebM recording -- see _ALLOWED_MIMES in app/ingestion/files.py."""
    return attachment.content_type.startswith("audio/") or attachment.content_type == "video/webm"


async def _transcribe_one_audio(db: Session, llm: LLMProvider, attachment: Attachment) -> None:
    """Transcribe one audio attachment in the foreground of
    process_capture_fully(), before segmentation/clarification/enrichment
    -- the same lease/commit discipline as _caption_one_image(), and for
    the same reason: a recording's contents must be able to inform all
    three. The transcript lands in Attachment.extracted_text, so it flows
    into searchable_text and chunking exactly like extracted PDF text
    already does, with no downstream changes at all.
    """
    try:
        data = (get_settings().uploads_dir / attachment.stored_path).read_bytes()
        transcript = await _transcribe_audio(llm, data)
        if not transcript:
            return
        attachment.extracted_text = transcript
        attachment.extraction_status = "ok"
        db.commit()
    except Exception:
        # Same catch-all rationale as _caption_one_image(): one
        # attachment's surprise failure (e.g. the file vanished from disk
        # between upload and now) must not abort the rest of
        # process_capture_fully for this capture's other attachments and
        # units -- this recording just stays untranscribed.
        logger.exception("Unexpected failure transcribing attachment %s", attachment.id)


async def _caption_one_image(db: Session, llm: LLMProvider, attachment: Attachment) -> None:
    """Caption (+ structured-extract) one image attachment, in the
    foreground of process_capture_fully() -- deliberately BEFORE
    segmentation/clarification/enrichment ever run for this capture, so
    an image's content can inform both (see docs/DECISIONS.md "Restore
    the multi-call ingestion pipeline": no longer an accepted "runs
    blind" trade-off now that all of this is backgrounded anyway).
    """
    try:
        data = (get_settings().uploads_dir / attachment.stored_path).read_bytes()
        # _caption_and_extract_image makes TWO generation calls (the
        # caption, then the structured-data extraction), and loading
        # this attachment opened a read transaction. Ending it here
        # keeps to app/db.py's "at most one LLM call per open
        # transaction" assumption -- with a vision call at tens of
        # seconds each, two inside one transaction is already a third of
        # the 180s budget. See docs/DECISIONS.md "One LLM call per open
        # transaction".
        db.commit()
        description = await _caption_and_extract_image(llm, data)
        if not description:
            return
        attachment.extracted_text = description
        attachment.extraction_status = "ok"
        db.commit()
    except Exception:
        # An unhandled exception here (e.g. the file went missing on
        # disk between upload and now) must not abort the rest of
        # process_capture_fully for the capture's other attachments/
        # units -- this attachment just stays uncaptioned.
        logger.exception("Unexpected failure captioning attachment %s", attachment.id)


async def finish_attachment_chunking(attachment_id: uuid.UUID, llm: LLMProvider | None = None) -> None:
    """Background-task entry point for chunk-level indexing of a
    non-image attachment, scheduled right after it's created (an image
    attachment's chunking decision instead happens inline inside
    process_capture_fully(), since it depends on a caption that doesn't
    exist yet at schedule time -- see should_chunk below).

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
        try:
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
        except Exception:
            # Fire-and-forget background task -- see the matching guard
            # in finish_image_captioning above for why this needs its
            # own catch-all rather than letting a surprise failure (e.g.
            # the file missing on disk) vanish with no record anywhere.
            logger.exception("Unexpected failure chunking attachment %s", attachment_id)
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
        values: dict = {
            "retrieval_text": retrieval_text,
            "searchable_text": searchable_text,
            "status": "ready",
            "pipeline_version": CURRENT_PIPELINE_VERSION,
        }
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
    if committed:
        # Project into the associative graph only after the unit is durably
        # "ready" -- the graph is an index over committed state, and a node
        # for a unit whose commit lost its lease race would be an index entry
        # for a result that was dropped. Deliberately outside the
        # embedding-present branch above: a unit whose embedding call failed
        # is still fully searchable via full-text and still belongs in the
        # graph (its entities came from `semantic`, not from the vector).
        db.refresh(unit)
        project_unit_safely(db, unit)


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
    try:
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

        # End the read transaction hybrid_search's SELECTs opened before
        # the structurer's own generation call. Otherwise one transaction
        # spans the query embedding AND the structuring call(s) -- the
        # same violation of app/db.py's "at most one LLM call per open
        # transaction" assumption that cost a seed run three memories in
        # process_capture_fully (see docs/DECISIONS.md "One LLM call per
        # open transaction"). Nothing is pending: the lease was committed
        # by _claim_structuring above, `candidates` is already
        # materialized into plain strings, and the session is
        # expire_on_commit=False so `unit`'s attributes survive this.
        db.commit()

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
    except Exception:
        # Mirrors _finalize_unit's outer guard -- without this, an
        # unexpected failure here (as opposed to the LLMErrors already
        # handled above) would leave the unit's structure_token set
        # forever: the lease is never released, so it's stuck in
        # structure_status="structuring" with no automatic recovery and
        # no record of what went wrong.
        logger.exception("Unexpected failure structuring unit %s", unit.id)
        _commit_structuring_if_token_matches(db, unit.id, token, {"structure_status": "failed"})
        return

    if _commit_structuring_if_token_matches(db, unit.id, token, values):
        # Re-project: structuring replaced `semantic` wholesale and wrote
        # `related_unit_ids`, so both the entity links and the association
        # edges derived from this unit are now stale. Projection being
        # idempotent is what makes "just run it again" the correct response to
        # any change in a unit's semantics.
        db.refresh(unit)
        project_unit_safely(db, unit)


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
    *,
    text: str,
    uploads: list[tuple[str, bytes]],
    explicit_metadata: dict | None = None,
    allow_clarification: bool = True,
) -> Capture:
    """Phase 0 ONLY: persist the capture's raw text and attachments, with
    zero LLM calls, so POST /api/captures can return an instant "saved"
    acknowledgement as soon as the data has reached the server (see
    docs/DECISIONS.md "Restore the multi-call ingestion pipeline"). No
    `llm` parameter -- nothing here ever calls the model. All
    LLM-dependent work (segmentation, clarification, enrichment, image
    understanding, embedding) is scheduled separately as
    process_capture_fully(), run entirely in a background task (see
    app/api/captures.py::create()). `capture.units` is always `[]` on
    return.
    """
    text = text.strip()
    if not text:
        raise IngestionError("text must not be empty")
    max_chars = get_settings().membase_max_text_chars
    if len(text) > max_chars:
        raise IngestionError(f"text exceeds max length of {max_chars} characters")

    capture = Capture(
        raw_text=text,
        explicit_metadata=explicit_metadata or {},
        allow_clarification=allow_clarification,
    )
    db.add(capture)
    db.flush()

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
    db.commit()  # capture + attachments durably saved, no LLM call yet

    db.refresh(capture)
    return capture


# --- capture-level processing lease (see docs/DECISIONS.md "Restore the
# multi-call ingestion pipeline") -- same claim-token-commit pattern as
# the per-unit lease above (_claim_processing/_commit_if_token_matches),
# one level up: guards process_capture_fully() against being run twice
# concurrently for the same capture (a duplicate schedule, or a manual
# /retry racing a still-live background task).

def _claim_capture_processing(db: Session, capture: Capture) -> str:
    token = uuid.uuid4().hex
    capture.processing_status = "processing"
    capture.processing_token = token
    capture.processing_started_at = dt.datetime.now(dt.UTC)
    db.commit()
    return token


def _commit_capture_if_token_matches(db: Session, capture_id: uuid.UUID, token: str, values: dict) -> bool:
    result = db.execute(
        update(Capture)
        .where(Capture.id == capture_id, Capture.processing_token == token)
        .values(**values, processing_token=None, processing_started_at=None)
    )
    db.commit()
    return result.rowcount > 0


async def _analyze_one_unit(
    llm: LLMProvider, content: str, attachment_text: str, prior_context: list[str], allow_clarification: bool
) -> UnitAnalysis:
    """Dedicated clarify call, then -- only if no question will actually
    be surfaced -- the dedicated enrichment call. Mirrors the pre-collapse
    ordering (see docs/DECISIONS.md): a unit about to be blocked on a
    question never wastes an enrichment call first. `allow_clarification`
    gates whether clarify.generate_question() is even called, not just
    whether its result is used -- when follow-up questions are disabled
    for this capture, enrichment must still run unconditionally.
    """
    question = None
    if allow_clarification:
        question = await generate_question(llm, content, [], attachment_text, prior_context)
    if question:
        return UnitAnalysis(
            content=content, semantic=empty_semantics("ok"),
            clarification_question=question["question"], clarification_field=question["field"],
        )
    semantic = await enrich_unit(llm, content, [], attachment_text, prior_context)
    return UnitAnalysis(content=content, semantic=semantic)


async def process_capture_fully(
    capture_id: uuid.UUID, llm: LLMProvider | None = None, *, auto_structure: bool = True
) -> None:
    """Background-task entry point scheduled exactly once per capture
    (see app/api/captures.py::create()) -- the ONE place all of a
    capture's LLM-dependent work happens: image understanding,
    segmentation, clarification decisions, enrichment, embedding, and
    (for units that don't need a question) the membase structurer's
    deeper pass. Nothing here blocks the HTTP response that created the
    capture -- create_capture() already returned before this ever runs.

    Ordering within this one coroutine reproduces the ordering guarantee
    BackgroundTasks used to provide across several separately-scheduled
    tasks (sequential registration order) -- here it's just sequential
    `await`s in one function, nothing to schedule. In particular, units
    don't exist yet when this is scheduled (segmentation hasn't run),
    so captures.py can no longer pre-schedule a per-unit finish_structuring
    task the way it used to -- this function chaining it internally,
    right after a unit reaches "ready", is what replaces that.

    `auto_structure=False` is a testability knob only (see
    tests/helpers.py's create_capture_and_finish_unstructured) -- there
    is no production caller that ever wants a unit to stay unstructured.
    """
    db = get_sessionmaker()()
    try:
        if llm is None:
            llm = get_llm_provider()
        capture = db.get(Capture, capture_id)
        if capture is None or capture.processing_status not in ("pending", "failed"):
            return
        token = _claim_capture_processing(db, capture)

        try:
            # Image understanding runs first and in full, before
            # anything downstream ever sees this capture's text -- no
            # longer an accepted "runs blind" trade-off now that this
            # whole function is backgrounded anyway (see
            # docs/DECISIONS.md).
            for attachment in capture.attachments:
                if is_uncaptioned_image(attachment):
                    await _caption_one_image(db, llm, attachment)
                elif is_untranscribed_audio(attachment):
                    # Same position in the sequence and the same reason as
                    # captioning: a recording's transcript has to exist
                    # before segmentation/clarify/enrich ever see this
                    # capture's text.
                    await _transcribe_one_audio(db, llm, attachment)
            db.refresh(capture)

            # Only an image's chunking decision belongs here -- it's the
            # one case that depends on state only this function just
            # produced (a caption that didn't exist at schedule time). A
            # non-image attachment's extracted_text already existed at
            # Phase 0, so its chunking is scheduled independently, right
            # alongside capture creation (see app/api/captures.py) --
            # redoing it here too would just be a wasted duplicate
            # _index_attachment_chunks() call (idempotent, but pointless).
            # Audio belongs in this same inline pass for exactly the
            # reason the comment above gives for images: a transcript, like
            # a caption, did not exist when captures.py scheduled the
            # independent non-image chunking task, so that task saw an
            # empty extracted_text and skipped. A long recording's
            # transcript would otherwise never be chunk-indexed.
            for attachment in capture.attachments:
                if (attachment.content_type.startswith("image/") or is_audio(attachment)) and should_chunk(attachment.extracted_text):
                    processed = ProcessedFile(
                        stored_path=attachment.stored_path, content_type=attachment.content_type,
                        size_bytes=attachment.size_bytes, extracted_text=attachment.extracted_text,
                        extraction_status=attachment.extraction_status, pages=None,
                    )
                    await _index_attachment_chunks(db, llm, attachment, processed)

            attachment_text = _attachment_text_for_analysis(capture)
            prior_context = _find_related_context(db, capture.raw_text)

            # End the read transaction those two lines opened before any
            # of the generation calls below. app/db.py sets
            # idle_in_transaction_session_timeout=180000 on every
            # connection, on the stated assumption that this pipeline
            # makes "at most one LLM call per open transaction" -- and
            # everything above holds to that (each of
            # _caption_one_image / _transcribe_one_audio /
            # _index_attachment_chunks commits its own work), but this
            # stretch did not: segmentation plus one analysis call per
            # proposed unit all ran inside the transaction opened by the
            # reads above, with the first commit only after db.add_all()
            # below. Postgres then killed the connection mid-pipeline,
            # which surfaced as an IdleInTransactionSessionTimeout out of
            # the commit at the end of a capture that had already cost
            # minutes of model time. Hit for real seeding a corpus of
            # image-bearing captures (scripts/seed_demo.py): a vision
            # caption makes attachment_text long, which makes every
            # analysis prompt slower, which is exactly when the sum
            # crosses 180s. Nothing between here and db.add_all() writes,
            # so there is nothing to lose by committing now -- and the
            # work already done (captions, transcripts, chunks) stays
            # durable either way.
            db.commit()

            proposed_contents = await segment_capture(llm, capture.raw_text)
            analyses: list[UnitAnalysis] = []
            for content in proposed_contents:
                try:
                    analyses.append(
                        await _analyze_one_unit(llm, content, attachment_text, prior_context, capture.allow_clarification)
                    )
                except Exception:
                    logger.exception("Unexpected failure analyzing a proposed unit for capture %s", capture.id)
                    analyses.append(UnitAnalysis(content=content, semantic=empty_semantics("unavailable")))

            units = [
                _unit_from_analysis(capture.id, a, capture.explicit_metadata, capture.allow_clarification)
                for a in analyses
            ]
            db.add_all(units)
            db.commit()  # every unit's authoritative_content + analysis durably saved

            for unit in units:
                if unit.status == "awaiting_clarification":
                    continue
                unit_token = _claim_processing(db, unit)
                await _finalize_unit(db, llm, unit, unit_token, attachment_text)
                db.refresh(unit)
                if auto_structure and unit.status == "ready":
                    await finish_structuring(unit.id, llm)
        except Exception:
            # Phase-0 data (raw_text, attachments) was already committed
            # by create_capture() long before this ever ran, so a crash
            # here never loses the user's words -- only this capture's
            # derived processing is left incomplete, retryable via
            # retry_capture_processing().
            logger.exception("Unexpected failure processing capture %s", capture_id)
            _commit_capture_if_token_matches(db, capture_id, token, {"processing_status": "failed"})
            return

        _commit_capture_if_token_matches(db, capture_id, token, {"processing_status": "done"})
    finally:
        db.close()


async def retry_capture_processing(db: Session, llm: LLMProvider, capture: Capture) -> Capture:
    """Recovery path for a capture whose process_capture_fully() never
    finished (e.g. a server restart dropped the background task) or
    failed outright. Mirrors retry_enrichment()'s staleness check for
    the per-unit lease.
    """
    if capture.processing_status == "done":
        return capture

    if capture.processing_status == "processing":
        stale_after = get_settings().membase_processing_lease_stale_seconds
        started = capture.processing_started_at
        age = (dt.datetime.now(dt.UTC) - started).total_seconds() if started else float("inf")
        if age < stale_after:
            return capture  # still plausibly in flight; nothing to reclaim yet

    if not capture.units:
        # Segmentation never completed (or never started) -- nothing
        # was ever committed past the capture-level lease claim, so
        # it's safe to run the whole pipeline again from scratch.
        capture.processing_status = "pending"
        capture.processing_token = None
        capture.processing_started_at = None
        db.commit()
        await process_capture_fully(capture.id, llm)
        db.refresh(capture)
        return capture

    # Segmentation already succeeded -- never re-segment (that would
    # duplicate units). Delegate to the existing per-unit recovery for
    # any unit stuck in draft/processing/failed instead.
    attachment_text = _attachment_text_for(capture)
    for unit in capture.units:
        if unit.status not in ("draft", "failed", "processing"):
            continue
        updated = await retry_enrichment(db, llm, unit)
        if updated.status == "processing" and updated.processing_token:
            unit_token = updated.processing_token
            await _finalize_unit(db, llm, updated, unit_token, attachment_text)
            db.refresh(updated)
            if updated.status == "ready":
                await finish_structuring(updated.id, llm)

    db.refresh(capture)
    if all(u.status in ("ready", "awaiting_clarification", "discarded") for u in capture.units):
        capture.processing_status = "done"
        db.commit()
    return capture


async def resolve_clarification(db: Session, llm: LLMProvider, unit: MemoryUnit, answer: str | None) -> MemoryUnit:
    """Answering is optional (soft client-side guard only, see
    docs/DECISIONS.md) -- a blank/omitted answer skips this question:
    nothing is added to `clarifications`. A non-blank answer is
    classified (answered/unknown/declined, see _classify_outcome) and
    recorded either way.

    Then, mirroring process_capture_fully()'s own ordering: a dedicated
    clarify.generate_question() call decides whether a FURTHER question
    is still warranted given everything answered so far; only once no
    further question will actually be surfaced (none proposed, the round
    budget is exhausted, or the only candidate was already declined/
    unknown) does enrichment.enrich_unit() run to re-derive the real
    semantic fields -- so a unit can never reach "processing" with a
    unit.semantic that was never actually enriched.
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

    question = None
    if unit.clarification_round < MAX_CLARIFICATION_ROUNDS:
        question = await generate_question(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
    # Never re-ask about a field the user already declined or couldn't
    # answer -- a hard guarantee in code, not just a prompt instruction
    # (see docs/DECISIONS.md), since the whole point of tracking the
    # outcome is to make re-asking impossible, not just discouraged.
    if question and question["field"] not in declined_or_unknown_fields:
        unit.clarification_round += 1
        unit.pending_clarification = {"question": question["question"], "field": question["field"], "round": unit.clarification_round}
        unit.status = "awaiting_clarification"
        db.commit()
        db.refresh(unit)
        return unit

    semantic = await enrich_unit(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
    semantic = dict(semantic)
    metadata_question = _apply_explicit_metadata(semantic, unit.explicit_metadata)
    unit.semantic = semantic
    if metadata_question:
        unit.clarification_round += 1
        unit.pending_clarification = {"question": metadata_question, "field": "metadata conflict", "round": unit.clarification_round}
        unit.status = "awaiting_clarification"
        db.commit()
        db.refresh(unit)
        return unit

    # No further question: finalize in the background, same as a unit
    # that never needed clarification.
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
    semantic = await enrich_unit(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
    semantic = dict(semantic)
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
    unit.pipeline_version = CURRENT_PIPELINE_VERSION
    db.commit()
    db.refresh(unit)
    await _detect_conflicts(db, llm, unit)
    project_unit_safely(db, unit)
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


# --- reprocessing pipeline (see docs/DECISIONS.md "Reprocessing
# pipeline" and scripts/reprocess_memories.py) -- a low-priority,
# manually/cron-triggered sweep over units with
# pipeline_version < CURRENT_PIPELINE_VERSION, re-deriving whatever a
# fixed pipeline bug would have gotten wrong. Two different policies by
# current status, both built around one rule: reprocessing must never
# make a unit LESS available than it already is.
#
# - "ready" units are already searchable. Mirrors the membase
#   structurer's own invariant (see "membase structurer" above): never
#   re-gate status, never call clarify.generate_question() here. Only
#   enrich_unit() reruns, so corrected attachment text can promote
#   values it previously missed into `semantic` -- the same thing
#   correct_unit() does, minus appending a synthetic "(user
#   correction)" entry to `clarifications`, which would misrepresent
#   provenance for something the user never did.
# - "awaiting_clarification" units are already excluded from search, so
#   there is no availability to protect -- any progress here is pure
#   upside. clarify.generate_question() reruns with the corrected
#   attachment text: if it now finds nothing to ask, the unit resolves
#   itself straight to "ready" with no user involvement at all (the
#   eyeglasses-prescription bug's actual fix path -- see
#   docs/user-todo.md). If a question is still warranted, it replaces
#   the single `pending_clarification` slot in place -- never appended
#   anywhere -- so a unit that already had exactly one open question
#   still has exactly one afterwards. A huge backlog of stuck units can
#   therefore never multiply the total number of questions a user sees;
#   it can only shrink it (auto-resolutions) or improve the wording of
#   what's already pending. `clarification_round` is deliberately left
#   untouched -- this is the system re-deriving the SAME open round
#   under corrected inputs, not a new round the user answered into.
async def reprocess_unit(db: Session, llm: LLMProvider, unit: MemoryUnit) -> MemoryUnit:
    if unit.status not in ("ready", "awaiting_clarification"):
        return unit  # draft/processing/failed/discarded: not this pass's job

    attachment_text = _attachment_text_for_analysis(unit.capture)
    prior_context = _find_related_context(db, unit.authoritative_content)

    if unit.status == "awaiting_clarification":
        declined_or_unknown_fields = {
            c["field"] for c in unit.clarifications if c.get("outcome") in ("declined", "unknown")
        }
        question = await generate_question(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
        if question and question["field"] not in declined_or_unknown_fields:
            unit.pending_clarification = {
                "question": question["question"], "field": question["field"], "round": unit.clarification_round,
            }
            unit.pipeline_version = CURRENT_PIPELINE_VERSION
            db.commit()
            db.refresh(unit)
            return unit

        # The fix removed the only remaining reason to ask -- resolve
        # straight through, same as resolve_clarification()'s own
        # no-further-question branch.
        unit.pending_clarification = None
        semantic = await enrich_unit(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
        semantic = dict(semantic)
        _apply_explicit_metadata(semantic, unit.explicit_metadata)
        unit.semantic = semantic
        token = _claim_processing(db, unit)
        await _finalize_unit(db, llm, unit, token, attachment_text)
        db.refresh(unit)
        return unit

    # status == "ready": silent re-enrichment only, never clarify.
    semantic = await enrich_unit(llm, unit.authoritative_content, unit.clarifications, attachment_text, prior_context)
    semantic = dict(semantic)
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
        logger.warning("Embedding failed while reprocessing unit %s", unit.id)
    unit.pipeline_version = CURRENT_PIPELINE_VERSION
    db.commit()
    db.refresh(unit)
    project_unit_safely(db, unit)
    return unit


async def _refresh_ready_units_with_attachment_excerpt(db: Session, llm: LLMProvider, capture: Capture) -> None:
    """Fold a short excerpt of the capture's combined attachment text into
    every already-"ready" sibling unit's retrieval/searchable text and
    re-embed -- without re-running full analysis (an unbounded number of
    attachment adds would otherwise mean an unbounded number of analysis
    calls over a capture's lifetime). Used by add_attachments() for a
    posthoc file added after the capture's units are already "ready" --
    a capture's own images are captioned before segmentation ever runs
    (see process_capture_fully), so their content is already part of
    the units' semantics by the time they're first embedded and never
    needs this lighter excerpt-only path.
    """
    attachment_text = _attachment_text_for(capture)
    ready_units = [u for u in capture.units if u.status == "ready"]
    if not ready_units:
        return

    excerpt = attachment_text[:1000]
    for unit in ready_units:
        base_retrieval_text = unit.retrieval_text.split(_ATTACHMENT_EXCERPT_MARKER, 1)[0]
        retrieval_text = f"{base_retrieval_text}{_ATTACHMENT_EXCERPT_MARKER}{excerpt}" if excerpt else base_retrieval_text
        unit.retrieval_text = retrieval_text
        unit.searchable_text = _build_searchable_text(unit.authoritative_content, attachment_text, retrieval_text)

    # One batched embed call across every ready sibling, not one call
    # per unit -- consistent with the rest of this codebase's explicit
    # discipline around minimizing LLM round-trips. Trade-off: a batch
    # failure leaves every sibling's embedding stale rather than just
    # the one that would have failed individually, which is acceptable
    # here since embedding failures are rare and already non-fatal
    # (the unit stays searchable via full-text, just not vector search).
    try:
        embeddings = await llm.embed([u.retrieval_text for u in ready_units])
        for unit, embedding in zip(ready_units, embeddings, strict=True):
            unit.embedding = embedding
            unit.embedding_model = llm.embedding_model_name
            unit.embedding_dim = llm.embedding_dim
    except LLMError:
        logger.warning(
            "Re-embedding after attachment add failed for %d unit(s) on capture %s", len(ready_units), capture.id
        )
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
            description = await _caption_and_extract_image(llm, data_bytes)
            if description:
                attachment.extracted_text = description
                attachment.extraction_status = "ok"
                processed.extracted_text = description
                db.commit()
        elif is_untranscribed_audio(attachment):
            data_bytes = (get_settings().uploads_dir / processed.stored_path).read_bytes()
            transcript = await _transcribe_audio(llm, data_bytes)
            if transcript:
                attachment.extracted_text = transcript
                attachment.extraction_status = "ok"
                # `processed` is kept in step with the row so the
                # should_chunk() call just below sees the transcript --
                # exactly as the captioning branch does for a caption.
                processed.extracted_text = transcript
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
    from app.memory.episodes import refresh_episode
    from app.models import EpisodeMembership

    capture_dir = get_settings().uploads_dir / str(capture.id)
    # Which episodes this capture's memories belonged to, captured BEFORE the
    # delete: the DB cascade removes the membership rows, but an Episode row
    # has no FK to anything and survives. Without this, deleting the last
    # memory of an episode leaves a phantom timeline entry -- an episode with
    # zero members still listed by GET /api/episodes, still titled after the
    # memories that no longer exist. refresh_episode() deletes an episode left
    # with no ready members, so re-running it for each affected episode after
    # the cascade is what actually removes it.
    affected_episode_ids = set(
        db.execute(
            select(EpisodeMembership.episode_id)
            .join(MemoryUnit, MemoryUnit.id == EpisodeMembership.unit_id)
            .where(MemoryUnit.capture_id == capture.id)
        ).scalars()
    )
    db.delete(capture)
    db.commit()
    for episode_id in affected_episode_ids:
        refresh_episode(db, episode_id)
    db.commit()
    shutil.rmtree(capture_dir, ignore_errors=True)


def discard_unit(db: Session, unit: MemoryUnit) -> MemoryUnit:
    """Explicit user cancellation of a unit awaiting clarification.
    Deliberately a status flip, not a hard delete. Plain `def`, not
    `async def` -- no `await` inside."""
    unit.status = "discarded"
    unit.pending_clarification = None
    db.commit()
    db.refresh(unit)
    # A discarded unit must leave the graph too, or the association edges it
    # left behind would keep routing associative retrieval through content the
    # user explicitly cancelled -- the exact gap the status filter in
    # hybrid_search() exists to close. project_unit() on a non-ready unit
    # removes and rebuilds nothing, which is precisely what's wanted here.
    project_unit_safely(db, unit)
    return unit
