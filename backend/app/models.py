import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

EMBEDDING_DIM = 768

# Authority hierarchy (highest to lowest; see docs/ARCHITECTURE.md and
# docs/DECISIONS.md): Capture.raw_text and Attachment originals > each
# MemoryUnit's authoritative_content > MemoryUnit.clarifications (still
# user-provided, just gathered after the fact) > MemoryUnit.semantic
# (LLM-derived) > retrieval_text/searchable_text/fts/embedding (fully
# regenerable indexes built from everything above).


class Capture(Base):
    """What the user actually submitted in one storage interaction.

    This is the "origin group": one capture may be decomposed into
    several independently retrievable MemoryUnits, all sharing
    `capture_id`. `raw_text` is authoritative and never rewritten.
    """

    __tablename__ = "captures"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    units: Mapped[list["MemoryUnit"]] = relationship(
        back_populates="capture", cascade="all, delete-orphan", order_by="MemoryUnit.created_at"
    )
    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="capture", cascade="all, delete-orphan", order_by="Attachment.created_at"
    )


class MemoryUnit(Base):
    """One independently retrievable piece of memory.

    `authoritative_content` is user-originated text (either a verbatim
    excerpt of the capture's raw_text chosen by segmentation, or the whole
    raw_text when segmentation didn't split/wasn't available -- see
    app/ingestion/segmentation.py). `clarifications` is an append-only list
    of verbatim user Q&A, also user-provided and never overwritten.
    Everything else (`semantic`, `retrieval_text`, `searchable_text`,
    `fts`, `embedding`) is LLM-derived or fully mechanically derived, and
    in principle regenerable from the two fields above.
    """

    __tablename__ = "memory_units"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("captures.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- authoritative (user-originated) ---
    authoritative_content: Mapped[str] = mapped_column(Text, nullable=False)
    clarifications: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # --- pipeline state (see app/ingestion/pipeline.py) ---
    status: Mapped[str] = mapped_column(Text, nullable=False, default="draft")
    pending_clarification: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Candidate questions from the current clarification round not yet
    # asked -- drained one at a time into pending_clarification as each is
    # answered (app/ingestion/clarify.py generates up to several at once;
    # the API/UI contract still only ever shows one at a time). Internal
    # bookkeeping only, not exposed via MemoryUnitOut.
    clarification_queue: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    clarification_round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- derived (LLM-derived semantic representation; regenerable) ---
    semantic: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    retrieval_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    searchable_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    fts = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(searchable_text, ''))", persisted=True),
        nullable=True,
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    capture: Mapped[Capture] = relationship(back_populates="units")


class Attachment(Base):
    """An uploaded file attached to a capture (and thus visible to every
    memory unit derived from it). The original file is always kept on
    disk (`stored_path`); `extracted_text` is best-effort and may be
    null/partial.
    """

    __tablename__ = "attachments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("captures.id", ondelete="CASCADE"), nullable=False, index=True
    )

    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_status: Mapped[str] = mapped_column(Text, nullable=False, default="skipped")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    capture: Mapped[Capture] = relationship(back_populates="attachments")
    chunks: Mapped[list["AttachmentChunk"]] = relationship(
        back_populates="attachment", cascade="all, delete-orphan", order_by="AttachmentChunk.chunk_index"
    )


class AttachmentChunk(Base):
    """A deterministically-chunked piece of an attachment's extracted
    text (spec section 27), independently full-text and vector indexed
    so a large document contributes more than just a short excerpt to
    retrieval. `capture_id` is denormalized from the parent attachment
    purely so retrieval can query chunks without a join. The original
    attachment (and its full extracted_text) remains authoritative;
    chunks are recomputable from it.
    """

    __tablename__ = "attachment_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attachment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("attachments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    capture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("captures.id", ondelete="CASCADE"), nullable=False, index=True
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    fts = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(text, ''))", persisted=True),
        nullable=True,
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    attachment: Mapped[Attachment] = relationship(back_populates="chunks")


class RecallSession(Base):
    """Retrieval-as-a-process state (see docs/ARCHITECTURE.md "Iterative
    recall"). A user's recall query is the start of retrieval, not a
    single stateless request: the session accumulates user-provided cues
    and agent search hypotheses across possibly several rounds before
    producing a grounded answer (or giving up explicitly).

    This is reasoning/session state, NOT memory. Nothing stored here
    (`cues`, `hypotheses`, `messages`) is ever written into `Capture` or
    `MemoryUnit` by this system -- the only way a fact discussed during
    recall becomes a durable memory is the user separately using the
    ordinary capture-creation flow. `final_sources`/`final_chunk_sources`
    are a snapshot of already-durable evidence (unit/chunk ids + the
    content as of answer time), not new data.

    Persisted in Postgres -- the app's only state store (see
    docs/DECISIONS.md) -- so an in-progress multi-turn recall survives an
    application restart, same durability guarantee as everything else.
    """

    __tablename__ = "recall_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)

    # active | needs_user_cue | success | exhausted | cancelled | error
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    current_question: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Conversation transcript: [{role: "user"|"system", text, created_at}].
    messages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # User-provided retrieval cues, accumulated across rounds:
    # [{text, confidence: certain|likely|uncertain, polarity: positive|negative, round, created_at}].
    # Search signal only -- never passed into answer-generation evidence.
    cues: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Agent-proposed search reformulations, kept visible for transparency
    # (spec: hypotheses must never be promoted to fact):
    # [{text, round, created_at}].
    hypotheses: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Union of every query string searched so far this session (original +
    # cues + reformulations), used only to avoid re-proposing an identical
    # reformulation -- never to skip re-searching the original query/cues.
    tried_queries: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Discriminating questions already asked, so the heuristic never repeats one.
    asked_questions: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    round_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Candidate count found in the most recent round, for a lightweight
    # "I found N related memories" progress indicator (no internal
    # reasoning/scores exposed).
    last_candidate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Snapshots (not live references) of the evidence a successful answer
    # was grounded in: [{id, capture_id, content, semantic, created_at,
    # vector_score, fts_score, fused_score}] / chunk equivalent.
    final_sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    final_chunk_sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
