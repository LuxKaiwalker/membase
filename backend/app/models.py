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
