import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

EMBEDDING_DIM = 768


class Memory(Base):
    """A single stored memory.

    `caption` is the authoritative, user-provided text and is never
    overwritten by derived processing. Everything else on this row other
    than `id`/`created_at` is machine-derived and, in principle,
    recomputable from `caption` + its attachments.
    """

    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # --- authoritative user data ---
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # --- derived data (recomputable; never the sole record of a fact) ---
    derived: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    pending_clarification: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Denormalized text used to build the full-text index below. Populated
    # by the ingestion pipeline from caption + attachments' extracted text.
    searchable_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    fts = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(searchable_text, ''))", persisted=True),
        nullable=True,
    )

    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)

    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan", order_by="Attachment.created_at"
    )


class Attachment(Base):
    """An uploaded file attached to a memory.

    The original file is always kept on disk (`stored_path`); `extracted_text`
    is a best-effort derived representation and may be null/partial.
    """

    __tablename__ = "attachments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True
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

    memory: Mapped[Memory] = relationship(back_populates="attachments")
