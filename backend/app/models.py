import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
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

    # --- capture-level processing lease (see app/ingestion/pipeline.py
    # "process_capture_fully") -- pending | processing | done | failed.
    # Mirrors the per-unit processing_token/processing_started_at lease
    # on MemoryUnit, one level up: guards the single background task
    # that does ALL of this capture's LLM-dependent work (image
    # understanding, segmentation, clarification, enrichment, embedding)
    # against running twice concurrently.
    processing_status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    processing_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Staged here at Phase 0 (create_capture(), zero LLM calls) and read
    # by process_capture_fully() once it actually runs -- see
    # docs/DECISIONS.md "Explicit metadata authority" / "Disable
    # follow-up questions".
    allow_clarification: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    explicit_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

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
    app/ingestion/analysis.py). `clarifications` is an append-only list
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
    # Streamlining phase (see docs/DECISIONS.md "Sequential clarification"):
    # clarification is strictly one-question-at-a-time, re-evaluated after
    # every answer -- there is deliberately no pre-generated queue of
    # future questions anymore.

    # Short-lived processing lease (see docs/DECISIONS.md "processing
    # lease"): claimed atomically alongside status="processing", checked
    # again when the background step commits its result, so a long LLM
    # call never needs a held database row lock. A stale
    # processing_started_at (no matching commit within a reasonable
    # window) is what POST /retry treats as "safe to reclaim".
    processing_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # User-provided structured fields (location/date/people/organizations/
    # topics) -- optional, bypasses LLM inference entirely, and always
    # takes authority over whatever the model separately infers for the
    # same field (see docs/DECISIONS.md "Explicit metadata authority").
    explicit_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Lightweight conflict markers (see docs/DECISIONS.md "Contradiction
    # detection"): [{"unit_id": str, "note": str}]. Purely additive and
    # advisory -- nothing here ever deletes, supersedes, or hides the
    # conflicting unit; it only tells answer generation to surface the
    # disagreement explicitly instead of silently picking a side.
    potential_conflicts: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # --- membase structurer (see docs/DECISIONS.md "The 'membase
    # structurer'") -- a second, independent background pass that
    # deepens an already-"ready" unit's representation. Entirely
    # separate from `status` above: structure_status is NEVER read by
    # hybrid_search() and never gates searchability -- a unit sits at
    # "unstructured" indefinitely without being any less searchable.
    structure_status: Mapped[str] = mapped_column(Text, nullable=False, default="unstructured")
    # unstructured | structuring | structured | failed
    structure_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    structure_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    structured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Advisory cross-references to other memories the structurer judged
    # genuinely related -- a flat, bounded list of MemoryUnit id
    # strings, deliberately NOT a knowledge graph.
    related_unit_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Which pipeline code version last derived this unit's
    # attachment-text-dependent state (see app/ingestion/pipeline.py's
    # CURRENT_PIPELINE_VERSION and scripts/reprocess_memories.py, the
    # low-priority background pass that re-derives units left behind by
    # a fixed bug -- docs/DECISIONS.md "Reprocessing pipeline"). 0 for
    # every row that existed before this column did.
    pipeline_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

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

    # --- explainable reconstruction (see docs/ARCHITECTURE.md "Evidence
    # trail"). All three are snapshots, same discipline as final_sources: a
    # record of what this session actually did, which must stay readable even
    # after the underlying memories change.
    #
    # One entry per round: which retrieval strategies ran, how many candidates
    # came from direct matching vs. association expansion, what was ruled out
    # by the user's own negative cues, which question was asked and its
    # approximate expected information gain, and the budget in force. This is
    # what makes "why did I get this result?" answerable after the fact
    # instead of only while the round is running.
    strategy_trail: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Per-candidate structured evidence behind the final answer, including
    # contradicting items and each item's provenance level (original /
    # extracted / inferred / hypothesis). See app/retrieval/evidence.py.
    final_evidence: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Calibrated confidence CATEGORY (strong | moderate | weak |
    # insufficient), never a bare percentage -- nothing in this system
    # produces a calibrated probability, so showing one would be fabricated
    # precision. Definition lives in app/retrieval/evidence.py::confidence_for.
    final_confidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Why the system declined to answer, in plain language, when it did.
    # Distinct from the fixed abstention message shown to the user: this is
    # the diagnosable reason (nothing matched / ambiguity unresolved / the
    # model judged the evidence ungrounded).
    abstention_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Associative memory graph (see docs/ARCHITECTURE.md "Associative memory
# graph")
#
# This is the queryable graph that `MemoryUnit.related_unit_ids` deliberately
# was not: typed nodes, typed edges, a weight, and -- the part that makes it
# usable as *memory* rather than as a pile of inferences -- an `origin` column
# on every edge recording whether the user said it, an extraction step derived
# it, or the system merely computed it.
#
# `related_unit_ids` is NOT replaced. It stays exactly as it was (an advisory
# flat list the structurer writes) and is additionally projected into
# `unit_associations` with origin="inferred". Nothing that reads it today
# changes behaviour.
#
# Every row here is derived data in the authority hierarchy's lowest tier: it
# is rebuilt from `MemoryUnit.semantic`/`explicit_metadata` by
# `app/memory/graph.py:project_unit()` and can be dropped and regenerated
# without losing anything the user said. That is why projection is idempotent
# (delete-then-rebuild per unit) rather than incremental.

# Edge/link origin, in descending authority -- the same distinction the
# proposal's "distinguish explicitly established relationships from inferred
# associations" requires, as a column rather than a heuristic:
#   "explicit"  -- the user stated it themselves (Capture.explicit_metadata,
#                  or a clarification answer). Outranks everything below.
#   "extracted" -- an LLM extraction step read it out of the user's own words
#                  (MemoryUnit.semantic). Faithful to the text, but derived.
#   "inferred"  -- nobody stated it; the system computed it (entity
#                  co-occurrence, the structurer's related-memory judgement).
ORIGIN_EXPLICIT = "explicit"
ORIGIN_EXTRACTED = "extracted"
ORIGIN_INFERRED = "inferred"
EDGE_ORIGINS = (ORIGIN_EXPLICIT, ORIGIN_EXTRACTED, ORIGIN_INFERRED)

# Entity kinds. Deliberately a small closed set mapped 1:1 onto semantic
# fields the pipeline already extracts -- not an open-ended ontology, which
# would need curation this system explicitly does not want.
ENTITY_KINDS = ("person", "organization", "location", "topic", "concept", "object")


class MemoryEntity(Base):
    """A canonical thing memories are *about* -- a person, organization,
    location, topic, concept, or (for images) a visible object.

    Entities are the hub nodes that make associative recall possible: two
    memories that share no wording at all are one hop apart if they mention
    the same lake. They are resolved deterministically (normalized name +
    alias folding, see app/memory/entities.py), never by an LLM call, so the
    graph costs no inference on top of what ingestion already spends.

    An entity is never authoritative about anything. `display_name` is just
    the first spelling seen; the memory it came from remains the only source
    of truth, reachable via `unit_entity_links.provenance`.
    """

    __tablename__ = "memory_entities"
    __table_args__ = (
        # One row per (kind, normalized name). The same string can legitimately
        # be two different entities of different kinds ("Zurich" the location
        # vs. "Zurich" the insurer), so the kind is part of the identity.
        UniqueConstraint("kind", "normalized_name", name="uq_memory_entities_kind_name"),
        CheckConstraint(
            "kind IN ('person', 'organization', 'location', 'topic', 'concept', 'object')",
            name="ck_memory_entities_kind",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    # Deterministic normalization (casefold, whitespace collapse, possessive
    # and punctuation stripping) -- the join key. See app/memory/entities.py.
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    # The first surface spelling this entity was seen under, for display only.
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    # Other surface forms folded into this entity, as normalized strings.
    # Populated from semantic.aliases; used so "ETH" and "ETH Zurich" resolve
    # to one node instead of two.
    aliases: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # How many distinct memory units link to this entity. Denormalized from
    # unit_entity_links purely so the noise filters (which need a
    # document-frequency figure for every candidate entity) don't have to
    # aggregate the link table on every projection.
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class UnitEntityLink(Base):
    """A memory unit mentions an entity -- the unit<->entity edge.

    `role` records which semantic field the mention came from, which is what
    lets the graph explain itself ("connected because both happened at Lake
    Bled" vs. "both are about hiking") instead of just asserting a link.
    `origin` records whether the user stated it or an extraction step inferred
    it, so a user-supplied location can outrank a model-guessed one at
    retrieval time.
    """

    __tablename__ = "unit_entity_links"
    __table_args__ = (
        # One row per (unit, entity, role): the same entity legitimately
        # appears under two roles (named in `people` AND as a relation's
        # subject), and both are worth keeping as separate evidence.
        UniqueConstraint("unit_id", "entity_id", "role", name="uq_unit_entity_links_unit_entity_role"),
        CheckConstraint(
            "origin IN ('explicit', 'extracted', 'inferred')", name="ck_unit_entity_links_origin"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Which semantic field produced this mention: "people" | "organizations" |
    # "locations" | "topics" | "definition_term" | "relation_subject" |
    # "relation_object" | "visual_concept".
    role: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str] = mapped_column(Text, nullable=False, default=ORIGIN_EXTRACTED)
    # Role-derived confidence in this single mention (not a probability --
    # see app/memory/graph.py's ROLE_WEIGHTS for the fixed table and what each
    # value means).
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    # {"field": str, "excerpt": str, "surface": str, "predicate": str | None}
    # -- enough to show the user the words this link was derived from.
    provenance: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EntityRelation(Base):
    """A subject-predicate-object triple between two entities, asserted by one
    specific memory unit.

    This is `semantic.relations` finally given somewhere to live where it can
    be traversed. `unit_id` is mandatory and not nullable: a relation with no
    memory behind it would be exactly the free-floating "fact" the memory-
    integrity rules forbid. Deleting the memory deletes the relation.
    """

    __tablename__ = "entity_relations"
    __table_args__ = (
        UniqueConstraint(
            "unit_id",
            "subject_entity_id",
            "object_entity_id",
            "normalized_predicate",
            name="uq_entity_relations_unit_triple",
        ),
        CheckConstraint(
            "origin IN ('explicit', 'extracted', 'inferred')", name="ck_entity_relations_origin"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    object_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The predicate as the extraction produced it ("recommended a paper to"),
    # kept verbatim for display, plus a normalized form for deduplication.
    predicate: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_predicate: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str] = mapped_column(Text, nullable=False, default=ORIGIN_EXTRACTED)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    # Temporal attachment: the asserting unit's event date, carried onto the
    # relation so "who did I meet in 2023" is answerable by edge filtering.
    # `occurred_on` is NULL whenever the date couldn't be parsed -- never
    # back-filled with a guess or with created_at (see app/memory/temporal.py).
    occurred_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    # day | month | season | year | unknown -- how precisely occurred_on is
    # actually known. "unknown" with a NULL date is the honest default.
    date_precision: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    provenance: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class UnitAssociation(Base):
    """A weighted edge directly between two memory units.

    Stored once per unordered pair+kind with `source_unit_id < target_unit_id`
    by UUID ordering (see app/memory/graph.py's `_ordered_pair`), because these
    associations are symmetric -- "these two memories share a location" reads
    the same from either end. Traversal queries both columns.

    `explanation` is deterministic, generated text (never an LLM call) so the
    UI can answer "why are these two connected?" without a round trip.
    """

    __tablename__ = "unit_associations"
    __table_args__ = (
        UniqueConstraint("source_unit_id", "target_unit_id", "kind", name="uq_unit_associations_pair_kind"),
        CheckConstraint("source_unit_id <> target_unit_id", name="ck_unit_associations_no_self_loop"),
        CheckConstraint(
            "origin IN ('explicit', 'extracted', 'inferred')", name="ck_unit_associations_origin"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # "shared_entity"    -- both units mention the same entities (inferred)
    # "same_capture"     -- siblings from one submission (extracted: the user
    #                       really did write them together)
    # "structurer_link"  -- the membase structurer judged them related
    #                       (inferred, mirrors related_unit_ids)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str] = mapped_column(Text, nullable=False, default=ORIGIN_INFERRED)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # The entity ids justifying a "shared_entity" edge, so the explanation can
    # be recomputed and the edge can be filtered by entity kind.
    shared_entity_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Human-readable, deterministically generated: "Both mention Lake Bled
    # (location) and hiking (topic)."
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Episodic memory (see docs/ARCHITECTURE.md "Episodes")
#
# An episode is a collection of memories belonging to one experience, event, or
# period -- a holiday, a semester, a meeting, a particular evening. It is the
# unit people actually recall by ("that trip where..."), as distinct from the
# `Capture` (one submission) and the `MemoryUnit` (one retrievable fact).
#
# Two things this schema is built to express honestly, because the proposal
# requires both and they are easy to lose:
#
# **Uncertainty about dates and boundaries.** `started_on`/`ended_on` are
# nullable and travel with a `date_precision` and a `boundary_confidence`. An
# episode whose members carry no usable date has NULL dates and says so,
# rather than being filed under whenever the memories happened to be typed.
# Nothing here ever invents a date (see app/memory/temporal.py).
#
# **Overlap and multi-membership.** Membership is a separate table with no
# uniqueness on `unit_id` alone, so one memory can belong to several episodes
# (a conference talk belongs to both "the conference" and "the Berlin trip")
# and two episodes can cover overlapping spans without contradiction.
#
# Like the graph, episodes are a rebuildable projection: `app/memory/episodes.py`
# derives them deterministically from member dates and entity overlap, with no
# LLM call, so they cost nothing per capture and can be regenerated after a
# clustering change.


class Episode(Base):
    """One experience, event, or period that several memories belong to."""

    __tablename__ = "episodes"
    __table_args__ = (
        CheckConstraint(
            "boundary_confidence IN ('asserted', 'inferred', 'uncertain')",
            name="ck_episodes_boundary_confidence",
        ),
        CheckConstraint("origin IN ('explicit', 'extracted', 'inferred')", name="ck_episodes_origin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Deterministically composed from the episode's dominant entities and its
    # date span ("Lake Bled, Markus - July 2023"). Not model-written: see
    # app/memory/episodes.py for why a generated title was deliberately not
    # worth an LLM call per episode.
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # "derived" today. Reserved so a future model-written title can be
    # distinguished from a composed one rather than silently replacing it.
    title_source: Mapped[str] = mapped_column(Text, nullable=False, default="derived")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # NULL when no member memory carries a resolvable date. Never back-filled
    # from created_at -- "I don't know when this was" and "this happened the
    # day it was written down" are different claims.
    started_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    ended_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    # day | month | season | year | unknown -- the LOOSEST precision among
    # members, since a span is only as precise as its vaguest endpoint.
    date_precision: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    # asserted  -- every dated member's date came from the user directly
    # inferred  -- derived from model-extracted event dates
    # uncertain -- at least one member has no usable date at all, so the real
    #              boundaries may extend beyond what is recorded here
    boundary_confidence: Mapped[str] = mapped_column(Text, nullable=False, default="uncertain")

    origin: Mapped[str] = mapped_column(Text, nullable=False, default="inferred")
    # {"seed_unit_id": str, "rule": str, ...} -- which memory started this
    # episode and which clustering rule grouped it.
    provenance: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    memberships: Mapped[list["EpisodeMembership"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )


class EpisodeMembership(Base):
    """A memory belongs to an episode.

    Deliberately NOT unique on `unit_id`: a memory can legitimately belong to
    more than one episode, and forcing a single assignment would mean picking
    one arbitrarily and discarding a real relationship.
    """

    __tablename__ = "episode_memberships"
    __table_args__ = (
        UniqueConstraint("episode_id", "unit_id", name="uq_episode_memberships_episode_unit"),
        CheckConstraint(
            "origin IN ('explicit', 'extracted', 'inferred')", name="ck_episode_memberships_origin"
        ),
        CheckConstraint(
            "confidence IN ('certain', 'likely', 'uncertain')", name="ck_episode_memberships_confidence"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memory_units.id", ondelete="CASCADE"), nullable=False, index=True
    )
    origin: Mapped[str] = mapped_column(Text, nullable=False, default="inferred")
    # How sure the grouping is: "certain" for memories the user submitted
    # together, down to "uncertain" for an undated memory grouped purely on
    # entity overlap.
    confidence: Mapped[str] = mapped_column(Text, nullable=False, default="uncertain")
    # {"rule": str, "shared_entity_ids": [...], "explanation": str} -- a
    # deterministic, human-readable reason this memory is in this episode.
    provenance: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    episode: Mapped[Episode] = relationship(back_populates="memberships")


