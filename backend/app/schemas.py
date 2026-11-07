import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    content_type: str
    size_bytes: int
    extraction_status: str


class MemoryUnitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    capture_id: uuid.UUID
    authoritative_content: str
    status: str
    clarifications: list
    pending_clarification: dict | None
    semantic: dict
    explicit_metadata: dict
    potential_conflicts: list
    structure_status: str
    structured_at: datetime | None
    related_unit_ids: list
    created_at: datetime


class CaptureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    raw_text: str
    created_at: datetime
    # pending | processing | done | failed -- lets the client tell "not
    # processed yet" (units still []) apart from "genuinely zero units"
    # (see app/ingestion/pipeline.py's process_capture_fully).
    processing_status: str
    attachments: list[AttachmentOut]
    units: list[MemoryUnitOut]


class ClarifyRequest(BaseModel):
    # Answering is optional (see docs/DECISIONS.md "obligatory" setting) --
    # omitted/blank means "skip this question", enforced only as a soft
    # client-side guard, not here.
    answer: str | None = None


class CorrectRequest(BaseModel):
    correction: str


class RecallRequest(BaseModel):
    query: str


class SourceOut(BaseModel):
    id: uuid.UUID
    capture_id: uuid.UUID
    content: str
    semantic: dict
    created_at: datetime
    vector_score: float | None
    fts_score: float | None
    fused_score: float
    # The source unit's capture's own attachments -- lets the Recall/
    # Remember UI offer a view/download link for a retrieved memory's
    # original files, not just document-chunk excerpts (see
    # app/retrieval/session.py's _snapshot_sources).
    attachments: list[AttachmentOut] = []


class ChunkSourceOut(BaseModel):
    id: uuid.UUID
    capture_id: uuid.UUID
    attachment_id: uuid.UUID
    attachment_filename: str
    page_number: int | None
    excerpt: str
    vector_score: float | None
    fts_score: float | None
    fused_score: float


class RecallResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[SourceOut]
    chunk_sources: list[ChunkSourceOut] = []


class RecallSessionMessageRequest(BaseModel):
    text: str


class SaveAsMemoryRequest(BaseModel):
    # Must match something the user actually said in this session (a
    # message or an extracted cue) -- enforced server-side, never an
    # arbitrary string or an agent-generated hypothesis/question (see
    # docs/DECISIONS.md "Save as memory").
    text: str


class MessageOut(BaseModel):
    role: str
    text: str
    created_at: str


class CueOut(BaseModel):
    text: str
    confidence: str
    polarity: str
    round: int
    created_at: str


class HypothesisOut(BaseModel):
    text: str
    round: int
    created_at: str


class RecallSessionOut(BaseModel):
    id: uuid.UUID
    original_query: str
    status: str
    current_question: str | None
    messages: list[MessageOut]
    cues: list[CueOut]
    hypotheses: list[HypothesisOut]
    round_count: int
    last_candidate_count: int
    final_answer: str | None
    sources: list[SourceOut]
    chunk_sources: list[ChunkSourceOut]
    created_at: datetime
    updated_at: datetime




# ---------------------------------------------------------------------------
# Associative memory graph (see docs/ARCHITECTURE.md "Associative memory
# graph"). Hand-built from app/memory/graph.py's dataclasses rather than
# from_attributes'd off the ORM: what the explorer needs is a node/edge view
# assembled across four tables, not a row-per-table dump.


class GraphNodeOut(BaseModel):
    id: str                      # "unit:<uuid>" | "entity:<uuid>"
    node_type: str               # "unit" | "entity"
    label: str
    kind: str                    # "memory" for units; the entity kind otherwise
    detail: str = ""
    degree: int = 0
    occurred_on: str | None = None
    # day | month | season | year | unknown -- the UI must not render a date at
    # more precision than this claims (see app/memory/temporal.py).
    date_precision: str = "unknown"


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    edge_type: str               # "mentions" | "association" | "relation"
    kind: str
    # explicit | extracted | inferred -- the authority distinction the UI
    # renders differently, so a computed association never looks like
    # something the user stated.
    origin: str
    weight: float
    label: str = ""
    explanation: str = ""


class GraphOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
    # True when a node cap dropped part of the graph -- surfaced so the UI can
    # say so rather than presenting a partial graph as the whole thing.
    truncated: bool = False
    total_units: int = 0
    total_entities: int = 0


class EntityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: str
    display_name: str
    normalized_name: str
    aliases: list
    mention_count: int


class EntityDetailOut(BaseModel):
    entity: EntityOut
    # Memories mentioning this entity, with the role/origin of each mention so
    # the caller can tell a user-stated attribution from an extracted one.
    mentions: list["EntityMentionOut"]
    relations: list["EntityRelationOut"]


class EntityMentionOut(BaseModel):
    unit_id: uuid.UUID
    capture_id: uuid.UUID
    content: str
    role: str
    origin: str
    weight: float
    provenance: dict
    created_at: datetime


class EntityRelationOut(BaseModel):
    id: uuid.UUID
    unit_id: uuid.UUID
    subject: str
    predicate: str
    object: str
    origin: str
    weight: float
    occurred_on: str | None
    date_precision: str


class AssociationStepOut(BaseModel):
    from_unit_id: uuid.UUID
    to_unit_id: uuid.UUID
    kind: str
    origin: str
    weight: float
    explanation: str


class AssociatedUnitOut(BaseModel):
    unit: MemoryUnitOut
    strength: float
    hops: int
    path: list[AssociationStepOut]
    # Deterministic, non-LLM justification for the whole path -- "why is this
    # memory here?" answered without a model call.
    why: str


class GraphStatsOut(BaseModel):
    ready_units: int
    entities: int
    entity_links: int
    relations: int
    associations: int
    # Entity counts per kind, for the explorer's filter chips.
    entities_by_kind: dict[str, int]
    # Association counts per kind, same purpose.
    associations_by_kind: dict[str, int]
    # The most-connected entities, for an at-a-glance "what your memory is
    # mostly about" panel on the overview page.
    top_entities: list[EntityOut]
