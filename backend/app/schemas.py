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


