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


class ChunkSourceOut(BaseModel):
    id: uuid.UUID
    capture_id: uuid.UUID
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


class LoginRequest(BaseModel):
    password: str
    # e.g. "iPhone", "work laptop" -- purely a label shown back in the
    # session list later, see AuthToken.device_label.
    device_label: str = ""


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    device_label: str
    created_at: datetime
    last_used_at: datetime | None


class LoginResponse(BaseModel):
    # Shown exactly once -- the client must store this itself (see
    # docs/DECISIONS.md "Password login with per-device tokens").
    token: str
    session: SessionOut
