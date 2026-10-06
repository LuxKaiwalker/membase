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
    created_at: datetime


class CaptureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    raw_text: str
    created_at: datetime
    attachments: list[AttachmentOut]
    units: list[MemoryUnitOut]


class ClarifyRequest(BaseModel):
    answer: str


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


class RecallResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[SourceOut]


class ErrorResponse(BaseModel):
    detail: str
