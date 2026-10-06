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


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    caption: str
    created_at: datetime
    derived: dict
    pending_clarification: dict | None
    attachments: list[AttachmentOut]


class ClarifyRequest(BaseModel):
    answer: str


class RecallRequest(BaseModel):
    query: str


class SourceOut(BaseModel):
    id: uuid.UUID
    caption: str
    derived: dict
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
