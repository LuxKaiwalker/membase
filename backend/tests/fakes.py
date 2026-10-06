import hashlib
from typing import Any

from app.llm.base import (
    LLMMalformedOutputError,
    LLMProvider,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.models import EMBEDDING_DIM

_EMPTY_METADATA = {
    "people": [],
    "location": "",
    "topics": [],
    "event_date": "",
    "summary": "",
    "clarification_question": "",
    "clarification_field": "",
}


def _hash_vector(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding with no real semantics -- good enough
    to exercise the vector-search code path and pgvector's fixed dimension,
    but NOT a stand-in for testing retrieval *quality* (see
    tests/test_integration_real_llm.py for that, which uses the real model).
    """
    digest = hashlib.sha256(text.encode()).digest()
    vals: list[float] = []
    while len(vals) < dim:
        digest = hashlib.sha256(digest).digest()
        vals.extend((b / 255.0) * 2 - 1 for b in digest)
    return vals[:dim]


class FakeLLMProvider(LLMProvider):
    """Fast, deterministic stand-in for LLMProvider used by unit tests.

    `fail` simulates backend failure modes: None | "unavailable" | "timeout"
    | "malformed". `structured`/`answer_response` let a test pin exactly
    what generate_structured() should return for metadata-extraction-shaped
    vs. answer-generation-shaped schemas, distinguished by looking for the
    "grounded" property (a bit of a shortcut, but keeps this fake small).
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        fail: str | None = None,
        structured: dict[str, Any] | None = None,
        answer_response: dict[str, Any] | None = None,
    ) -> None:
        self._dim = dim
        self.fail = fail
        self.structured = structured
        self.answer_response = answer_response

    @property
    def embedding_model_name(self) -> str:
        return "fake-embed"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _maybe_fail_connectivity(self) -> None:
        if self.fail == "unavailable":
            raise LLMUnavailableError("fake: backend unreachable")
        if self.fail == "timeout":
            raise LLMTimeoutError("fake: backend timed out")

    async def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        self._maybe_fail_connectivity()
        return "ok"

    async def generate_structured(
        self, prompt: str, *, schema: dict[str, Any], system: str | None = None
    ) -> dict[str, Any]:
        self._maybe_fail_connectivity()
        if self.fail == "malformed":
            raise LLMMalformedOutputError("fake: model did not return valid JSON")
        if "grounded" in schema.get("properties", {}):
            return self.answer_response or {"answer": "stub answer", "grounded": True}
        return self.structured or dict(_EMPTY_METADATA)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._maybe_fail_connectivity()
        return [_hash_vector(t, self._dim) for t in texts]
