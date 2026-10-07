import hashlib
from typing import Any

from app.llm.base import (
    LLMMalformedOutputError,
    LLMProvider,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.models import EMBEDDING_DIM

_EMPTY_SEMANTICS = {
    "meaning": "",
    "aliases": [],
    "people": [],
    "organizations": [],
    "locations": [],
    "topics": [],
    "definitions": [],
    "relations": [],
    "event_date": "",
    "clarification_question": "",
    "clarification_field": "",
}

_DEFAULT_SEGMENTATION = {"units": []}  # empty -> pipeline falls back to no-split
_DEFAULT_EVALUATE = {"sufficient": True, "reformulated_queries": []}
_DEFAULT_ANSWER = {"answer": "stub answer", "grounded": True}
_DEFAULT_CUES = {"cues": []}


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


class _ResponseQueue:
    """Wraps a configured response: a single dict is returned every call; a
    list is popped in order, with the last item repeating once exhausted.
    None falls back to `default`."""

    def __init__(self, configured: Any, default: dict) -> None:
        self._default = default
        if configured is None:
            self._items: list[dict] | None = None
        elif isinstance(configured, list):
            self._items = list(configured)
        else:
            self._items = [configured]

    def next(self) -> dict:
        if self._items is None:
            return self._default
        if len(self._items) > 1:
            return self._items.pop(0)
        return self._items[0]


class FakeLLMProvider(LLMProvider):
    """Fast, deterministic stand-in for LLMProvider used by unit tests.

    `fail` simulates backend failure modes: None | "unavailable" | "timeout"
    | "malformed" (applies to every call). The five `*_response` params let
    a test pin exactly what generate_structured() returns for each of the
    five structured-output shapes the app uses, dispatched by a
    distinguishing key in the schema's properties:
      - "units"               -> segmentation
      - "sufficient"          -> agent's evaluate-sufficiency step
      - "grounded"            -> final answer generation
      - "cues"                -> recall-session cue extraction
      - (none of the above)   -> semantic enrichment

    Each may be a single dict (returned every time) or a list (consumed in
    order, popped in order with the last value held once exhausted) --
    useful for simulating a multi-round agent or a clarification that
    changes the model's answer on the second call.
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        fail: str | None = None,
        segmentation_response: Any = None,
        enrichment_response: Any = None,
        evaluate_response: Any = None,
        answer_response: Any = None,
        cue_response: Any = None,
    ) -> None:
        self._dim = dim
        self.fail = fail
        self._segmentation = _ResponseQueue(segmentation_response, _DEFAULT_SEGMENTATION)
        self._enrichment = _ResponseQueue(enrichment_response, dict(_EMPTY_SEMANTICS))
        self._evaluate = _ResponseQueue(evaluate_response, _DEFAULT_EVALUATE)
        self._answer = _ResponseQueue(answer_response, _DEFAULT_ANSWER)
        self._cues = _ResponseQueue(cue_response, dict(_DEFAULT_CUES))
        self.calls: list[str] = []  # schema "kind" per generate_structured call, for tests that assert call counts

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

        props = schema.get("properties", {})
        if "units" in props:
            self.calls.append("segmentation")
            return self._segmentation.next()
        if "sufficient" in props:
            self.calls.append("evaluate")
            return self._evaluate.next()
        if "grounded" in props:
            self.calls.append("answer")
            return self._answer.next()
        if "cues" in props:
            self.calls.append("cues")
            return self._cues.next()
        self.calls.append("enrichment")
        return self._enrichment.next()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._maybe_fail_connectivity()
        return [_hash_vector(t, self._dim) for t in texts]
