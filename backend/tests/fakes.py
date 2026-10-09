import hashlib
import re
from typing import Any

from app.llm.base import (
    LLMMalformedOutputError,
    LLMProvider,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.models import EMBEDDING_DIM

# Shared default for every semantic field analyze_capture/reanalyze_unit
# can produce -- _parse_semantics()/_as_*() in app/ingestion/analysis.py
# already tolerate missing keys gracefully, but spelling out the full
# shape here keeps fake responses self-documenting.
_EMPTY_SEMANTIC_FIELDS = {
    "meaning": "", "aliases": [], "people": [], "organizations": [], "locations": [],
    "topics": [], "definitions": [], "relations": [], "event_date": "",
    "clarification_question": "", "clarification_field": "",
}
_DEFAULT_EVALUATE = {"sufficient": True, "reformulated_queries": []}
_DEFAULT_ANSWER = {"answer": "stub answer", "grounded": True}
_DEFAULT_CUES = {"cues": []}

# analyze_capture's prompt always ends "Text:\n{raw_text}\n\nAnalyze
# this." (see app/ingestion/analysis.py) -- extracting it lets the fake
# echo the real text back as the default single unit's verbatim
# "content" without a test needing to repeat it in segmentation_response.
_CAPTURE_TEXT_RE = re.compile(r"Text:\n(.*)\n\nAnalyze this\.", re.DOTALL)


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
    None falls back to `default` (which may itself be None, as a sentinel
    meaning "nothing configured, caller builds a dynamic default")."""

    def __init__(self, configured: Any, default: Any) -> None:
        self._default = default
        if configured is None:
            self._items: list[Any] | None = None
        elif isinstance(configured, list):
            self._items = list(configured)
        else:
            self._items = [configured]

    def next(self) -> Any:
        if self._items is None:
            return self._default
        if len(self._items) > 1:
            return self._items.pop(0)
        return self._items[0]


class FakeLLMProvider(LLMProvider):
    """Fast, deterministic stand-in for LLMProvider used by unit tests.

    `fail` simulates backend failure modes: None | "unavailable" | "timeout"
    | "malformed" (applies to every call). Dispatched by a distinguishing
    key in the schema's properties:
      - "units"       -> analyze_capture() (segmentation + per-unit
                         analysis in one call, see
                         app/ingestion/analysis.py). If
                         `segmentation_response` is configured, returned
                         as-is (for multi-unit tests); otherwise a single
                         unit is built automatically, echoing the real
                         text being analyzed as its verbatim "content"
                         and drawing semantic/clarification fields from
                         `enrichment_response`.
      - "sufficient"  -> agent's evaluate-sufficiency step
      - "grounded"    -> final answer generation
      - "cues"        -> recall-session cue extraction
      - (none of the above) -> reanalyze_unit() (a single unit's
                         post-clarification re-analysis -- same semantic
                         shape as above minus "content"/"units"), drawn
                         from the same `enrichment_response` queue.

    Each `*_response` may be a single dict (returned every time) or a
    list (consumed in order, one per actual call -- unlike the old
    three-sub-call enrich(), analyze_capture()/reanalyze_unit() are each
    exactly one call now, so a list's Nth item is simply the Nth call's
    response, no multiplier to account for).
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
        image_caption_response: Any = None,
    ) -> None:
        self._dim = dim
        self.fail = fail
        self._segmentation = _ResponseQueue(segmentation_response, None)
        self._enrichment = _ResponseQueue(enrichment_response, dict(_EMPTY_SEMANTIC_FIELDS))
        self._evaluate = _ResponseQueue(evaluate_response, _DEFAULT_EVALUATE)
        self._answer = _ResponseQueue(answer_response, _DEFAULT_ANSWER)
        self._cues = _ResponseQueue(cue_response, dict(_DEFAULT_CUES))
        if isinstance(image_caption_response, list):
            wrapped_caption_response = [{"caption": c} for c in image_caption_response]
        elif isinstance(image_caption_response, str):
            wrapped_caption_response = {"caption": image_caption_response}
        else:
            wrapped_caption_response = image_caption_response
        self._image_caption = _ResponseQueue(wrapped_caption_response, {"caption": "a fake image"})
        self.calls: list[str] = []  # dispatch "kind" per generate_structured call, for tests that assert call counts
        self.images_received: list[list[bytes]] = []
        self.embed_calls: list[list[str]] = []  # texts passed per embed() call, for tests asserting dedup

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

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        images: list[bytes] | None = None,
    ) -> str:
        self._maybe_fail_connectivity()
        if images:
            self.images_received.append(images)
            self.calls.append("image_caption")
            return self._image_caption.next()["caption"]
        return "ok"

    async def generate_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        self._maybe_fail_connectivity()
        if self.fail == "malformed":
            raise LLMMalformedOutputError("fake: model did not return valid JSON")

        props = schema.get("properties", {})
        if "units" in props:
            self.calls.append("analyze_capture")
            configured = self._segmentation.next()
            if configured is not None:
                return configured
            match = _CAPTURE_TEXT_RE.search(prompt)
            content = match.group(1) if match else prompt
            return {"units": [{"content": content, **self._enrichment.next()}]}
        if "sufficient" in props:
            self.calls.append("evaluate")
            return self._evaluate.next()
        if "grounded" in props:
            self.calls.append("answer")
            return self._answer.next()
        if "cues" in props:
            self.calls.append("cues")
            return self._cues.next()
        self.calls.append("reanalyze")
        return dict(self._enrichment.next())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        self._maybe_fail_connectivity()
        return [_hash_vector(t, self._dim) for t in texts]
