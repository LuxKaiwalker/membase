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

# Shared default for every semantic field enrich_unit() can produce --
# _as_*()/empty_semantics() in app/ingestion/analysis.py already
# tolerate missing keys gracefully, but spelling out the full shape here
# keeps fake responses self-documenting. No clarification fields here
# any more -- clarify.generate_question() is a separate call now with
# its own response queue (see `clarification_response` below).
_EMPTY_SEMANTIC_FIELDS = {
    "meaning": "", "aliases": [], "people": [], "organizations": [], "locations": [],
    "topics": [], "definitions": [], "relations": [], "event_date": "",
}
_DEFAULT_CLARIFICATION = {"clarification_question": "", "clarification_field": ""}
_DEFAULT_IMAGE_STRUCTURED = {"has_structured_data": False, "fields": []}
_DEFAULT_EVALUATE = {"sufficient": True, "reformulated_queries": []}
_DEFAULT_ANSWER = {"answer": "stub answer", "grounded": True}
_DEFAULT_CUES = {"cues": []}

# segment_capture()'s prompt always ends "Text:\n{raw_text}\n\nSplit
# this into memory units." (see app/ingestion/segmentation.py) --
# extracting it lets the fake echo the real text back as the default
# single unit's verbatim "content" without a test needing to repeat it
# in segmentation_response.
_CAPTURE_TEXT_RE = re.compile(r"Text:\n(.*)\n\nSplit this into memory units\.", re.DOTALL)


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

    def peek(self) -> Any:
        """Returns the current item WITHOUT consuming it -- used for
        enrich_unit()'s 3 sub-calls, which must all see the same
        per-unit response (see `advance()` below)."""
        if self._items is None:
            return self._default
        return self._items[0] if self._items else self._default

    def advance(self) -> None:
        """Consumes the current item -- called once, after the LAST of
        enrich_unit()'s 3 sub-calls, so one queued item corresponds to
        one enrich_unit() invocation, not one sub-call."""
        if self._items and len(self._items) > 1:
            self._items.pop(0)


def _derive_clarification_response(enrichment_response: Any) -> Any:
    """Backward-compatible convenience: many test fixtures bundle
    clarification_question/clarification_field into the same single
    `enrichment_response` dict a one-call-does-everything fixture used
    to configure. If the caller didn't explicitly pass
    `clarification_response`, and `enrichment_response` is a plain dict
    (not a per-unit list -- where this derivation can't unambiguously
    line a clarify call up with the right unit) carrying either key,
    derive a constant clarification_response from it so those fixtures
    keep working unmodified under the restored multi-call dispatch."""
    if isinstance(enrichment_response, dict) and (
        "clarification_question" in enrichment_response or "clarification_field" in enrichment_response
    ):
        return {
            "clarification_question": enrichment_response.get("clarification_question", ""),
            "clarification_field": enrichment_response.get("clarification_field", ""),
        }
    return None


class FakeLLMProvider(LLMProvider):
    """Fast, deterministic stand-in for LLMProvider used by unit tests.

    `fail` simulates backend failure modes: None | "unavailable" | "timeout"
    | "malformed" (applies to every call). Dispatched by a distinguishing
    key in the schema's properties -- the restored multi-call pipeline
    (see app/ingestion/{segmentation,clarify,enrichment}.py) makes
    several separate `generate_structured` calls per unit instead of one
    combined call, each with its own narrow schema:
      - "units"                 -> segment_capture() (content-only, no
                                    semantic fields any more). If
                                    `segmentation_response` is configured,
                                    returned as-is (for multi-unit tests);
                                    otherwise a single unit is built
                                    automatically, echoing the real text
                                    being segmented as its verbatim
                                    "content".
      - "clarification_question" -> clarify.generate_question(), drawn
                                    from `clarification_response`
                                    (default: no question).
      - "meaning" (no "people")  -> enrichment's narrative sub-call
      - "people"                 -> enrichment's entities sub-call
      - "aliases"                -> enrichment's retrieval-aids sub-call
        These three share ONE `enrichment_response` queue item per unit
        (peeked by narrative/entities, consumed by retrieval-aids, the
        last of the three) -- a list's Nth item is the Nth UNIT's
        response, not the Nth sub-call's.
      - "has_structured_data"    -> the image structured-data-extraction
                                    call, drawn from
                                    `image_structured_response`.
      - "sufficient"             -> agent's evaluate-sufficiency step
      - "grounded"                -> final answer generation
      - "cues"                   -> recall-session cue extraction

    Each `*_response` may be a single dict (returned every time) or a
    list (consumed in order).
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        fail: str | None = None,
        segmentation_response: Any = None,
        enrichment_response: Any = None,
        clarification_response: Any = None,
        image_structured_response: Any = None,
        evaluate_response: Any = None,
        answer_response: Any = None,
        cue_response: Any = None,
        image_caption_response: Any = None,
    ) -> None:
        self._dim = dim
        self.fail = fail
        self._segmentation = _ResponseQueue(segmentation_response, None)
        self._enrichment = _ResponseQueue(enrichment_response, dict(_EMPTY_SEMANTIC_FIELDS))
        if clarification_response is None:
            clarification_response = _derive_clarification_response(enrichment_response)
        self._clarification = _ResponseQueue(clarification_response, dict(_DEFAULT_CLARIFICATION))
        self._image_structured = _ResponseQueue(image_structured_response, dict(_DEFAULT_IMAGE_STRUCTURED))
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
        if "related_indices" in props:
            # app/ingestion/structurer.py's deep second pass: one
            # combined call (still unsplit -- it's a quality-only pass
            # with no clarification decision riding on it), so this
            # just returns the configured dict as-is (same semantics as
            # the old unified-call "reanalyze" catch-all).
            self.calls.append("structure")
            return dict(self._enrichment.next())
        if "has_structured_data" in props:
            self.calls.append("image_structured")
            return self._image_structured.next()
        if "units" in props:
            self.calls.append("segment")
            configured = self._segmentation.next()
            if configured is not None:
                return configured
            match = _CAPTURE_TEXT_RE.search(prompt)
            content = match.group(1) if match else prompt
            return {"units": [{"content": content}]}
        if "clarification_question" in props:
            self.calls.append("clarify")
            return self._clarification.next()
        if "people" in props:
            self.calls.append("entities")
            item = self._enrichment.peek()
            return {
                "people": item.get("people", []),
                "organizations": item.get("organizations", []),
                "locations": item.get("locations", []),
            }
        if "aliases" in props:
            self.calls.append("retrieval_aids")
            item = self._enrichment.peek()
            self._enrichment.advance()
            return {
                "aliases": item.get("aliases", []),
                "topics": item.get("topics", []),
                "definitions": item.get("definitions", []),
                "relations": item.get("relations", []),
            }
        if "meaning" in props:
            self.calls.append("narrative")
            item = self._enrichment.peek()
            return {"meaning": item.get("meaning", ""), "event_date": item.get("event_date", "")}
        if "sufficient" in props:
            self.calls.append("evaluate")
            return self._evaluate.next()
        if "grounded" in props:
            self.calls.append("answer")
            return self._answer.next()
        if "cues" in props:
            self.calls.append("cues")
            return self._cues.next()
        raise AssertionError(f"FakeLLMProvider: no dispatch rule matched schema properties {list(props)}")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        self._maybe_fail_connectivity()
        return [_hash_vector(t, self._dim) for t in texts]
