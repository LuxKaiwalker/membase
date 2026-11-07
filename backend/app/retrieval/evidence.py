"""Structured evidence for a retrieval candidate, and the calibrated
confidence categories built on top of it.

Two requirements drive this module, and they are the same requirement seen
from two sides:

- Retrieval has to *re-rank* on more than one signal (relevance, contextual
  consistency, evidence quality), which means those signals have to exist as
  separable things rather than as one fused float.
- The user has to be able to inspect *why* a result came back, including what
  argued against it.

So evidence is collected as a list of typed items, each carrying its own
provenance level, and both the ranking and the explanation are computed from
that one list. There is no second, display-only "explanation" that could drift
from what actually drove the ranking.

**Provenance levels** are the four the proposal requires, and they are a
property of each evidence item rather than of the candidate as a whole --
because a single candidate routinely mixes them (the user's own words, an
extracted entity, a computed association, and a search hypothesis can all
point at the same memory):

    "original"   -- the user's own text: authoritative_content, raw_text, a
                    clarification answer, user-supplied explicit metadata.
    "extracted"  -- an LLM extraction step read it out of the user's words:
                    semantic fields, entity mentions, relations.
    "inferred"   -- nobody stated it; the system computed it: a co-occurrence
                    association, a structurer link, a temporal adjacency.
    "hypothesis" -- a reformulated query or a guess the system made *while
                    searching*. Never evidence for an answer, only an
                    explanation of how the search went. Kept here so it can be
                    displayed and audited, explicitly excluded from scoring.

**Confidence** is a category, never a bare number shown to the user. The rule
table in `confidence_for` is the definition -- there is no separate prose
description of it that could disagree with the code.
"""

from dataclasses import dataclass, field

from app.models import MemoryUnit
from app.retrieval.pipeline import ScoredChunk, ScoredUnit

PROVENANCE_ORIGINAL = "original"
PROVENANCE_EXTRACTED = "extracted"
PROVENANCE_INFERRED = "inferred"
PROVENANCE_HYPOTHESIS = "hypothesis"

POLARITY_SUPPORTING = "supporting"
POLARITY_CONTRADICTING = "contradicting"
POLARITY_NEUTRAL = "neutral"

# Evidence kinds. Each is produced in exactly one place, so a kind appearing in
# a trail is traceable to the code that put it there.
KIND_DIRECT_MATCH = "direct_match"          # cleared the retrieval bar on its own
KIND_LEXICAL_MATCH = "lexical_match"        # full-text overlap specifically
KIND_ASSOCIATION = "association"            # reached by walking the memory graph
KIND_SHARED_ENTITY = "shared_entity"        # names something the query/cues named
KIND_CUE_MATCH = "cue_match"                # matches a cue the user confirmed
KIND_CUE_CONFLICT = "cue_conflict"          # matches something the user ruled out
KIND_CONTRADICTION = "contradiction"        # another stored memory disagrees
KIND_SIBLING = "sibling"                    # same original note as a stronger hit
KIND_ATTACHMENT = "attachment"              # an attached file's content matched
KIND_DEGRADED = "degraded_semantics"        # enrichment never completed for it
KIND_TEMPORAL = "temporal"                  # its date fits the query's timeframe
KIND_HYPOTHESIS = "hypothesis"              # a reformulation the system tried


@dataclass
class EvidenceItem:
    kind: str
    polarity: str
    provenance: str
    # One sentence a user can read, generated deterministically.
    detail: str
    # Contribution magnitude in [0, 1]. For supporting items this is added
    # (scaled, see associative.py's rerank); for contradicting items it is
    # subtracted. Hypothesis items always carry 0.0 -- see the module
    # docstring.
    weight: float = 0.0
    source_unit_id: str | None = None
    entity_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "polarity": self.polarity,
            "provenance": self.provenance,
            "detail": self.detail,
            "weight": round(self.weight, 4),
            "source_unit_id": self.source_unit_id,
            "entity_id": self.entity_id,
        }


# Route by which a candidate entered consideration. The distinction matters for
# both ranking and honesty: a memory that only ever surfaced because it is two
# hops from something that matched is a weaker claim than one that matched the
# query itself, and the UI says so.
ROUTE_DIRECT = "direct"
ROUTE_ASSOCIATION = "association"
ROUTE_BOTH = "both"


@dataclass
class Candidate:
    """One memory under consideration, with everything known about why."""

    unit: MemoryUnit
    route: str
    # Present when the unit cleared the retrieval bar on its own.
    direct: ScoredUnit | None = None
    # Present when the unit was reached through the association graph. Typed
    # as Any to avoid a circular import with app/memory/graph.py; it is an
    # app.memory.graph.AssociatedUnit.
    association: object | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    # Score before evidence adjustments -- retrieval relevance only.
    base_score: float = 0.0
    # Score after re-ranking on contextual consistency and evidence quality.
    score: float = 0.0

    @property
    def unit_id(self) -> str:
        return str(self.unit.id)

    @property
    def capture_id(self) -> str:
        return str(self.unit.capture_id)

    def add(self, item: EvidenceItem) -> None:
        self.evidence.append(item)

    def supporting(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.polarity == POLARITY_SUPPORTING]

    def contradicting(self) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.polarity == POLARITY_CONTRADICTING]

    def has_kind(self, kind: str) -> bool:
        return any(e.kind == kind for e in self.evidence)

    def as_dict(self) -> dict:
        """Serializable form for the API and for the session's stored trail."""
        association = self.association
        return {
            "unit_id": self.unit_id,
            "capture_id": self.capture_id,
            "content": self.unit.authoritative_content,
            "meaning": (self.unit.semantic or {}).get("meaning", ""),
            "created_at": self.unit.created_at.isoformat() if self.unit.created_at else None,
            "event_date": (self.unit.semantic or {}).get("event_date", ""),
            "route": self.route,
            "base_score": round(self.base_score, 4),
            "score": round(self.score, 4),
            "vector_score": self.direct.vector_score if self.direct else None,
            "fts_score": self.direct.fts_score if self.direct else None,
            "fused_score": self.direct.fused_score if self.direct else None,
            "association_strength": getattr(association, "strength", None),
            "association_hops": getattr(association, "hops", None),
            "association_why": getattr(association, "why", "") or "",
            "confidence": self.confidence,
            "evidence": [e.as_dict() for e in self.evidence],
        }

    @property
    def confidence(self) -> str:
        return confidence_for(self)


# --------------------------------------------------------------------------
# Confidence categories
# --------------------------------------------------------------------------
#
# Deliberately four coarse buckets rather than a percentage. A percentage
# implies a calibrated probability, and nothing in this system produces one:
# the fused retrieval score is a rank signal whose absolute value depends on
# the corpus, and the association weights are a documented heuristic. Showing
# "73% confident" would be a fabricated number -- exactly the kind of false
# precision the memory-integrity rules exist to prevent.
#
# What these DO mean is defined entirely by the thresholds below.

CONFIDENCE_STRONG = "strong"
CONFIDENCE_MODERATE = "moderate"
CONFIDENCE_WEAK = "weak"
CONFIDENCE_INSUFFICIENT = "insufficient"

CONFIDENCE_DESCRIPTIONS = {
    CONFIDENCE_STRONG: (
        "This memory matched your query directly and strongly, with no stored "
        "memory contradicting it."
    ),
    CONFIDENCE_MODERATE: (
        "This memory matched your query, but either not decisively or with "
        "some supporting evidence coming from inferred associations rather "
        "than from the memory's own words."
    ),
    CONFIDENCE_WEAK: (
        "This memory was reached mainly through associations with other "
        "memories, or it matched only weakly. Treat it as a lead, not an "
        "answer."
    ),
    CONFIDENCE_INSUFFICIENT: (
        "There is not enough evidence to put this memory forward as the "
        "answer."
    ),
}

# A direct fused score at or above this is "strongly matched". Same value as
# the agent's existing high-confidence shortcut (app/retrieval/agent.py's
# HIGH_CONFIDENCE_FUSED_SCORE), deliberately: two places in the system should
# not disagree about what "a strong match" means. Like that one, this is an
# engineering default sanity-checked against the eval corpus, not swept --
# see docs/TODO.md.
STRONG_DIRECT_SCORE = 0.80
MODERATE_DIRECT_SCORE = 0.55
WEAK_SCORE_FLOOR = 0.20


def confidence_for(candidate: Candidate) -> str:
    """Map a candidate's evidence onto a confidence category.

    The rules, in order of application:

    1. A stored memory that contradicts this one caps confidence at
       `moderate`, whatever the scores say. A high retrieval score is not
       evidence that the *other* memory is wrong.
    2. Semantics that never finished enriching (the LLM was down during
       ingestion) cap confidence at `moderate`: the memory is real, but the
       representation the match was made against is incomplete.
    3. A candidate reached only by association never exceeds `weak`. It was
       not matched; it was suggested by something that was.
    4. Otherwise the direct score decides, against the thresholds above.
    """
    contradicted = candidate.has_kind(KIND_CONTRADICTION)
    degraded = candidate.has_kind(KIND_DEGRADED)
    direct_score = candidate.direct.fused_score if candidate.direct else 0.0

    if candidate.route == ROUTE_ASSOCIATION:
        strength = getattr(candidate.association, "strength", 0.0) or 0.0
        return CONFIDENCE_WEAK if strength >= WEAK_SCORE_FLOOR else CONFIDENCE_INSUFFICIENT

    if direct_score >= STRONG_DIRECT_SCORE:
        return CONFIDENCE_MODERATE if (contradicted or degraded) else CONFIDENCE_STRONG
    if direct_score >= MODERATE_DIRECT_SCORE:
        return CONFIDENCE_WEAK if (contradicted or degraded) else CONFIDENCE_MODERATE
    if direct_score >= WEAK_SCORE_FLOOR:
        return CONFIDENCE_WEAK
    return CONFIDENCE_INSUFFICIENT


def overall_confidence(candidates: list[Candidate]) -> str:
    """Confidence in the *answer*, which is not simply the best candidate's.

    Two candidates of comparable strength from different original notes means
    the system does not actually know which one the user meant, so the answer
    is at most `moderate` even if each candidate alone looks strong.
    """
    if not candidates:
        return CONFIDENCE_INSUFFICIENT
    ordered = sorted(candidates, key=lambda c: -c.score)
    best = confidence_for(ordered[0])
    if len(ordered) > 1 and ordered[0].capture_id != ordered[1].capture_id:
        margin = ordered[0].score - ordered[1].score
        if margin < 0.15 and best == CONFIDENCE_STRONG:
            return CONFIDENCE_MODERATE
    return best


# --------------------------------------------------------------------------
# Evidence formatting for answer generation
# --------------------------------------------------------------------------


def provenance_label(level: str) -> str:
    return {
        PROVENANCE_ORIGINAL: "your own words",
        PROVENANCE_EXTRACTED: "extracted from your words",
        PROVENANCE_INFERRED: "inferred by the system",
        PROVENANCE_HYPOTHESIS: "a search guess, not a fact",
    }.get(level, level)


def format_candidate_evidence(candidate: Candidate, label: str) -> str:
    """One evidence block for the answer prompt.

    Associated (non-directly-matched) candidates are labelled as such in the
    prompt text itself, so the model is told plainly that the memory was
    suggested by an association rather than matched -- the prompt-level
    counterpart to the `route` distinction the UI renders.
    """
    unit = candidate.unit
    semantic = unit.semantic or {}
    lines = [f"[{label}] (created {unit.created_at.date().isoformat()})"]
    lines.append(f"Content: {unit.authoritative_content}")
    if semantic.get("meaning"):
        lines.append(f"Meaning: {semantic['meaning']}")
    if semantic.get("aliases"):
        lines.append(f"Also known as: {', '.join(semantic['aliases'])}")
    if semantic.get("definitions"):
        lines.append(
            "Definitions: "
            + "; ".join(f"{d['term']} = {d['definition']}" for d in semantic["definitions"])
        )
    if semantic.get("event_date"):
        lines.append(f"Event date (as recorded): {semantic['event_date']}")
    if candidate.route == ROUTE_ASSOCIATION:
        why = getattr(candidate.association, "why", "") or "a connection to another memory"
        lines.append(
            "NOTE: this memory did not match the question directly. It was "
            f"reached through an association: {why} Use it only as supporting "
            "context, and do not present it as the answer on its own."
        )
    for item in candidate.contradicting():
        lines.append(f"[Conflict] {item.detail}")
    return "\n".join(lines)


def format_evidence_block(
    candidates: list[Candidate],
    context: list[MemoryUnit],
    chunks: list[ScoredChunk] | None = None,
) -> str:
    """The full evidence text handed to answer generation.

    Mirrors app/retrieval/pipeline.py's `format_evidence` -- same ordering,
    same labels for the parts that exist in both -- so swapping one for the
    other cannot change the model's reading of a primary vs. a context block.
    """
    from app.retrieval.pipeline import _format_chunk, _format_unit  # local: avoids a cycle

    direct = [c for c in candidates if c.route != ROUTE_ASSOCIATION]
    associated = [c for c in candidates if c.route == ROUTE_ASSOCIATION]

    blocks = [format_candidate_evidence(c, f"Primary evidence {i}") for i, c in enumerate(direct, start=1)]
    blocks += [
        format_candidate_evidence(c, f"Associated memory {i}") for i, c in enumerate(associated, start=1)
    ]
    blocks += [_format_unit(u, "Related context (same note)") for u in context]
    blocks += [_format_chunk(c) for c in (chunks or [])]
    return "\n\n".join(blocks)
