"""Retrieval level 6: associative and iterative recall.

The existing retrieval hierarchy (docs/ARCHITECTURE.md) already implements
levels 1-5 -- exact content, unit-level, clarification-enriched, semantic, and
fused vector+full-text search. What it never had was level 6's *associative*
half: a way to reach a memory that shares no wording with the query at all,
only a connection to something that does.

That is what this module adds, as a **stage** rather than a replacement.
`hybrid_search` and `automatic_search` keep their exact contracts; this runs
after them, takes their ranked output as seeds, walks the memory graph, and
returns a richer candidate set with structured evidence attached.

Three properties are deliberate:

**Adaptive, not always-on.** `should_expand` declines to walk the graph when
the direct search already produced a dominant, evidentially-complete winner --
the same condition the agent's existing sufficiency short-circuit uses. A
clear query must not pay for association exploration it cannot benefit from.
This is the "simple queries answered directly, ambiguous queries explored
deeply" requirement, implemented as a deterministic gate rather than as a
model judgement.

**Bounded on every axis.** Hops, expanded candidates, total candidates, and
wall-clock all have explicit ceilings in `RetrievalBudget`, and the budget is
threaded through rather than read from globals, so a benchmark can run the
same code at a different budget without monkeypatching.

**Associated candidates never silently become answers.** An expanded candidate
carries `route="association"`, is capped at `weak` confidence
(app/retrieval/evidence.py), is labelled as not-directly-matched in the answer
prompt, and cannot outrank a direct match purely through evidence bonuses (see
`rerank`'s clamp). Reaching a memory by inference is a lead, not a finding.
"""

import logging
import re
import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.memory import graph as memory_graph
from app.memory.entities import normalize_name
from app.memory.temporal import PRECISION_UNKNOWN, dates_compatible, describe, parse_event_date
from app.models import MemoryEntity, UnitEntityLink
from app.retrieval.evidence import (
    KIND_ASSOCIATION,
    KIND_ATTACHMENT,
    KIND_CONTRADICTION,
    KIND_CUE_MATCH,
    KIND_DEGRADED,
    KIND_DIRECT_MATCH,
    KIND_LEXICAL_MATCH,
    KIND_SHARED_ENTITY,
    KIND_TEMPORAL,
    POLARITY_CONTRADICTING,
    POLARITY_NEUTRAL,
    POLARITY_SUPPORTING,
    PROVENANCE_EXTRACTED,
    PROVENANCE_INFERRED,
    PROVENANCE_ORIGINAL,
    ROUTE_ASSOCIATION,
    ROUTE_BOTH,
    ROUTE_DIRECT,
    Candidate,
    EvidenceItem,
)
from app.retrieval.pipeline import ScoredChunk, ScoredUnit

logger = logging.getLogger(__name__)


@dataclass
class RetrievalBudget:
    """Explicit limits for one retrieval round.

    Passed in rather than read from module globals so the evaluation harness
    can compare budgets without patching the module it is measuring -- the
    problem docs/DECISIONS.md's threshold sweep had to work around by
    assigning to `retrieval_pipeline.VECTOR_SIM_THRESHOLD` and restoring it.
    """

    # Graph hops. 2 is the hard ceiling in app/memory/graph.py; past that the
    # decayed strength falls below any useful floor anyway, and the candidate
    # set stops being explainable to a user.
    max_association_hops: int = 2
    # How many association-route candidates may be added.
    max_expanded_candidates: int = 8
    # Ceiling on the whole candidate set handed to evidence collection and
    # answer generation. Bounds prompt size, which is the real cost driver.
    max_total_candidates: int = 12
    # Minimum decayed association strength worth considering at all.
    min_association_strength: float = 0.12
    # Only seeds at least this relevant are walked from -- expanding out of
    # noise produces noise two hops deeper.
    min_seed_score_for_expansion: float = 0.35
    # Wall-clock ceiling for the non-LLM work in one round. Graph traversal is
    # pure SQL and normally takes milliseconds; this exists so a pathological
    # corpus cannot make a round hang, not because it is expected to bind.
    deadline_seconds: float = 10.0


# Re-ranking adjustments. Deliberately small relative to the retrieval score
# itself: these refine an ordering, they do not override it. A candidate that
# matched the query badly should not win on accumulated bonuses.
#
# CUE_MATCH_BONUS is deliberately ZERO. A confirmed cue is already priced into
# the candidate's score upstream, by `apply_positive_cue_boost`
# (POSITIVE_CUE_BOOST = 0.3, which is large on purpose -- see
# app/retrieval/discriminate.py). Adding a second reward here, plus a third via
# SHARED_ENTITY_BONUS when the cue also names one of the candidate's entities,
# counted one signal three times -- observed live, where it pinned an unrelated
# memory to the score ceiling for sharing a single person with the query. The
# cue-match evidence item is still produced and still shown: it belongs in the
# explanation, it just must not be paid for twice.
CUE_MATCH_BONUS = 0.0
SHARED_ENTITY_BONUS = 0.06
TEMPORAL_MATCH_BONUS = 0.08
TEMPORAL_MISMATCH_PENALTY = 0.10
CONTRADICTION_PENALTY = 0.05
DEGRADED_PENALTY = 0.08
# Hard ceiling on an association-route candidate's final score, so no amount
# of accumulated supporting evidence lets an inferred lead outrank a genuine
# direct match. Set just below MODERATE_DIRECT_SCORE in evidence.py.
ASSOCIATION_SCORE_CEILING = 0.54

# Evidence-quality bonus is capped so that a candidate with many weak
# supporting items cannot overtake one with a decisively better match.
MAX_EVIDENCE_BONUS = 0.25


@dataclass
class RetrievalTrace:
    """What the engine actually did this round -- the record behind the
    "retrieval strategy used" the explainability requirement asks for."""

    strategies: list[str] = field(default_factory=list)
    direct_candidates: int = 0
    expanded_candidates: int = 0
    ruled_out: list[dict] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    expansion_skipped_reason: str | None = None
    elapsed_ms: int = 0
    budget: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "strategies": self.strategies,
            "direct_candidates": self.direct_candidates,
            "expanded_candidates": self.expanded_candidates,
            "ruled_out": self.ruled_out,
            "hypotheses": self.hypotheses,
            "expansion_skipped_reason": self.expansion_skipped_reason,
            "elapsed_ms": self.elapsed_ms,
            "budget": self.budget,
        }


# --------------------------------------------------------------------------
# Building candidates
# --------------------------------------------------------------------------


def build_direct_candidates(ranked: list[ScoredUnit]) -> list[Candidate]:
    """Wrap directly-retrieved units as candidates with their match evidence."""
    candidates = []
    for scored in ranked:
        candidate = Candidate(
            unit=scored.unit,
            route=ROUTE_DIRECT,
            direct=scored,
            base_score=scored.fused_score,
            score=scored.fused_score,
        )
        candidate.add(
            EvidenceItem(
                kind=KIND_DIRECT_MATCH,
                polarity=POLARITY_SUPPORTING,
                # The match is against retrieval_text, which is built from the
                # user's own words plus derived fields -- so the *match* is
                # extracted-level, even though the content it points at is
                # original-level.
                provenance=PROVENANCE_EXTRACTED,
                detail=(
                    "This memory matched your query directly "
                    f"(relevance {scored.fused_score:.2f})."
                ),
                weight=min(1.0, scored.fused_score),
                source_unit_id=str(scored.unit.id),
            )
        )
        if scored.fts_score:
            candidate.add(
                EvidenceItem(
                    kind=KIND_LEXICAL_MATCH,
                    polarity=POLARITY_SUPPORTING,
                    provenance=PROVENANCE_ORIGINAL,
                    detail="It shares literal wording with your query.",
                    weight=0.0,  # already reflected in the fused score
                    source_unit_id=str(scored.unit.id),
                )
            )
        candidates.append(candidate)
    return candidates


def should_expand(candidates: list[Candidate]) -> tuple[bool, str | None]:
    """Decide whether this query needs the graph walked at all.

    Returns `(expand, skip_reason)`. The skip reason is recorded in the trace
    so a user inspecting a fast answer can see that exploration was skipped
    deliberately, not forgotten.
    """
    if not candidates:
        # Nothing matched, so there is nothing to associate *from*. Expanding
        # from an empty seed set cannot find anything, and pretending to try
        # would just be a slower abstention.
        return False, "nothing matched directly, so there was nothing to explore from"

    ordered = sorted(candidates, key=lambda c: -c.base_score)
    top = ordered[0]
    runner_up = ordered[1].base_score if len(ordered) > 1 else 0.0
    semantic = top.unit.semantic or {}
    degraded = semantic.get("enrichment_status") == "unavailable"

    # Same three-signal test as the agent's sufficiency short-circuit
    # (app/retrieval/agent.py::_is_high_confidence): dominant absolute score,
    # clear margin, and complete semantics. Kept in sync with it on purpose --
    # "this query is already answered" should not mean two different things in
    # two places.
    from app.retrieval.agent import HIGH_CONFIDENCE_FUSED_SCORE, HIGH_CONFIDENCE_MARGIN

    if (
        top.base_score >= HIGH_CONFIDENCE_FUSED_SCORE
        and (top.base_score - runner_up) >= HIGH_CONFIDENCE_MARGIN
        and not degraded
    ):
        return False, "one memory matched clearly enough that exploring associations could not improve it"

    return True, None


def expand_with_associations(
    db: Session,
    candidates: list[Candidate],
    budget: RetrievalBudget,
    *,
    started_at: float | None = None,
) -> list[Candidate]:
    """Walk the memory graph outward from the strongest candidates.

    Returns NEW association-route candidates only; candidates already present
    directly are given an extra evidence item instead (a memory that both
    matched and is associated with another match is better supported, but it
    is still a direct match, not an inferred one).
    """
    existing = {str(c.unit.id): c for c in candidates}
    seeds = {
        c.unit.id: c.base_score
        for c in candidates
        if c.base_score >= budget.min_seed_score_for_expansion
    }
    if not seeds:
        # Every candidate is too weak to be a trustworthy starting point. Fall
        # back to the single best one rather than giving up entirely -- a weak
        # cue is exactly the case associative recall exists for, and refusing
        # to explore here would defeat the point.
        best = max(candidates, key=lambda c: c.base_score, default=None)
        if best is None:
            return []
        seeds = {best.unit.id: best.base_score}

    associated = memory_graph.neighbors(
        db,
        seeds,
        max_hops=budget.max_association_hops,
        limit=budget.max_expanded_candidates * 2,
        min_strength=budget.min_association_strength,
    )

    new_candidates: list[Candidate] = []
    for assoc in associated:
        if started_at is not None and (time.monotonic() - started_at) > budget.deadline_seconds:
            logger.warning("Association expansion hit its wall-clock budget; returning what was found")
            break
        unit_id = str(assoc.unit.id)
        if unit_id in existing:
            existing[unit_id].route = ROUTE_BOTH
            existing[unit_id].association = assoc
            existing[unit_id].add(
                EvidenceItem(
                    kind=KIND_ASSOCIATION,
                    polarity=POLARITY_SUPPORTING,
                    provenance=PROVENANCE_INFERRED,
                    detail=f"It is also connected to another matching memory. {assoc.why}".strip(),
                    weight=min(1.0, assoc.strength),
                    source_unit_id=unit_id,
                )
            )
            continue
        if len(new_candidates) >= budget.max_expanded_candidates:
            break
        candidate = Candidate(
            unit=assoc.unit,
            route=ROUTE_ASSOCIATION,
            association=assoc,
            base_score=assoc.strength,
            score=assoc.strength,
        )
        candidate.add(
            EvidenceItem(
                kind=KIND_ASSOCIATION,
                polarity=POLARITY_SUPPORTING,
                provenance=PROVENANCE_INFERRED,
                detail=(
                    f"Reached by association ({assoc.hops} "
                    f"{'hop' if assoc.hops == 1 else 'hops'}) from a memory that did match. "
                    f"{assoc.why}"
                ).strip(),
                weight=min(1.0, assoc.strength),
                source_unit_id=unit_id,
            )
        )
        new_candidates.append(candidate)
    return new_candidates


# --------------------------------------------------------------------------
# Evidence collection
# --------------------------------------------------------------------------


def _entities_for_units(db: Session, unit_ids: list) -> dict[str, list[MemoryEntity]]:
    """Distinct entities per unit.

    Deduplicated by entity id, which is load-bearing rather than tidy: a unit
    legitimately links the same entity under several roles (named in `people`,
    and again as a relation's subject and object), so the raw link rows yield
    "Markus" two or three times for one memory. Without this, `collect_evidence`
    emitted the identical "It is about Markus (person)" item three times AND
    `rerank` added the shared-entity bonus three times for one shared entity --
    observed for real, where it pushed an unrelated memory that merely
    mentioned the same person to the top of the ranking with a `strong`
    confidence it had not earned.
    """
    if not unit_ids:
        return {}
    rows = db.execute(
        select(UnitEntityLink.unit_id, MemoryEntity)
        .join(MemoryEntity, MemoryEntity.id == UnitEntityLink.entity_id)
        .where(UnitEntityLink.unit_id.in_(unit_ids))
    ).all()
    out: dict[str, dict] = {}
    for unit_id, entity in rows:
        out.setdefault(str(unit_id), {})[entity.id] = entity
    return {unit_id: list(entities.values()) for unit_id, entities in out.items()}


def _mentions_term(haystack: str, needle: str) -> bool:
    """Word-boundary containment, the same rule
    app/retrieval/discriminate.py::_matches_cue uses -- a plain substring test
    lets "Ann" match inside "annual"."""
    if len(needle) < 3:
        return False
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def collect_evidence(
    db: Session,
    candidates: list[Candidate],
    *,
    query: str,
    positive_cues: list[str] | None = None,
    chunks: list[ScoredChunk] | None = None,
) -> None:
    """Attach contextual-consistency and evidence-quality items in place.

    Everything here is deterministic: entity overlap, date compatibility,
    stored contradiction markers, and degraded-enrichment detection. No LLM
    call -- which is what makes the evidence trail reproducible for the same
    corpus and query.
    """
    if not candidates:
        return

    positive_cues = [c for c in (positive_cues or []) if c and c.strip()]
    normalized_query = normalize_name(query)
    cue_texts = {normalize_name(c): c for c in positive_cues if normalize_name(c)}
    entities_by_unit = _entities_for_units(db, [c.unit.id for c in candidates])

    query_date, query_precision = parse_event_date(query)
    if query_precision == PRECISION_UNKNOWN:
        for cue in positive_cues:
            query_date, query_precision = parse_event_date(cue)
            if query_precision != PRECISION_UNKNOWN:
                break

    chunk_capture_ids = {str(c.chunk.capture_id) for c in (chunks or [])}

    for candidate in candidates:
        semantic = candidate.unit.semantic or {}
        normalized_text = normalize_name(candidate.unit.retrieval_text or candidate.unit.authoritative_content)

        for normalized_cue, original_cue in cue_texts.items():
            if _mentions_term(normalized_text, normalized_cue):
                candidate.add(
                    EvidenceItem(
                        kind=KIND_CUE_MATCH,
                        polarity=POLARITY_SUPPORTING,
                        # The cue is the user's own words; that this memory
                        # contains it is a fact about stored text, not an
                        # inference.
                        provenance=PROVENANCE_ORIGINAL,
                        detail=f"It mentions “{original_cue}”, which you confirmed.",
                        weight=1.0,
                        source_unit_id=candidate.unit_id,
                    )
                )

        for entity in entities_by_unit.get(candidate.unit_id, []):
            name = entity.normalized_name
            if not name:
                continue
            in_query = _mentions_term(normalized_query, name)
            in_cue = any(_mentions_term(normalize_name(c), name) for c in positive_cues)
            if in_query or in_cue:
                candidate.add(
                    EvidenceItem(
                        kind=KIND_SHARED_ENTITY,
                        polarity=POLARITY_SUPPORTING,
                        provenance=PROVENANCE_EXTRACTED,
                        detail=(
                            f"It is about {entity.display_name} ({entity.kind}), which "
                            f"{'your query' if in_query else 'something you told me'} named."
                        ),
                        # Only an entity named by the QUERY adds to the score.
                        # One named by a cue is the same signal
                        # `apply_positive_cue_boost` already rewarded upstream,
                        # so scoring it here too would double-count it. The
                        # item is still recorded either way, because "it is
                        # about Markus, who you mentioned" is worth telling the
                        # user regardless of whether it moves the ranking.
                        weight=0.8 if in_query else 0.0,
                        source_unit_id=candidate.unit_id,
                        entity_id=str(entity.id),
                    )
                )

        candidate_date, candidate_precision = parse_event_date(semantic.get("event_date"))
        if query_date is not None and candidate_date is not None:
            compatible = dates_compatible(query_date, query_precision, candidate_date, candidate_precision)
            candidate.add(
                EvidenceItem(
                    kind=KIND_TEMPORAL,
                    polarity=POLARITY_SUPPORTING if compatible else POLARITY_CONTRADICTING,
                    provenance=PROVENANCE_EXTRACTED,
                    detail=(
                        f"Its recorded date ({describe(candidate_date, candidate_precision)}) "
                        + (
                            f"fits the timeframe you described ({describe(query_date, query_precision)})."
                            if compatible
                            else f"does not fit the timeframe you described ({describe(query_date, query_precision)})."
                        )
                    ),
                    weight=1.0,
                    source_unit_id=candidate.unit_id,
                )
            )

        for conflict in candidate.unit.potential_conflicts or []:
            note = conflict.get("note") if isinstance(conflict, dict) else None
            if note:
                candidate.add(
                    EvidenceItem(
                        kind=KIND_CONTRADICTION,
                        polarity=POLARITY_CONTRADICTING,
                        provenance=PROVENANCE_EXTRACTED,
                        detail=note,
                        weight=1.0,
                        source_unit_id=(
                            str(conflict.get("unit_id")) if conflict.get("unit_id") else None
                        ),
                    )
                )

        if semantic.get("enrichment_status") == "unavailable":
            candidate.add(
                EvidenceItem(
                    kind=KIND_DEGRADED,
                    polarity=POLARITY_CONTRADICTING,
                    provenance=PROVENANCE_INFERRED,
                    detail=(
                        "This memory's semantic analysis never completed, so it was matched "
                        "against incomplete information."
                    ),
                    weight=1.0,
                    source_unit_id=candidate.unit_id,
                )
            )

        if candidate.capture_id in chunk_capture_ids:
            candidate.add(
                EvidenceItem(
                    kind=KIND_ATTACHMENT,
                    polarity=POLARITY_SUPPORTING,
                    provenance=PROVENANCE_ORIGINAL,
                    detail="An attached file from the same note also matched your query.",
                    weight=0.6,
                    source_unit_id=candidate.unit_id,
                )
            )


# --------------------------------------------------------------------------
# Re-ranking
# --------------------------------------------------------------------------


def rerank(candidates: list[Candidate]) -> list[Candidate]:
    """Re-score on relevance + contextual consistency + evidence quality.

    `base_score` (retrieval relevance) stays untouched and is what the UI
    shows as the match score; `score` is the ranking key. Keeping both means
    a user can see that a candidate ranked above another despite a lower raw
    match, and the evidence list says why.
    """
    for candidate in candidates:
        bonus = 0.0
        penalty = 0.0
        for item in candidate.evidence:
            if item.kind == KIND_CUE_MATCH:
                bonus += CUE_MATCH_BONUS * item.weight
            elif item.kind == KIND_SHARED_ENTITY:
                bonus += SHARED_ENTITY_BONUS * item.weight
            elif item.kind == KIND_TEMPORAL:
                if item.polarity == POLARITY_SUPPORTING:
                    bonus += TEMPORAL_MATCH_BONUS * item.weight
                else:
                    penalty += TEMPORAL_MISMATCH_PENALTY * item.weight
            elif item.kind == KIND_ATTACHMENT:
                bonus += SHARED_ENTITY_BONUS * item.weight
            elif item.kind == KIND_CONTRADICTION:
                # Small, not disqualifying: a contradiction means the store
                # disagrees with itself, which the answer must SAY (see
                # _ANSWER_SYSTEM) rather than resolve by hiding one side.
                penalty += CONTRADICTION_PENALTY * item.weight
            elif item.kind == KIND_DEGRADED:
                penalty += DEGRADED_PENALTY * item.weight

        bonus = min(bonus, MAX_EVIDENCE_BONUS)
        score = candidate.base_score + bonus - penalty
        if candidate.route == ROUTE_ASSOCIATION:
            score = min(score, ASSOCIATION_SCORE_CEILING)
        candidate.score = max(0.0, min(1.0, score))

    # Deterministic ordering: score, then direct-before-associated at equal
    # score, then id. Without the last two an identical query could return the
    # same candidates in a different order run to run.
    candidates.sort(
        key=lambda c: (-c.score, c.route == ROUTE_ASSOCIATION, c.unit_id)
    )
    return candidates


def apply_budget(candidates: list[Candidate], budget: RetrievalBudget) -> list[Candidate]:
    return candidates[: budget.max_total_candidates]
