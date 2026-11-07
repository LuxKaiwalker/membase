"""Tests for app/retrieval/associative.py -- retrieval level 6, associative and
iterative recall -- plus the RecallSession round that drives it.

DB-backed but LLM-free: the engine itself makes no model call (that is what
makes its evidence trail reproducible), so candidates are built from units
written straight to the database, exactly as tests/test_graph.py does for the
projection. The session-level tests at the bottom monkeypatch
`automatic_search` the same way tests/test_recall_session.py does, so the thing
under test is the round's own candidate reasoning rather than hybrid search's
numeric behaviour.
"""

import json

from sqlalchemy import select

from app.memory.graph import ASSOC_SHARED_ENTITY, _ordered_pair, project_unit
from app.models import ORIGIN_INFERRED, Capture, MemoryUnit, UnitAssociation
from app.retrieval import session as session_module
from app.retrieval.agent import (
    HIGH_CONFIDENCE_FUSED_SCORE,
    HIGH_CONFIDENCE_MARGIN,
    AutomaticSearchResult,
)
from app.retrieval.discriminate import POSITIVE_CUE_BOOST, apply_positive_cue_boost
from app.retrieval.associative import (
    ASSOCIATION_SCORE_CEILING,
    CUE_MATCH_BONUS,
    MAX_EVIDENCE_BONUS,
    SHARED_ENTITY_BONUS,
    TEMPORAL_MISMATCH_PENALTY,
    RetrievalBudget,
    apply_budget,
    build_direct_candidates,
    collect_evidence,
    expand_with_associations,
    rerank,
    should_expand,
)
from app.retrieval.evidence import (
    CONFIDENCE_INSUFFICIENT,
    CONFIDENCE_MODERATE,
    CONFIDENCE_STRONG,
    CONFIDENCE_WEAK,
    KIND_ASSOCIATION,
    KIND_CONTRADICTION,
    KIND_CUE_MATCH,
    KIND_DEGRADED,
    KIND_DIRECT_MATCH,
    KIND_LEXICAL_MATCH,
    KIND_SHARED_ENTITY,
    KIND_TEMPORAL,
    POLARITY_CONTRADICTING,
    POLARITY_SUPPORTING,
    PROVENANCE_EXTRACTED,
    PROVENANCE_HYPOTHESIS,
    PROVENANCE_INFERRED,
    PROVENANCE_ORIGINAL,
    ROUTE_ASSOCIATION,
    ROUTE_BOTH,
    ROUTE_DIRECT,
    Candidate,
    EvidenceItem,
)
from app.retrieval.pipeline import ScoredUnit
from app.retrieval.session import (
    STATUS_EXHAUSTED,
    STATUS_NEEDS_USER_CUE,
    STATUS_SUCCESS,
    continue_session,
    start_session,
)
from tests.fakes import FakeLLMProvider

_CONFIDENCE_CATEGORIES = {
    CONFIDENCE_STRONG,
    CONFIDENCE_MODERATE,
    CONFIDENCE_WEAK,
    CONFIDENCE_INSUFFICIENT,
}

_PROVENANCE_LEVELS = {
    PROVENANCE_ORIGINAL,
    PROVENANCE_EXTRACTED,
    PROVENANCE_INFERRED,
    PROVENANCE_HYPOTHESIS,
}


# --------------------------------------------------------------------------
# Fixtures, in the style of tests/test_graph.py
# --------------------------------------------------------------------------


def _ready_unit(
    db,
    *,
    text,
    semantic=None,
    capture=None,
    status="ready",
    retrieval_text=None,
    potential_conflicts=None,
):
    """A capture plus one unit, written straight to the DB."""
    if capture is None:
        capture = Capture(raw_text=text, explicit_metadata={})
        db.add(capture)
        db.flush()
    unit = MemoryUnit(
        capture_id=capture.id,
        authoritative_content=text,
        status=status,
        semantic=semantic or {},
        explicit_metadata={},
        retrieval_text=retrieval_text if retrieval_text is not None else text,
        searchable_text=text,
        potential_conflicts=potential_conflicts or [],
    )
    db.add(unit)
    db.commit()
    db.refresh(unit)
    return unit


def _associate(db, unit_a, unit_b, *, kind=ASSOC_SHARED_ENTITY, weight=0.6, explanation="Both mention Anna (person)."):
    """Ensure an association of this kind exists between two units. Idempotent,
    for the same reason tests/test_graph.py's version is: projection may
    already have written the same pair+kind."""
    source_id, target_id = _ordered_pair(unit_a.id, unit_b.id)
    existing = db.execute(
        select(UnitAssociation).where(
            UnitAssociation.source_unit_id == source_id,
            UnitAssociation.target_unit_id == target_id,
            UnitAssociation.kind == kind,
        )
    ).scalar_one_or_none()
    if existing is not None:
        existing.weight = weight
        existing.explanation = explanation
        db.commit()
        return
    db.add(
        UnitAssociation(
            source_unit_id=source_id,
            target_unit_id=target_id,
            kind=kind,
            origin=ORIGIN_INFERRED,
            weight=weight,
            shared_entity_ids=[],
            explanation=explanation,
        )
    )
    db.commit()


def _scored(unit, score, *, vector_score=None, fts_score=None):
    return ScoredUnit(
        unit=unit,
        vector_score=score if vector_score is None else vector_score,
        fts_score=score if fts_score is None else fts_score,
        fused_score=score,
    )


def _candidate(unit, score, *, route=ROUTE_DIRECT):
    """A candidate at a chosen base score, bypassing retrieval."""
    direct = _scored(unit, score) if route != ROUTE_ASSOCIATION else None
    return Candidate(unit=unit, route=route, direct=direct, base_score=score, score=score)


def _add_items(candidate, kind, count, *, polarity=POLARITY_SUPPORTING, weight=1.0, provenance=PROVENANCE_EXTRACTED):
    for _ in range(count):
        candidate.add(
            EvidenceItem(kind=kind, polarity=polarity, provenance=provenance, detail="because.", weight=weight)
        )


def _kinds(candidate):
    return [e.kind for e in candidate.evidence]


def _item_of_kind(candidate, kind):
    return next(e for e in candidate.evidence if e.kind == kind)


# --------------------------------------------------------------------------
# build_direct_candidates
# --------------------------------------------------------------------------


def test_a_directly_retrieved_unit_becomes_a_direct_candidate_with_match_evidence(db):
    unit = _ready_unit(db, text="Anna recommended a paper.", semantic={"people": ["Anna"]})

    candidates = build_direct_candidates([_scored(unit, 0.72)])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.route == ROUTE_DIRECT
    assert candidate.direct is not None
    # base_score is retrieval relevance and must arrive untouched -- it is what
    # the UI shows as the match score, separately from the re-ranked `score`.
    assert candidate.base_score == 0.72
    assert candidate.score == 0.72
    assert KIND_DIRECT_MATCH in _kinds(candidate)
    match = _item_of_kind(candidate, KIND_DIRECT_MATCH)
    assert match.polarity == POLARITY_SUPPORTING
    assert match.provenance == PROVENANCE_EXTRACTED
    assert match.source_unit_id == str(unit.id)
    assert "0.72" in match.detail


def test_a_lexical_overlap_is_recorded_separately_but_adds_no_weight(db):
    unit = _ready_unit(db, text="Anna recommended a paper.")

    with_fts = build_direct_candidates([_scored(unit, 0.60, fts_score=0.4)])[0]
    without_fts = build_direct_candidates([_scored(unit, 0.60, fts_score=0.0)])[0]

    assert KIND_LEXICAL_MATCH in _kinds(with_fts)
    assert KIND_LEXICAL_MATCH not in _kinds(without_fts)
    # Zero weight: the lexical overlap is already inside the fused score, so
    # counting it again in the evidence bonus would double-count it.
    assert _item_of_kind(with_fts, KIND_LEXICAL_MATCH).weight == 0.0


# --------------------------------------------------------------------------
# should_expand
# --------------------------------------------------------------------------


def test_expansion_is_skipped_when_one_memory_already_won_clearly(db):
    """Same three-signal test as the agent's sufficiency short-circuit
    (app/retrieval/agent.py::_is_high_confidence): dominant absolute score,
    clear margin, complete semantics. A clear query must not pay for
    exploration it cannot benefit from."""
    top = _ready_unit(db, text="Anna recommended a paper.", semantic={"meaning": "a paper"})
    other = _ready_unit(db, text="Markus recommended a book.")
    candidates = [
        _candidate(top, HIGH_CONFIDENCE_FUSED_SCORE),
        _candidate(other, HIGH_CONFIDENCE_FUSED_SCORE - HIGH_CONFIDENCE_MARGIN - 0.01),
    ]

    expand, reason = should_expand(candidates)

    assert expand is False
    # The reason is recorded in the trace so a user inspecting a fast answer
    # can see exploration was skipped deliberately, not forgotten.
    assert isinstance(reason, str) and reason.strip()


def test_expansion_is_skipped_when_nothing_matched_at_all(db):
    expand, reason = should_expand([])

    assert expand is False
    assert isinstance(reason, str) and reason.strip()
    assert "nothing" in reason


def test_expansion_runs_when_the_candidates_are_close_together(db):
    top = _ready_unit(db, text="Anna recommended a paper.")
    other = _ready_unit(db, text="Markus recommended a book.")
    candidates = [_candidate(top, 0.90), _candidate(other, 0.85)]

    expand, reason = should_expand(candidates)

    assert expand is True
    assert reason is None


def test_expansion_runs_for_a_high_scoring_candidate_whose_semantics_are_degraded(db):
    """A unit whose enrichment never completed was matched against incomplete
    information, so a high score is not grounds to stop looking."""
    top = _ready_unit(
        db, text="Anna recommended a paper.", semantic={"enrichment_status": "unavailable"}
    )
    candidates = [_candidate(top, 0.98)]

    expand, reason = should_expand(candidates)

    assert expand is True
    assert reason is None
    # Without the degradation it would have been skipped, so the status is
    # genuinely what flipped the decision.
    top.semantic = {"meaning": "a paper"}
    db.commit()
    assert should_expand([_candidate(top, 0.98)])[0] is False


# --------------------------------------------------------------------------
# expand_with_associations
# --------------------------------------------------------------------------


def test_a_unit_reachable_only_by_association_becomes_a_new_association_candidate(db):
    seed = _ready_unit(db, text="Hiking with Anna at Lake Bled.", semantic={"people": ["Anna"]})
    reachable = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    _associate(db, seed, reachable, weight=0.6)
    candidates = [_candidate(seed, 0.90)]

    expanded = expand_with_associations(db, candidates, RetrievalBudget())

    assert [c.unit_id for c in expanded] == [str(reachable.id)]
    new = expanded[0]
    assert new.route == ROUTE_ASSOCIATION
    assert new.direct is None
    assert new.base_score == 0.90 * 0.6
    # Every association candidate has to be able to explain itself back to a
    # memory that actually matched, or it is an unexplainable result.
    assert new.association.why == "Both mention Anna (person)."
    item = _item_of_kind(new, KIND_ASSOCIATION)
    assert item.provenance == PROVENANCE_INFERRED
    assert "Both mention Anna (person)." in item.detail
    assert "1 hop" in item.detail


def test_a_unit_that_already_matched_directly_becomes_route_both_instead_of_a_duplicate(db):
    """A memory that both matched and is associated with another match is
    better supported -- but it is still a direct match, not an inferred one,
    and it must not appear twice in the candidate set."""
    seed = _ready_unit(db, text="Hiking with Anna at Lake Bled.", semantic={"people": ["Anna"]})
    # Below min_seed_score_for_expansion, so it is a candidate but not a seed
    # (a seed is never returned as its own neighbour).
    weak_direct = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    _associate(db, seed, weak_direct, weight=0.6)
    candidates = [_candidate(seed, 0.90), _candidate(weak_direct, 0.20)]

    expanded = expand_with_associations(db, candidates, RetrievalBudget())

    assert expanded == []
    assert len(candidates) == 2
    assert candidates[1].route == ROUTE_BOTH
    assert candidates[1].direct is not None
    assert candidates[1].base_score == 0.20  # retrieval relevance is unchanged
    assert KIND_ASSOCIATION in _kinds(candidates[1])
    assert "also connected to another matching memory" in _item_of_kind(candidates[1], KIND_ASSOCIATION).detail
    assert [c.unit_id for c in candidates].count(str(weak_direct.id)) == 1


def test_expansion_respects_the_expanded_candidate_budget(db):
    seed = _ready_unit(db, text="Hiking with Anna at Lake Bled.", semantic={"people": ["Anna"]})
    others = [_ready_unit(db, text=f"Anna memory {i}.", semantic={"people": ["Anna"]}) for i in range(3)]
    for other in others:
        _associate(db, seed, other, weight=0.6)
    candidates = [_candidate(seed, 0.90)]

    assert len(expand_with_associations(db, candidates, RetrievalBudget(max_expanded_candidates=1))) == 1
    assert len(expand_with_associations(db, [_candidate(seed, 0.90)], RetrievalBudget())) == 3


def test_expansion_still_explores_from_the_best_weak_candidate(db):
    """A weak cue is exactly the case associative recall exists for, so when no
    candidate clears the seed bar the single best one is used anyway rather
    than giving up."""
    seed = _ready_unit(db, text="Hiking with Anna at Lake Bled.", semantic={"people": ["Anna"]})
    reachable = _ready_unit(db, text="Anna sent me a paper.", semantic={"people": ["Anna"]})
    _associate(db, seed, reachable, weight=0.9)
    budget = RetrievalBudget()
    candidates = [_candidate(seed, 0.25)]
    assert candidates[0].base_score < budget.min_seed_score_for_expansion

    expanded = expand_with_associations(db, candidates, budget)

    assert [c.unit_id for c in expanded] == [str(reachable.id)]


# --------------------------------------------------------------------------
# collect_evidence
# --------------------------------------------------------------------------


def test_a_confirmed_cue_appearing_in_the_memorys_own_text_is_original_provenance(db):
    unit = _ready_unit(
        db,
        text="We discussed the Phoenix migration over lunch.",
        retrieval_text="Original:\nWe discussed the Phoenix migration over lunch.",
    )
    candidates = [_candidate(unit, 0.50)]

    collect_evidence(db, candidates, query="what did we talk about at lunch?", positive_cues=["Phoenix migration"])

    item = _item_of_kind(candidates[0], KIND_CUE_MATCH)
    # That the stored text contains the user's own confirmed words is a fact
    # about stored text, not an inference -- hence "original", not "inferred".
    assert item.provenance == PROVENANCE_ORIGINAL
    assert item.polarity == POLARITY_SUPPORTING
    assert "Phoenix migration" in item.detail


def test_a_cue_that_appears_nowhere_adds_no_evidence(db):
    unit = _ready_unit(db, text="We discussed the Phoenix migration over lunch.")
    candidates = [_candidate(unit, 0.50)]

    collect_evidence(db, candidates, query="what did we talk about?", positive_cues=["kayaking"])

    assert KIND_CUE_MATCH not in _kinds(candidates[0])


def test_an_entity_the_query_names_becomes_shared_entity_evidence(db):
    unit = _ready_unit(db, text="Anna recommended a paper.", semantic={"people": ["Anna"]})
    project_unit(db, unit)
    db.commit()
    candidates = [_candidate(unit, 0.50)]

    collect_evidence(db, candidates, query="What did Anna recommend to me?")

    item = _item_of_kind(candidates[0], KIND_SHARED_ENTITY)
    assert item.provenance == PROVENANCE_EXTRACTED
    assert item.entity_id is not None
    assert "Anna" in item.detail
    assert "your query" in item.detail


def test_a_stored_conflict_marker_becomes_contradicting_evidence(db):
    other = _ready_unit(db, text="It was Boris who recommended it.")
    unit = _ready_unit(
        db,
        text="Anna recommended a paper.",
        potential_conflicts=[{"note": "Another memory says Boris recommended it.", "unit_id": str(other.id)}],
    )
    candidates = [_candidate(unit, 0.90)]

    collect_evidence(db, candidates, query="Who recommended the paper?")

    item = _item_of_kind(candidates[0], KIND_CONTRADICTION)
    assert item.polarity == POLARITY_CONTRADICTING
    assert item.detail == "Another memory says Boris recommended it."
    assert item.source_unit_id == str(other.id)


def test_incomplete_enrichment_is_recorded_as_degraded_evidence(db):
    unit = _ready_unit(db, text="Anna recommended a paper.", semantic={"enrichment_status": "unavailable"})
    candidates = [_candidate(unit, 0.90)]

    collect_evidence(db, candidates, query="What did Anna recommend?")

    item = _item_of_kind(candidates[0], KIND_DEGRADED)
    assert item.polarity == POLARITY_CONTRADICTING
    assert item.provenance == PROVENANCE_INFERRED
    assert "never completed" in item.detail


def test_a_date_in_the_query_supports_a_compatible_memory_and_contradicts_an_incompatible_one(db):
    fits = _ready_unit(db, text="Anna recommended a paper.", semantic={"event_date": "2024-03-15"})
    does_not_fit = _ready_unit(db, text="Markus recommended a book.", semantic={"event_date": "2023-07-10"})
    candidates = [_candidate(fits, 0.50), _candidate(does_not_fit, 0.50)]

    collect_evidence(db, candidates, query="What was recommended to me in March 2024?")

    supporting = _item_of_kind(candidates[0], KIND_TEMPORAL)
    contradicting = _item_of_kind(candidates[1], KIND_TEMPORAL)
    assert supporting.polarity == POLARITY_SUPPORTING
    assert contradicting.polarity == POLARITY_CONTRADICTING
    # Rendered at no more precision than is actually known, so a month-level
    # query never displays as a specific day.
    assert "March 2024" in supporting.detail
    assert "does not fit" in contradicting.detail


def test_an_undated_memory_gets_no_temporal_evidence_either_way(db):
    """An unknown date is compatible with nothing -- including with another
    unknown date -- so silence is the only honest outcome here."""
    undated = _ready_unit(db, text="Anna recommended a paper.", semantic={"event_date": ""})
    candidates = [_candidate(undated, 0.50)]

    collect_evidence(db, candidates, query="What was recommended to me in March 2024?")

    assert KIND_TEMPORAL not in _kinds(candidates[0])


def test_every_collected_evidence_item_carries_a_known_provenance_level(db):
    unit = _ready_unit(
        db,
        text="Anna recommended a paper over lunch.",
        semantic={"people": ["Anna"], "event_date": "2024-03-15", "enrichment_status": "unavailable"},
        retrieval_text="Original:\nAnna recommended a paper over lunch.",
        potential_conflicts=[{"note": "Another memory disagrees."}],
    )
    project_unit(db, unit)
    db.commit()
    candidates = build_direct_candidates([_scored(unit, 0.60)])

    collect_evidence(
        db, candidates, query="What did Anna recommend in March 2024?", positive_cues=["lunch"]
    )

    assert len(candidates[0].evidence) > 4
    assert {e.provenance for e in candidates[0].evidence} <= _PROVENANCE_LEVELS


# --------------------------------------------------------------------------
# rerank
# --------------------------------------------------------------------------


def test_a_cue_match_raises_a_candidates_score_above_an_identical_rival(db):
    """A confirmed cue does lift a candidate over an otherwise-identical rival
    -- but the lift is applied by `apply_positive_cue_boost`, upstream of
    re-ranking, not by `rerank`. Asserted at the layer that actually does it,
    so the test cannot pass while the behaviour has silently moved.
    """
    cued_unit = _ready_unit(db, text="Anna recommended a paper.", semantic={"people": ["Anna"]})
    plain_unit = _ready_unit(db, text="Markus recommended a book.", semantic={"people": ["Markus"]})
    ranked = [_scored(cued_unit, 0.50), _scored(plain_unit, 0.50)]

    boosted = apply_positive_cue_boost(ranked, ["Anna"])
    by_id = {str(s.unit.id): s for s in boosted}

    cued = by_id[str(cued_unit.id)]
    plain = by_id[str(plain_unit.id)]
    assert cued.fused_score > plain.fused_score
    assert cued.fused_score == 0.50 + POSITIVE_CUE_BOOST
    assert plain.fused_score == 0.50
    # The pre-boost score is kept, so confidence can still be judged on how
    # well the memory itself matched rather than on the user's corroboration.
    assert cued.prior_fused_score == 0.50
    assert plain.prior_fused_score == 0.50

    # And re-ranking does not add the same signal a second time.
    candidates = build_direct_candidates(boosted)
    for candidate in candidates:
        _add_items(candidate, KIND_CUE_MATCH, 1, provenance=PROVENANCE_ORIGINAL)
    rerank(candidates)
    assert {c.score for c in candidates} == {0.50 + POSITIVE_CUE_BOOST, 0.50}


def test_a_temporal_mismatch_lowers_a_candidates_score(db):
    mismatched = _candidate(_ready_unit(db, text="Markus recommended a book."), 0.50)
    _add_items(mismatched, KIND_TEMPORAL, 1, polarity=POLARITY_CONTRADICTING)

    rerank([mismatched])

    assert mismatched.score == 0.50 - TEMPORAL_MISMATCH_PENALTY


def test_many_weak_supporting_items_cannot_overtake_a_decisively_better_match(db):
    """The evidence bonus refines an ordering; it does not override it. Without
    the cap, ten shared-entity matches would be worth 0.6 and a badly-matching
    memory could win on accumulated bonuses alone."""
    padded = _candidate(_ready_unit(db, text="Anna recommended a paper."), 0.50)
    better = _candidate(_ready_unit(db, text="Markus recommended a book."), 0.80)
    _add_items(padded, KIND_SHARED_ENTITY, 10)

    ordered = rerank([padded, better])

    assert padded.score == 0.50 + MAX_EVIDENCE_BONUS
    assert 10 * SHARED_ENTITY_BONUS > MAX_EVIDENCE_BONUS  # the cap really bound here
    assert ordered[0] is better
    assert better.score > padded.score


def test_a_confirmed_cue_does_not_add_to_the_score_because_it_is_priced_in_upstream(db):
    """A cue match is worth 0 in `rerank` on purpose.

    `apply_positive_cue_boost` has already added POSITIVE_CUE_BOOST (0.3) to
    the candidate's fused score for exactly this signal. Rewarding it again
    here -- and a third time via the shared-entity item that the same cue
    produces -- counted one signal three times, which was observed live
    pinning a memory that merely shared a person with the query to the score
    ceiling. The evidence item is still recorded, because it belongs in the
    explanation.
    """
    candidate = _candidate(_ready_unit(db, text="Anna recommended a paper."), 0.50)
    _add_items(candidate, KIND_CUE_MATCH, 3, provenance=PROVENANCE_ORIGINAL)

    rerank([candidate])

    assert CUE_MATCH_BONUS == 0.0
    assert candidate.score == 0.50
    # Shown, not scored.
    assert len([e for e in candidate.evidence if e.kind == KIND_CUE_MATCH]) == 3


def test_an_association_candidate_can_never_exceed_the_association_ceiling(db):
    """**The guarantee that an inferred lead cannot outrank a real match.**

    An association-route candidate is a memory that never matched the query:
    it was suggested by one that did. No amount of accumulated supporting
    evidence -- and no association strength, however high -- may lift it past
    ASSOCIATION_SCORE_CEILING, which sits below the score a merely moderate
    direct match reaches.
    """
    lead = _candidate(_ready_unit(db, text="Anna sent me a paper."), 0.50, route=ROUTE_ASSOCIATION)
    very_strong_lead = _candidate(
        _ready_unit(db, text="Anna sent me a second paper."), 0.99, route=ROUTE_ASSOCIATION
    )
    for candidate in (lead, very_strong_lead):
        _add_items(candidate, KIND_CUE_MATCH, 10, provenance=PROVENANCE_ORIGINAL)
        _add_items(candidate, KIND_SHARED_ENTITY, 5)
        _add_items(candidate, KIND_TEMPORAL, 1)
    real_match = _candidate(_ready_unit(db, text="Markus recommended a book."), 0.60)

    ordered = rerank([lead, very_strong_lead, real_match])

    assert lead.score == ASSOCIATION_SCORE_CEILING
    assert very_strong_lead.score == ASSOCIATION_SCORE_CEILING
    # A direct match with NO supporting evidence at all still wins.
    assert ordered[0] is real_match
    assert real_match.score > ASSOCIATION_SCORE_CEILING


def test_ordering_is_deterministic_and_puts_a_direct_match_first_at_equal_score(db):
    direct = _candidate(_ready_unit(db, text="Anna recommended a paper."), ASSOCIATION_SCORE_CEILING)
    associated = _candidate(
        _ready_unit(db, text="Anna sent me a paper."), ASSOCIATION_SCORE_CEILING, route=ROUTE_ASSOCIATION
    )
    other = _candidate(_ready_unit(db, text="Markus recommended a book."), 0.30)

    first = [c.unit_id for c in rerank([associated, direct, other])]
    second = [c.unit_id for c in rerank([other, direct, associated])]
    third = [c.unit_id for c in rerank([direct, other, associated])]

    assert direct.score == associated.score
    assert first[0] == direct.unit_id
    # An identical query must not return the same candidates in a different
    # order run to run, whatever order they arrived in.
    assert first == second == third


# --------------------------------------------------------------------------
# apply_budget
# --------------------------------------------------------------------------


def test_apply_budget_truncates_to_the_total_candidate_ceiling(db):
    candidates = [
        _candidate(_ready_unit(db, text=f"memory {i}"), 0.9 - i * 0.1) for i in range(5)
    ]

    kept = apply_budget(candidates, RetrievalBudget(max_total_candidates=2))

    # Bounds prompt size, which is the real cost driver.
    assert [c.unit_id for c in kept] == [c.unit_id for c in candidates[:2]]
    assert len(apply_budget(candidates, RetrievalBudget())) == 5


# --------------------------------------------------------------------------
# Session-level integration
# --------------------------------------------------------------------------


def _patch_rounds(monkeypatch, results):
    """Each call to automatic_search returns the next result in order (the last
    repeats if more rounds happen than configured) -- same pattern as
    tests/test_recall_session.py."""
    state = {"n": 0}

    async def _fake(db, llm, query, *, extra_queries=None, tried=None):
        idx = min(state["n"], len(results) - 1)
        state["n"] += 1
        return results[idx]

    monkeypatch.setattr(session_module, "automatic_search", _fake)


def _search_result(ranked):
    return AutomaticSearchResult(ranked=ranked, ranked_chunks=[], tried_queries=set(), hypotheses=[])


async def test_a_successful_round_records_its_strategies_confidence_and_evidence(db, monkeypatch):
    unit = _ready_unit(
        db,
        text="Anna recommended a tactile sensing paper.",
        semantic={"meaning": "Anna recommended a paper", "people": ["Anna"]},
    )
    _patch_rounds(monkeypatch, [_search_result([_scored(unit, 0.92)])])
    llm = FakeLLMProvider(answer_response={"answer": "Anna recommended a paper.", "grounded": True})

    session = await start_session(db, llm, "What did Anna recommend?")

    assert session.status == STATUS_SUCCESS
    assert session.strategy_trail
    entry = session.strategy_trail[-1]
    assert entry["round"] == 1
    assert "hybrid_search" in entry["strategies"]
    assert "evidence_collection" in entry["strategies"]
    assert entry["direct_candidates"] == 1
    assert entry["budget"]["max_total_candidates"] == RetrievalBudget().max_total_candidates
    # A category, never a bare percentage.
    assert session.final_confidence in _CONFIDENCE_CATEGORIES
    assert session.final_evidence
    assert json.loads(json.dumps(session.final_evidence)) == session.final_evidence
    assert session.final_evidence[0]["route"] == ROUTE_DIRECT
    assert session.final_evidence[0]["confidence"] in _CONFIDENCE_CATEGORIES
    assert session.final_evidence[0]["evidence"]
    assert session.abstention_reason is None


async def test_a_skipped_expansion_is_recorded_with_its_reason(db, monkeypatch):
    unit = _ready_unit(db, text="Anna recommended a paper.", semantic={"meaning": "a paper"})
    _patch_rounds(monkeypatch, [_search_result([_scored(unit, 0.95)])])
    llm = FakeLLMProvider(answer_response={"answer": "Anna recommended a paper.", "grounded": True})

    session = await start_session(db, llm, "What did Anna recommend?")

    entry = session.strategy_trail[-1]
    assert entry["expansion_skipped_reason"]
    assert "association_expansion" not in entry["strategies"]
    assert entry["expanded_candidates"] == 0


async def test_an_associated_memory_reaches_the_answers_evidence_capped_at_weak(db, monkeypatch):
    seed = _ready_unit(
        db, text="Hiking with Anna at Lake Bled.", semantic={"meaning": "a hike", "people": ["Anna"]}
    )
    associated = _ready_unit(db, text="Anna sent me a paper afterwards.", semantic={"people": ["Anna"]})
    _associate(db, seed, associated, weight=0.8)
    # Mid-range score: not dominant enough to skip exploration.
    _patch_rounds(monkeypatch, [_search_result([_scored(seed, 0.50)])])
    llm = FakeLLMProvider(answer_response={"answer": "You hiked with Anna at Lake Bled.", "grounded": True})

    session = await start_session(db, llm, "Where did I hike with Anna?")

    assert session.status == STATUS_SUCCESS
    entry = session.strategy_trail[-1]
    assert "association_expansion" in entry["strategies"]
    assert entry["expanded_candidates"] == 1
    by_route = {e["route"]: e for e in session.final_evidence}
    assert ROUTE_ASSOCIATION in by_route
    lead = by_route[ROUTE_ASSOCIATION]
    # Reaching a memory by inference is a lead, not a finding.
    assert lead["confidence"] == CONFIDENCE_WEAK
    assert lead["association_why"]
    assert lead["score"] <= ASSOCIATION_SCORE_CEILING


async def test_abstaining_because_nothing_matched_explains_itself_and_still_records_the_round(db, monkeypatch):
    _patch_rounds(monkeypatch, [_search_result([])])
    llm = FakeLLMProvider()

    session = await start_session(db, llm, "something nobody ever told me")

    assert session.status == STATUS_EXHAUSTED
    assert session.final_answer is None
    # The diagnosable reason, distinct from the fixed abstention message shown
    # to the user -- an abstention with no recorded reason is not inspectable.
    assert isinstance(session.abstention_reason, str)
    assert session.abstention_reason.strip()
    assert "relevance bar" in session.abstention_reason
    assert len(session.strategy_trail) == 1
    assert session.strategy_trail[0]["round"] == 1
    assert session.strategy_trail[0]["direct_candidates"] == 0
    assert session.strategy_trail[0]["expanded_candidates"] == 0


async def test_a_candidate_the_user_ruled_out_is_named_in_the_rounds_trace(db, monkeypatch):
    unit_a = _ready_unit(
        db,
        text="Anna recommended a paper to me.",
        semantic={"meaning": "Anna recommended a paper", "people": ["Anna"], "locations": ["ETH"]},
    )
    unit_b = _ready_unit(
        db,
        text="Markus recommended a book to me.",
        semantic={"meaning": "Markus recommended a book", "people": ["Markus"], "locations": ["Paris"]},
    )
    ambiguous = _search_result([_scored(unit_a, 0.50), _scored(unit_b, 0.49)])
    _patch_rounds(monkeypatch, [ambiguous, ambiguous])
    llm = FakeLLMProvider(
        cue_response={"cues": [{"text": "Anna", "confidence": "certain", "polarity": "negative"}]},
        answer_response={"answer": "Markus recommended a book.", "grounded": True},
    )

    session = await start_session(db, llm, "Who recommended something to me?")
    assert session.status == STATUS_NEEDS_USER_CUE

    session = await continue_session(db, llm, session, "No, it wasn't Anna.")

    assert session.status == STATUS_SUCCESS
    entry = session.strategy_trail[-1]
    # Excluding silently would make a narrowing session look like it simply
    # stopped finding things, when the user is the one who ruled them out.
    assert [r["unit_id"] for r in entry["ruled_out"]] == [str(unit_a.id)]
    assert entry["ruled_out"][0]["reason"] == "you ruled this out"
    assert "Anna recommended a paper" in entry["ruled_out"][0]["content"]
    assert {e["unit_id"] for e in session.final_evidence} == {str(unit_b.id)}


async def test_the_strategy_trail_accumulates_one_entry_per_round(db, monkeypatch):
    """Appended rather than replaced: the point of the trail is showing how the
    search narrowed across rounds, which a latest-round-only field cannot."""
    unit_a = _ready_unit(
        db, text="Anna recommended a paper.", semantic={"people": ["Anna"], "locations": ["ETH"]}
    )
    unit_b = _ready_unit(
        db, text="Markus recommended a book.", semantic={"people": ["Markus"], "locations": ["Paris"]}
    )
    ambiguous = _search_result([_scored(unit_a, 0.50), _scored(unit_b, 0.49)])
    _patch_rounds(monkeypatch, [ambiguous, ambiguous])
    llm = FakeLLMProvider(cue_response={"cues": []})

    session = await start_session(db, llm, "Who recommended something to me?")
    session = await continue_session(db, llm, session, "hmm, not sure")

    assert [e["round"] for e in session.strategy_trail] == [1, 2]
    # Each asking round records the question it chose, including the expected
    # gain that justified choosing it.
    for entry in session.strategy_trail:
        assert entry["question"]["question"] in session.asked_questions
        assert "expected_gain_bits_approx" in entry["question"]
    assert json.loads(json.dumps(session.strategy_trail)) == session.strategy_trail


async def test_a_session_round_actually_expands_through_the_graph(db, monkeypatch):
    """The end-to-end property that was silently broken for the entire product
    path: a round with ambiguous direct candidates and a real association must
    put an association-route candidate in front of the user.

    It was not broken in `expand_with_associations` itself -- every unit test
    of that function passed throughout. It was broken at the *call site*:
    `session.py` handed it the time the round began, which is before
    `automatic_search`'s LLM calls, so a 13-57s round always blew the 10s
    traversal budget and the loop broke on its first neighbour. Every session
    reported `expanded_candidates: 0` while the same graph returned eleven
    neighbours when queried directly.

    This test asserts the behaviour at the session boundary, which is where the
    defect lived, rather than at the function boundary, which was fine.
    """
    seed = _ready_unit(
        db,
        text="Took this photo on the hike above Lake Bled with Markus.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"]},
    )
    rival = _ready_unit(
        db,
        text="A weekend at Lake Thun with Nadia.",
        semantic={"people": ["Nadia"], "locations": ["Lake Thun"]},
    )
    # Shares no query wording -- reachable ONLY by association from the seed.
    neighbour = _ready_unit(
        db,
        text="Markus is doing a doctorate in glaciology at ETH Zurich.",
        semantic={"people": ["Markus"], "organizations": ["ETH Zurich"]},
    )
    _associate(db, seed, neighbour, weight=0.8, explanation="Both mention Markus (person).")

    # Two close candidates from different captures, so `should_expand` says yes.
    _patch_rounds(monkeypatch, [_search_result([_scored(seed, 0.62), _scored(rival, 0.60)])])
    llm = FakeLLMProvider(answer_response={"answer": "The Lake Bled photo.", "grounded": True})

    session = await start_session(db, llm, "a photo near a lake with mountains")

    entry = session.strategy_trail[-1]
    assert "association_expansion" in entry["strategies"]
    assert entry["expansion_skipped_reason"] is None
    assert entry["expanded_candidates"] >= 1, "the graph walk contributed nothing"

    routes = {c["unit_id"]: c["route"] for c in session.final_evidence}
    assert routes.get(str(neighbour.id)) == ROUTE_ASSOCIATION
    reached = next(c for c in session.final_evidence if c["unit_id"] == str(neighbour.id))
    # And it arrives explainable, not just present.
    assert reached["association_why"]
    assert reached["confidence"] in {CONFIDENCE_WEAK, CONFIDENCE_INSUFFICIENT}
