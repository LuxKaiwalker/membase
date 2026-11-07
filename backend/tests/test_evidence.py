"""Tests for app/retrieval/evidence.py -- structured evidence, the four
confidence categories, and the evidence block handed to answer generation.

Pure unit tests. Candidates are built by hand over a minimal fake unit: every
rule in this module is a function of (route, direct fused score, evidence
kinds, capture grouping) and nothing else, so paying for ingestion or retrieval
here would only obscure which input drives which outcome.
"""

import datetime as dt
import json

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
    KIND_SHARED_ENTITY,
    MODERATE_DIRECT_SCORE,
    POLARITY_CONTRADICTING,
    POLARITY_NEUTRAL,
    POLARITY_SUPPORTING,
    PROVENANCE_EXTRACTED,
    PROVENANCE_HYPOTHESIS,
    PROVENANCE_INFERRED,
    PROVENANCE_ORIGINAL,
    ROUTE_ASSOCIATION,
    ROUTE_BOTH,
    ROUTE_DIRECT,
    STRONG_DIRECT_SCORE,
    WEAK_SCORE_FLOOR,
    Candidate,
    EvidenceItem,
    confidence_for,
    format_evidence_block,
    overall_confidence,
    provenance_label,
)
from app.retrieval.pipeline import ScoredUnit

_PROVENANCE_LEVELS = {
    PROVENANCE_ORIGINAL,
    PROVENANCE_EXTRACTED,
    PROVENANCE_INFERRED,
    PROVENANCE_HYPOTHESIS,
}

_CONFIDENCE_CATEGORIES = {
    CONFIDENCE_STRONG,
    CONFIDENCE_MODERATE,
    CONFIDENCE_WEAK,
    CONFIDENCE_INSUFFICIENT,
}


class _FakeUnit:
    def __init__(self, unit_id, capture_id, content, semantic=None, created_at=None):
        self.id = unit_id
        self.capture_id = capture_id
        self.authoritative_content = content
        self.semantic = semantic or {}
        self.created_at = created_at or dt.datetime(2024, 3, 15, 9, 30, tzinfo=dt.UTC)


class _FakeAssociation:
    """Stands in for app.memory.graph.AssociatedUnit, which Candidate holds
    untyped to avoid a circular import."""

    def __init__(self, strength, hops=1, why="Both mention Anna (person)."):
        self.strength = strength
        self.hops = hops
        self.why = why


def _direct_candidate(score, *, unit_id="unit-1", capture_id="cap-1", semantic=None, content="Anna recommended a paper."):
    unit = _FakeUnit(unit_id, capture_id, content, semantic=semantic)
    scored = ScoredUnit(unit=unit, vector_score=score, fts_score=score, fused_score=score)
    return Candidate(unit=unit, route=ROUTE_DIRECT, direct=scored, base_score=score, score=score)


def _association_candidate(strength, *, unit_id="unit-2", capture_id="cap-2", why="Both mention Anna (person)."):
    unit = _FakeUnit(unit_id, capture_id, "Anna sent me a link.")
    return Candidate(
        unit=unit,
        route=ROUTE_ASSOCIATION,
        association=_FakeAssociation(strength, why=why),
        base_score=strength,
        score=strength,
    )


def _item(kind, *, polarity=POLARITY_SUPPORTING, provenance=PROVENANCE_EXTRACTED, detail="because.", weight=1.0):
    return EvidenceItem(kind=kind, polarity=polarity, provenance=provenance, detail=detail, weight=weight)


# --------------------------------------------------------------------------
# confidence_for
# --------------------------------------------------------------------------


def test_a_candidate_reached_only_by_association_is_capped_at_weak():
    """It was not matched; it was suggested by something that was. A strong
    association is still an inference, so no association strength -- however
    high -- promotes it past `weak`."""
    strong_link = _association_candidate(0.95)

    assert confidence_for(strong_link) == CONFIDENCE_WEAK
    # ...and it stays capped even once it has accumulated supporting evidence.
    strong_link.add(_item(KIND_CUE_MATCH, provenance=PROVENANCE_ORIGINAL))
    strong_link.add(_item(KIND_SHARED_ENTITY))
    assert confidence_for(strong_link) == CONFIDENCE_WEAK


def test_a_very_weak_association_is_insufficient_rather_than_weak():
    assert confidence_for(_association_candidate(WEAK_SCORE_FLOOR)) == CONFIDENCE_WEAK
    assert confidence_for(_association_candidate(WEAK_SCORE_FLOOR - 0.01)) == CONFIDENCE_INSUFFICIENT
    assert confidence_for(_association_candidate(0.0)) == CONFIDENCE_INSUFFICIENT


def test_a_high_direct_score_with_nothing_against_it_is_strong():
    candidate = _direct_candidate(0.92)
    candidate.add(_item(KIND_DIRECT_MATCH))

    assert confidence_for(candidate) == CONFIDENCE_STRONG
    assert candidate.confidence == CONFIDENCE_STRONG


def test_a_stored_contradiction_caps_even_a_perfect_match_at_moderate():
    """A high retrieval score is not evidence that the *other* memory is
    wrong, so the score cannot overrule a disagreement in the store."""
    candidate = _direct_candidate(1.0)
    candidate.add(_item(KIND_DIRECT_MATCH))
    candidate.add(
        _item(KIND_CONTRADICTION, polarity=POLARITY_CONTRADICTING, detail="Another memory says it was Boris.")
    )

    assert confidence_for(candidate) == CONFIDENCE_MODERATE


def test_degraded_semantics_caps_a_high_score_at_moderate():
    """The memory is real, but the representation the match was made against
    is incomplete (enrichment never finished), so the match means less than
    its score suggests."""
    candidate = _direct_candidate(0.95, semantic={"enrichment_status": "unavailable"})
    candidate.add(_item(KIND_DIRECT_MATCH))
    candidate.add(_item(KIND_DEGRADED, polarity=POLARITY_CONTRADICTING, provenance=PROVENANCE_INFERRED))

    assert confidence_for(candidate) == CONFIDENCE_MODERATE


def test_the_direct_score_thresholds_define_the_remaining_categories():
    assert confidence_for(_direct_candidate(STRONG_DIRECT_SCORE)) == CONFIDENCE_STRONG
    assert confidence_for(_direct_candidate(0.65)) == CONFIDENCE_MODERATE
    assert confidence_for(_direct_candidate(MODERATE_DIRECT_SCORE)) == CONFIDENCE_MODERATE
    assert confidence_for(_direct_candidate(0.35)) == CONFIDENCE_WEAK
    assert confidence_for(_direct_candidate(WEAK_SCORE_FLOOR)) == CONFIDENCE_WEAK
    assert confidence_for(_direct_candidate(0.02)) == CONFIDENCE_INSUFFICIENT


def test_a_mid_scoring_match_with_a_contradiction_drops_a_further_step():
    candidate = _direct_candidate(0.65)
    candidate.add(_item(KIND_CONTRADICTION, polarity=POLARITY_CONTRADICTING))

    assert confidence_for(candidate) == CONFIDENCE_WEAK


def test_a_candidate_that_both_matched_and_is_associated_is_judged_as_a_match():
    """route="both" means the memory matched the query *and* turned out to be
    connected to another match. It is better supported than a plain direct
    hit, and must not be demoted to the association cap."""
    candidate = _direct_candidate(0.90)
    candidate.route = ROUTE_BOTH
    candidate.association = _FakeAssociation(0.4)
    candidate.add(_item(KIND_ASSOCIATION, provenance=PROVENANCE_INFERRED))

    assert confidence_for(candidate) == CONFIDENCE_STRONG


# --------------------------------------------------------------------------
# overall_confidence
# --------------------------------------------------------------------------


def test_two_close_candidates_from_different_notes_downgrade_a_strong_answer():
    """Confidence in the *answer*, not in the best candidate: two comparably
    strong memories from different original notes means the system does not
    actually know which one the user meant."""
    best = _direct_candidate(0.92, unit_id="unit-1", capture_id="cap-1")
    rival = _direct_candidate(0.88, unit_id="unit-2", capture_id="cap-2")
    best.score, rival.score = 0.92, 0.88

    assert confidence_for(best) == CONFIDENCE_STRONG
    assert overall_confidence([best, rival]) == CONFIDENCE_MODERATE


def test_two_close_candidates_from_the_same_note_do_not_downgrade_the_answer():
    """Sibling units of one submission are one episode, not an ambiguity --
    there is no "which did you mean?" to resolve between them."""
    best = _direct_candidate(0.92, unit_id="unit-1", capture_id="cap-1")
    sibling = _direct_candidate(0.88, unit_id="unit-2", capture_id="cap-1")
    best.score, sibling.score = 0.92, 0.88

    assert overall_confidence([best, sibling]) == CONFIDENCE_STRONG


def test_a_clearly_separated_runner_up_does_not_downgrade_the_answer():
    best = _direct_candidate(0.92, unit_id="unit-1", capture_id="cap-1")
    distant = _direct_candidate(0.40, unit_id="unit-2", capture_id="cap-2")
    best.score, distant.score = 0.92, 0.40

    assert overall_confidence([best, distant]) == CONFIDENCE_STRONG


def test_no_candidates_at_all_is_insufficient():
    assert overall_confidence([]) == CONFIDENCE_INSUFFICIENT


def test_overall_confidence_is_always_one_of_the_four_categories():
    cases = [
        [],
        [_direct_candidate(0.95)],
        [_direct_candidate(0.60)],
        [_direct_candidate(0.25)],
        [_direct_candidate(0.01)],
        [_association_candidate(0.5)],
        [_association_candidate(0.05)],
    ]
    # Never a percentage: nothing in this system produces a calibrated
    # probability, so showing one would be fabricated precision.
    assert {overall_confidence(c) for c in cases} <= _CONFIDENCE_CATEGORIES


# --------------------------------------------------------------------------
# Candidate: partitioning and serialization
# --------------------------------------------------------------------------


def test_supporting_and_contradicting_partition_the_evidence_list():
    candidate = _direct_candidate(0.70)
    support = _item(KIND_DIRECT_MATCH)
    conflict = _item(KIND_CONTRADICTION, polarity=POLARITY_CONTRADICTING)
    neutral = _item(KIND_ASSOCIATION, polarity=POLARITY_NEUTRAL, provenance=PROVENANCE_INFERRED)
    for element in (support, conflict, neutral):
        candidate.add(element)

    assert candidate.supporting() == [support]
    assert candidate.contradicting() == [conflict]
    # A neutral item belongs to neither side but is still on the record -- the
    # trail is one list, not a supporting list plus a display-only one.
    assert neutral in candidate.evidence
    assert candidate.has_kind(KIND_CONTRADICTION) is True
    assert candidate.has_kind(KIND_CUE_MATCH) is False


def test_candidate_as_dict_is_json_serializable_and_carries_confidence_and_evidence():
    candidate = _direct_candidate(0.85, semantic={"meaning": "Anna recommended a paper", "event_date": "2024-03-15"})
    candidate.add(_item(KIND_DIRECT_MATCH, detail="This memory matched your query directly."))
    candidate.add(_item(KIND_CONTRADICTION, polarity=POLARITY_CONTRADICTING, detail="Another memory disagrees."))

    payload = candidate.as_dict()

    # The session persists this straight into a JSONB column, so anything
    # non-serializable here is a runtime failure at answer time, not a
    # formatting nicety.
    assert json.loads(json.dumps(payload)) == payload
    assert payload["unit_id"] == "unit-1"
    assert payload["capture_id"] == "cap-1"
    assert payload["route"] == ROUTE_DIRECT
    assert payload["confidence"] == CONFIDENCE_MODERATE
    assert payload["event_date"] == "2024-03-15"
    assert payload["fused_score"] == 0.85
    assert [e["kind"] for e in payload["evidence"]] == [KIND_DIRECT_MATCH, KIND_CONTRADICTION]
    assert all(e["provenance"] in _PROVENANCE_LEVELS for e in payload["evidence"])


def test_an_association_candidate_serializes_its_path_and_no_retrieval_scores():
    candidate = _association_candidate(0.44, why="Both mention Anna (person).")
    candidate.add(_item(KIND_ASSOCIATION, provenance=PROVENANCE_INFERRED))

    payload = candidate.as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["route"] == ROUTE_ASSOCIATION
    assert payload["association_strength"] == 0.44
    assert payload["association_hops"] == 1
    assert payload["association_why"] == "Both mention Anna (person)."
    # It never matched the query, so there is no retrieval score to report --
    # and reporting 0.0 would read as "matched, badly".
    assert payload["fused_score"] is None
    assert payload["vector_score"] is None
    assert payload["confidence"] == CONFIDENCE_WEAK


# --------------------------------------------------------------------------
# Evidence formatting for answer generation
# --------------------------------------------------------------------------


def test_a_directly_matched_candidate_is_labelled_primary_evidence():
    candidate = _direct_candidate(0.85, semantic={"meaning": "Anna recommended a paper", "event_date": "2024-03-15"})
    candidate.add(_item(KIND_DIRECT_MATCH))

    block = format_evidence_block([candidate], [], None)

    assert "[Primary evidence 1]" in block
    assert "Associated memory" not in block
    assert "Anna recommended a paper" in block
    assert "Event date (as recorded): 2024-03-15" in block
    assert "NOTE:" not in block


def test_an_association_candidate_is_labelled_and_carries_the_do_not_answer_note():
    """The guardrail that keeps an inferred lead from being answered as fact.
    The route distinction the UI renders has to exist in the prompt text too,
    or the model is free to present a memory that never matched the question
    as the answer to it."""
    candidate = _association_candidate(0.40, why="Both mention Anna (person).")
    candidate.add(_item(KIND_ASSOCIATION, provenance=PROVENANCE_INFERRED))

    block = format_evidence_block([candidate], [], None)

    assert "[Associated memory 1]" in block
    assert "Primary evidence" not in block
    assert "NOTE: this memory did not match the question directly." in block
    assert "Both mention Anna (person)." in block
    assert "Use it only as supporting context" in block
    assert "do not present it as the answer on its own." in block


def test_primary_evidence_is_ordered_before_associated_memories():
    direct = _direct_candidate(0.85, unit_id="unit-1", capture_id="cap-1")
    both = _direct_candidate(0.80, unit_id="unit-3", capture_id="cap-3")
    both.route = ROUTE_BOTH
    both.association = _FakeAssociation(0.4)
    associated = _association_candidate(0.40, unit_id="unit-2", capture_id="cap-2")

    block = format_evidence_block([associated, direct, both], [], None)

    assert block.index("[Primary evidence 1]") < block.index("[Associated memory 1]")
    # route="both" matched the query itself, so it is primary evidence -- and
    # must therefore NOT carry the don't-answer-with-this note.
    assert "[Primary evidence 2]" in block
    assert block.count("NOTE: this memory did not match the question directly.") == 1


def test_a_contradiction_is_shown_in_the_candidates_own_block():
    candidate = _direct_candidate(0.85)
    candidate.add(
        _item(KIND_CONTRADICTION, polarity=POLARITY_CONTRADICTING, detail="Another memory says it was Boris.")
    )

    block = format_evidence_block([candidate], [], None)

    # The answer must be able to SAY the store disagrees with itself, which it
    # cannot do if the conflicting side never reaches the prompt.
    assert "[Conflict] Another memory says it was Boris." in block


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_every_evidence_item_carries_one_of_the_four_provenance_levels():
    """Provenance is a property of each item, not of the candidate, because a
    single candidate routinely mixes all four: the user's own words, an
    extraction, a computed association, and a search guess can all point at
    the same memory."""
    candidate = _direct_candidate(0.70)
    candidate.add(_item(KIND_CUE_MATCH, provenance=PROVENANCE_ORIGINAL))
    candidate.add(_item(KIND_SHARED_ENTITY, provenance=PROVENANCE_EXTRACTED))
    candidate.add(_item(KIND_ASSOCIATION, provenance=PROVENANCE_INFERRED))
    candidate.add(
        _item(KIND_ASSOCIATION, polarity=POLARITY_NEUTRAL, provenance=PROVENANCE_HYPOTHESIS, weight=0.0)
    )

    levels = {e.provenance for e in candidate.evidence}

    assert levels == _PROVENANCE_LEVELS
    assert {e["provenance"] for e in candidate.as_dict()["evidence"]} == _PROVENANCE_LEVELS
    # Each level reads as something different to a user; two levels sharing a
    # phrasing would make the distinction undisplayable.
    labels = {provenance_label(level) for level in _PROVENANCE_LEVELS}
    assert len(labels) == len(_PROVENANCE_LEVELS)
    assert provenance_label(PROVENANCE_HYPOTHESIS) == "a search guess, not a fact"


def test_an_evidence_items_weight_is_rounded_but_its_text_is_untouched():
    item = EvidenceItem(
        kind=KIND_CUE_MATCH,
        polarity=POLARITY_SUPPORTING,
        provenance=PROVENANCE_ORIGINAL,
        detail="It mentions “ETH”, which you confirmed.",
        weight=0.123456789,
        source_unit_id="unit-1",
        entity_id="entity-1",
    )

    payload = item.as_dict()

    assert payload["weight"] == 0.1235
    assert payload["detail"] == "It mentions “ETH”, which you confirmed."
    assert payload["source_unit_id"] == "unit-1"
    assert payload["entity_id"] == "entity-1"
    assert json.loads(json.dumps(payload)) == payload
