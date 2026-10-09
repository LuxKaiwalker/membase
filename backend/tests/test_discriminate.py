from app.retrieval.discriminate import apply_negative_cues, apply_positive_cue_boost, is_ambiguous
from app.retrieval.pipeline import ScoredUnit


class _FakeUnit:
    def __init__(self, capture_id, semantic, retrieval_text=""):
        self.id = capture_id
        self.capture_id = capture_id
        self.semantic = semantic
        self.retrieval_text = retrieval_text


# Streamlining phase (see docs/DECISIONS.md "Simplify cue-to-candidate
# matching"): a cue the model never extracted into any structured field
# must still match via a normalized substring check against the unit's
# own retrieval_text, rather than requiring yet another specialized
# field to be added every time this happens.


def test_cue_not_in_any_structured_field_still_matches_via_retrieval_text():
    unit = _FakeUnit(
        "cap-a",
        semantic={"people": [], "locations": [], "organizations": [], "topics": [], "relations": [], "definitions": []},
        retrieval_text="Original:\nWe discussed the Phoenix migration over lunch.",
    )
    a = ScoredUnit(unit=unit, vector_score=None, fts_score=0.5, fused_score=0.5)

    filtered = apply_negative_cues([a], ["Phoenix migration"])
    assert filtered == []

    boosted = apply_positive_cue_boost([a], ["Phoenix migration"])
    assert boosted[0].fused_score > a.fused_score


def test_structured_field_match_still_takes_the_exact_path():
    unit = _FakeUnit("cap-a", semantic={"people": ["Anna"]}, retrieval_text="")
    a = ScoredUnit(unit=unit, vector_score=None, fts_score=0.5, fused_score=0.5)

    assert apply_negative_cues([a], ["Anna"]) == []


def test_short_cue_does_not_match_via_substring_fallback():
    """A 1-2 char cue would match almost anything by substring -- the
    fallback path is deliberately skipped for these; only an exact
    structured-field match can match a cue that short."""
    unit = _FakeUnit("cap-a", semantic={}, retrieval_text="This memory is about an ox on a farm.")
    a = ScoredUnit(unit=unit, vector_score=None, fts_score=0.5, fused_score=0.5)

    assert apply_negative_cues([a], ["ox"]) == [a]


def test_two_weak_barely_relevant_matches_are_not_ambiguous():
    """A close margin between two candidates only means genuine
    ambiguity if the runner-up itself clears a real relevance bar --
    two weak matches near the noise floor that happen to be close to
    each other mean "nothing solid was found", not "two strong
    contenders" (see docs/DECISIONS.md "Improve ambiguity detection")."""
    a = ScoredUnit(unit=_FakeUnit("cap-a", semantic={}), vector_score=None, fts_score=0.2, fused_score=0.20)
    b = ScoredUnit(unit=_FakeUnit("cap-b", semantic={}), vector_score=None, fts_score=0.19, fused_score=0.19)
    assert is_ambiguous([a, b]) is False


def test_two_strong_close_matches_are_ambiguous():
    a = ScoredUnit(unit=_FakeUnit("cap-a", semantic={}), vector_score=None, fts_score=0.6, fused_score=0.60)
    b = ScoredUnit(unit=_FakeUnit("cap-b", semantic={}), vector_score=None, fts_score=0.55, fused_score=0.55)
    assert is_ambiguous([a, b]) is True


def test_no_match_leaves_candidate_untouched():
    unit = _FakeUnit("cap-a", semantic={}, retrieval_text="Something unrelated entirely.")
    a = ScoredUnit(unit=unit, vector_score=None, fts_score=0.5, fused_score=0.5)

    assert apply_negative_cues([a], ["nothing matches this"]) == [a]
    assert apply_positive_cue_boost([a], ["nothing matches this"])[0].fused_score == a.fused_score
