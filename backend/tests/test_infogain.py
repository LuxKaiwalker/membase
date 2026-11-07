"""Tests for app/retrieval/infogain.py -- expected-information-gain question
selection.

Pure unit tests: the module is a deterministic function of (capture grouping,
fused scores, semantic attribute values), so the candidates here are built
directly as ScoredUnits over a minimal fake unit rather than through ingestion.
`_ATTRIBUTE_TEMPLATES`'s phrasings are asserted verbatim on purpose -- callers
reverse-map a generated question back to its field by matching the template
text (see app/retrieval/discriminate.py), so the exact wording is contract.
"""

import pytest

from app.retrieval.infogain import (
    MIN_EXPECTED_GAIN,
    candidate_distribution,
    entropy,
    expected_gain,
    group_by_capture,
)
from app.retrieval.infogain import best_question as choose_question
from app.retrieval.pipeline import ScoredUnit


class _FakeUnit:
    """Only the three attributes infogain actually reads."""

    def __init__(self, capture_id, semantic, unit_id=None):
        self.id = unit_id or capture_id
        self.capture_id = capture_id
        self.semantic = semantic


def _scored(capture_id, semantic, score, unit_id=None):
    return ScoredUnit(
        unit=_FakeUnit(capture_id, semantic, unit_id=unit_id),
        vector_score=None,
        fts_score=score,
        fused_score=score,
    )


def _close(a, b):
    return abs(a - b) < 1e-6


# Four distinct origin captures, each nameable by its own person and sharing a
# topic with exactly one other. The shared topics give an even 2-2 bisection by
# count; the people give a 1-3 split that isolates a single capture. Reused by
# both the skewed and the uniform selection tests below, which differ ONLY in
# the fused scores -- that is the whole point being demonstrated.
def _four_capture_candidates(scores):
    return [
        _scored("cap-a", {"people": ["Anna"], "topics": ["ferry"]}, scores[0]),
        _scored("cap-b", {"people": ["Boris"], "topics": ["ferry"]}, scores[1]),
        _scored("cap-c", {"people": ["Chris"], "topics": ["museum"]}, scores[2]),
        _scored("cap-d", {"people": ["Dana"], "topics": ["museum"]}, scores[3]),
    ]


# --------------------------------------------------------------------------
# candidate_distribution
# --------------------------------------------------------------------------


def test_candidate_distribution_weights_a_capture_by_its_best_unit_not_the_sum():
    """Two sibling units of one note are one candidate episode. Summing their
    scores would make a verbosely-segmented memory the most likely candidate
    purely for having been split into more pieces -- here the sum (0.80) would
    beat the single-unit capture (0.50), while the correct best-of (0.40) loses
    to it."""
    ranked = [
        _scored("cap-a", {"people": ["Anna"]}, 0.40, unit_id="a1"),
        _scored("cap-a", {"people": ["Anna"]}, 0.40, unit_id="a2"),
        _scored("cap-b", {"people": ["Boris"]}, 0.50, unit_id="b1"),
    ]

    distribution = candidate_distribution(ranked)

    assert set(distribution) == {"cap-a", "cap-b"}
    assert distribution["cap-a"] < distribution["cap-b"]
    assert _close(distribution["cap-a"], 0.40 / 0.90)
    assert _close(distribution["cap-b"], 0.50 / 0.90)
    assert _close(sum(distribution.values()), 1.0)
    # The grouping itself keeps every unit -- only the weighting collapses them.
    assert len(group_by_capture(ranked)["cap-a"]) == 2


def test_candidate_distribution_is_uniform_when_every_candidate_scored_zero():
    ranked = [
        _scored("cap-a", {}, 0.0),
        _scored("cap-b", {}, 0.0),
        _scored("cap-c", {}, 0.0),
    ]

    distribution = candidate_distribution(ranked)

    # No basis to prefer any candidate, so uniform is the honest distribution
    # rather than whatever order they happened to arrive in.
    assert all(_close(p, 1 / 3) for p in distribution.values())


def test_candidate_distribution_of_nothing_is_empty():
    assert candidate_distribution([]) == {}


# --------------------------------------------------------------------------
# entropy
# --------------------------------------------------------------------------


def test_entropy_of_a_single_certain_candidate_is_zero():
    assert entropy({"cap-a": 1.0}) == 0.0


def test_entropy_of_two_equally_likely_candidates_is_one_bit():
    assert _close(entropy({"cap-a": 0.5, "cap-b": 0.5}), 1.0)


def test_entropy_of_a_skewed_pair_is_between_zero_and_one_bit():
    skewed = entropy({"cap-a": 0.9, "cap-b": 0.1})
    assert 0.0 < skewed < 1.0
    assert skewed < entropy({"cap-a": 0.5, "cap-b": 0.5})


# --------------------------------------------------------------------------
# expected_gain
# --------------------------------------------------------------------------


def test_a_question_every_candidate_answers_the_same_way_gains_nothing():
    distribution = {"cap-a": 0.5, "cap-b": 0.5}

    # Matching ALL candidates and matching NONE are the same failure: the
    # answer is predetermined, so it carries no information either way.
    assert expected_gain(distribution, {"cap-a", "cap-b"}) == 0.0
    assert expected_gain(distribution, set()) == 0.0
    assert expected_gain(distribution, {"cap-unknown"}) == 0.0


def test_an_even_split_of_two_equal_candidates_gains_one_bit():
    distribution = {"cap-a": 0.5, "cap-b": 0.5}

    assert _close(expected_gain(distribution, {"cap-a"}), 1.0)


def test_expected_gain_of_an_empty_distribution_is_zero():
    assert expected_gain({}, {"cap-a"}) == 0.0


# --------------------------------------------------------------------------
# best_question: the behaviour this module exists for
# --------------------------------------------------------------------------


def test_with_skewed_scores_the_best_question_isolates_the_likely_candidate_not_the_even_split():
    """**This is the whole point of the module.**

    The question that best bisects the candidate set *by count* and the
    question that carries the most information are different questions as soon
    as the candidates are not equally plausible, and the old heuristic
    (app/retrieval/discriminate.py before infogain existed) maximised the
    former. Here capture A holds 75% of the plausibility mass:

      - "Was this about ferry?" splits the four captures 2-2 by count -- a
        perfect bisection by the old criterion -- but leaves A sharing a side
        with a near-worthless candidate, so it resolves little (~0.65 bits).
      - "Did this involve Anna?" splits them 1-3, which the old criterion
        scores as a *worse* question, yet it fully settles the only thing in
        doubt: whether the dominant candidate is the right one (~0.81 bits).

    best_question must return the second one.
    """
    ranked = _four_capture_candidates([0.90, 0.10, 0.10, 0.10])
    distribution = candidate_distribution(ranked)

    isolating_gain = expected_gain(distribution, {"cap-a"})
    bisecting_gain = expected_gain(distribution, {"cap-a", "cap-b"})
    # The count-based bisection really is even, so the old heuristic would
    # have preferred it -- this is not a straw man.
    assert len({"cap-a", "cap-b"}) == len({"cap-c", "cap-d"})
    assert bisecting_gain < isolating_gain

    choice = choose_question(ranked, already_asked=[])

    assert choice is not None
    assert choice.question == "Did this involve Anna?"
    assert choice.field == "people"
    assert choice.yes_group_ids == ("cap-a",)
    assert choice.no_group_ids == ("cap-b", "cap-c", "cap-d")
    assert _close(choice.expected_gain, isolating_gain)
    assert _close(choice.entropy_before, entropy(distribution))


def test_with_uniform_scores_an_even_bisection_achieves_the_maximum_raw_gain():
    """The documented reduction: when every candidate is equally plausible, the
    even bisection is what maximises raw information gain -- so the old
    count-based heuristic is the special case this generalises, rather than a
    behaviour that was replaced.

    Stated about `expected_gain` specifically, not about which question gets
    *chosen*: choice is made on utility (gain x answerability), and the test
    below covers why those can differ.
    """
    ranked = _four_capture_candidates([0.50, 0.50, 0.50, 0.50])
    distribution = candidate_distribution(ranked)

    # "ferry" splits the four captures 2-2; "Anna" isolates one, 1-3.
    bisection_gain = expected_gain(distribution, {"cap-a", "cap-b"})
    lopsided_gain = expected_gain(distribution, {"cap-a"})

    # One yes/no answer over four equally likely candidates can remove at most
    # one of the two bits of uncertainty, and the even split removes all of it
    # that a single question can.
    assert _close(entropy(distribution), 2.0)
    assert _close(bisection_gain, 1.0)
    assert bisection_gain > lopsided_gain


def test_a_concrete_attribute_outranks_a_better_split_on_a_model_invented_topic():
    """Choice is by utility, not raw gain, and that is the point.

    With uniform scores the topic question has the better split (2-2, a full
    bit) while the person question is lopsided (1-3, ~0.81 bits) -- yet the
    person question wins, because `topics` are labels the extraction model
    invented and the user cannot reliably confirm one. Asked "Was this about
    Nature?" (a real observed case) neither answer is meaningful, so the
    entropy calculation credits that question with information it does not
    actually deliver.
    """
    ranked = _four_capture_candidates([0.50, 0.50, 0.50, 0.50])

    choice = choose_question(ranked, already_asked=[])

    assert choice is not None
    assert choice.field == "people"
    assert choice.question == "Did this involve Anna?"
    # It won on answerability despite a strictly worse split -- both halves of
    # that claim asserted, so a regression in either direction is caught.
    assert choice.expected_gain < 1.0
    assert choice.utility == pytest.approx(choice.expected_gain * 1.0)
    topic_gain = expected_gain(candidate_distribution(ranked), {"cap-a", "cap-b"})
    assert topic_gain > choice.expected_gain
    assert topic_gain * 0.3 < choice.utility


def test_a_topic_question_is_still_asked_when_nothing_concrete_discriminates():
    """Answerability demotes topics, it does not forbid them: a corpus whose
    candidates share every person but differ in topic should still get a
    question rather than none."""
    ranked = [
        _scored("cap-a", {"people": ["Anna"], "topics": ["ferry"]}, 0.5),
        _scored("cap-b", {"people": ["Anna"], "topics": ["museum"]}, 0.5),
    ]

    choice = choose_question(ranked, already_asked=[])

    assert choice is not None
    assert choice.field == "topics"


def test_a_value_the_user_already_gave_as_a_cue_is_never_asked_about():
    """Re-asking what you have just been told gains nothing and costs the user
    real effort. Observed live: the system asked about Switzerland, the user
    replied "not Switzerland, I'm fairly sure it was with Markus", and the next
    round asked whether it involved Markus."""
    ranked = _four_capture_candidates([0.90, 0.50, 0.50, 0.50])

    without_cue = choose_question(ranked, already_asked=[])
    assert without_cue is not None
    assert without_cue.value == "Anna"

    with_cue = choose_question(ranked, already_asked=[], known_values=["anna"])

    assert with_cue is None or with_cue.value != "Anna"


def test_no_question_is_worth_asking_when_fewer_than_two_captures_remain():
    one_capture = [_scored("cap-a", {"people": ["Anna"], "topics": ["ferry"]}, 0.90)]
    assert choose_question(one_capture, already_asked=[]) is None
    assert choose_question([], already_asked=[]) is None

    # Two units, one origin note: that is one real episode, so separating them
    # resolves nothing the user cares about.
    siblings = [
        _scored("cap-a", {"people": ["Anna"]}, 0.90, unit_id="a1"),
        _scored("cap-a", {"people": ["Boris"]}, 0.80, unit_id="a2"),
    ]
    assert choose_question(siblings, already_asked=[]) is None


def test_no_question_is_asked_when_every_gain_falls_below_the_threshold():
    """A near-certain top candidate leaves almost no uncertainty to remove, so
    every available question is a round trip that would barely move the
    ranking -- and asking costs the user real effort."""
    ranked = [
        _scored("cap-a", {"people": ["Anna"]}, 1.0),
        _scored("cap-b", {"people": ["Boris"]}, 0.001),
    ]
    assert entropy(candidate_distribution(ranked)) < MIN_EXPECTED_GAIN

    assert choose_question(ranked, already_asked=[]) is None
    # The None came from the gain threshold, not from there being no
    # distinguishing attribute at all: drop the floor and a question appears.
    permissive = choose_question(ranked, already_asked=[], min_gain=0.0)
    assert permissive is not None
    assert permissive.question == "Did this involve Anna?"


def test_best_question_never_repeats_an_already_asked_question():
    ranked = _four_capture_candidates([0.50, 0.50, 0.50, 0.50])

    first = choose_question(ranked, already_asked=[])
    assert first is not None

    second = choose_question(ranked, already_asked=[first.question])
    assert second is not None
    assert second.question != first.question

    # Exhausting every question this candidate set can offer must yield None,
    # never a repeat and never something generic.
    asked = []
    for _ in range(20):
        choice = choose_question(ranked, already_asked=asked)
        if choice is None:
            break
        assert choice.question not in asked
        asked.append(choice.question)
    assert choose_question(ranked, already_asked=asked) is None


def test_best_question_is_stable_across_identical_calls():
    """Exact gain ties are common (a question and its complement always tie).
    An unstable tie-break would make a session's question sequence
    irreproducible and the eval harness flaky."""
    ranked = _four_capture_candidates([0.50, 0.50, 0.50, 0.50])

    first = choose_question(ranked, already_asked=[])
    second = choose_question(ranked, already_asked=[])

    assert first == second


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def test_question_choice_labels_its_gain_as_an_approximation():
    """The distribution comes from normalised fused retrieval scores, which are
    a rank signal and not calibrated probabilities -- so the bit figure orders
    questions but is not a measurement. Every key that surfaces it says so."""
    ranked = _four_capture_candidates([0.90, 0.10, 0.10, 0.10])
    choice = choose_question(ranked, already_asked=[])

    payload = choice.as_dict()

    gain_keys = [k for k in payload if "gain" in k or "entropy" in k]
    assert gain_keys
    assert all("approx" in k for k in gain_keys)
    assert payload["question"] == choice.question
    assert payload["field"] == "people"
    assert payload["value"] == "Anna"
    assert payload["separates"] == {"yes": ["cap-a"], "no": ["cap-b", "cap-c", "cap-d"]}
