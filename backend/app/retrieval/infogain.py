"""Choosing the clarifying question that is actually worth asking.

The previous rule (app/retrieval/discriminate.py, before this module existed)
picked the attribute value that best *bisected* the candidate set: a 50/50
split scored 1.0, everything else less. That is a reasonable proxy, but it is
only correct when every candidate is equally plausible. When one candidate is
far more likely than the rest, splitting the set in half by count can be close
to worthless -- the likely candidate ends up on one side and the answer tells
you almost nothing you did not already believe.

What this module computes instead is the **expected information gain** of a
question: the mutual information between the user's answer and which memory
they actually meant, under a plausibility distribution derived from retrieval
scores.

    H(p)                      entropy of the current candidate distribution
    P(yes), P(no)             probability the answer is yes / no
    H(p | yes), H(p | no)     entropy of the posterior in each case
    gain = H(p) - [ P(yes)*H(p|yes) + P(no)*H(p|no) ]

Maximising `gain` is a strict generalisation of the old heuristic: with a
uniform distribution the maximum-gain split *is* the even bisection, so the
previous behaviour is the special case this reduces to. What changes is the
skewed case, which is the common one.

Two honest caveats, kept here rather than buried:

- The distribution is derived from normalised fused retrieval scores. Those
  are a rank signal, not calibrated probabilities (see
  docs/DECISIONS.md "Improve ambiguity detection"), so `gain` is a principled
  *ordering* of questions, not a trustworthy absolute number of bits. It is
  reported in the session trail as an approximation and labelled as one.
- The model assumes the user answers truthfully and that a candidate's stored
  metadata correctly reflects reality. A user who misremembers ("yes, Anna was
  there" when she was not) will be led away from the right memory -- which is
  exactly why a negative cue excludes rather than eliminating permanently, and
  why the original query is re-searched every round (see
  app/retrieval/session.py).

Grouping is by `capture_id`, not by unit: sibling units of one submission are
the same real episode, and a question that only separates two halves of one
note has not resolved anything the user cares about.
"""

import math
import uuid
from dataclasses import dataclass

from app.retrieval.pipeline import ScoredUnit

# Questions whose expected gain is below this are not worth a round trip --
# asking costs the user real effort, and the proposal is explicit that a poor
# question is worse than none. 0.05 bits is "this answer would barely move the
# ranking"; a genuinely discriminating yes/no on two close candidates yields
# close to 1.0 bit.
MIN_EXPECTED_GAIN = 0.05

# How reliably a user can actually answer a question about each attribute.
#
# This is not a fudge factor, it corrects a real flaw in using raw information
# gain. The gain calculation assumes the answer is truthful and confident --
# that is what makes "the user said yes" worth a full bit. But the attributes
# are not equally answerable. A person's name, a place, or an organization is
# something the user either remembers or does not, and can say so. A *topic*
# is a label the extraction model invented: the user never wrote "Nature", so
# asked "Was this about Nature?" they cannot meaningfully confirm or deny it,
# and whichever way they answer is closer to noise than evidence.
#
# Observed for real on the demonstration corpus: maximising raw gain picked
# "Was this about Nature?" (0.98 of 2.32 bits) over asking about Markus or
# Lake Bled, because that label happened to bisect the candidate distribution
# best. It is a near-optimal question by the metric and a nearly useless one to
# a person trying to recall a photograph.
#
# Discounting by answerability is therefore a *better* estimate of expected
# gain, not a departure from it: a question whose answer is unreliable does not
# actually deliver the information the entropy calculation credits it with.
FIELD_ANSWERABILITY = {
    # Concrete, and the user either knows or doesn't.
    "people": 1.0,
    "locations": 1.0,
    "organizations": 0.95,
    "event_date": 0.85,
    # An entity named as a relation endpoint or a defined term is usually
    # concrete too, but the surface string is often a phrase rather than a
    # name ("AHV number 756..."), which is harder to answer cleanly.
    "relations": 0.6,
    "definitions": 0.6,
    # Model-generated thematic labels. Kept rather than excluded: when a
    # memory has no named people or places at all, a topic question is better
    # than no question. It just must never outrank a concrete one.
    "topics": 0.3,
}
DEFAULT_ANSWERABILITY = 0.5


@dataclass(frozen=True)
class QuestionChoice:
    question: str
    field: str
    value: str
    # Raw mutual information between a truthful answer and candidate identity.
    expected_gain: float
    # `expected_gain` discounted by how answerable this attribute actually is
    # (see FIELD_ANSWERABILITY). This is what questions are ranked by.
    utility: float
    entropy_before: float
    # Candidate origin-capture ids on each side of the question, so the
    # session trail can show what the question was actually separating.
    yes_group_ids: tuple[str, ...]
    no_group_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "field": self.field,
            "value": self.value,
            # Labelled "approximate" wherever it surfaces -- see the module
            # docstring on why this is an ordering signal, not a measurement.
            "expected_gain_bits_approx": round(self.expected_gain, 4),
            # What the question was actually chosen by: gain discounted by how
            # answerable the attribute is. Reported alongside the raw gain so a
            # question that won on answerability rather than on raw split is
            # visible as such.
            "utility_approx": round(self.utility, 4),
            "answerability": round(
                FIELD_ANSWERABILITY.get(self.field, DEFAULT_ANSWERABILITY), 4
            ),
            "entropy_before_bits_approx": round(self.entropy_before, 4),
            "separates": {"yes": list(self.yes_group_ids), "no": list(self.no_group_ids)},
        }


def group_by_capture(ranked: list[ScoredUnit]) -> dict[uuid.UUID, list[ScoredUnit]]:
    groups: dict[uuid.UUID, list[ScoredUnit]] = {}
    for scored in ranked:
        groups.setdefault(scored.unit.capture_id, []).append(scored)
    return groups


def candidate_distribution(ranked: list[ScoredUnit]) -> dict[uuid.UUID, float]:
    """Normalised plausibility per origin capture.

    A group's weight is its best unit's fused score (not the sum): three
    sibling units from one note are one candidate episode, and summing would
    make a verbosely-segmented memory look more likely purely for having been
    split into more pieces.
    """
    groups = group_by_capture(ranked)
    if not groups:
        return {}
    best = {cid: max(s.fused_score for s in units) for cid, units in groups.items()}
    total = sum(max(0.0, v) for v in best.values())
    if total <= 0:
        # Every candidate scored zero: no basis to prefer any, so uniform is
        # the honest distribution rather than an arbitrary ordering.
        uniform = 1.0 / len(best)
        return {cid: uniform for cid in best}
    return {cid: max(0.0, v) / total for cid, v in best.items()}


def entropy(distribution: dict[uuid.UUID, float]) -> float:
    """Shannon entropy in bits. Zero for a single certain candidate."""
    total = 0.0
    for p in distribution.values():
        if p > 0:
            total -= p * math.log2(p)
    return total


def _conditional_entropy(distribution: dict[uuid.UUID, float], subset: set[uuid.UUID]) -> tuple[float, float]:
    """`(mass, entropy)` of the posterior restricted to `subset`."""
    mass = sum(distribution[cid] for cid in subset if cid in distribution)
    if mass <= 0:
        return 0.0, 0.0
    posterior = {cid: distribution[cid] / mass for cid in subset if cid in distribution}
    return mass, entropy(posterior)


def expected_gain(
    distribution: dict[uuid.UUID, float], yes_groups: set[uuid.UUID]
) -> float:
    """Mutual information between a yes/no answer and candidate identity."""
    if not distribution:
        return 0.0
    all_groups = set(distribution)
    no_groups = all_groups - yes_groups
    if not yes_groups or not no_groups:
        # The question does not separate anything: every candidate would give
        # the same answer, so the answer carries no information.
        return 0.0
    before = entropy(distribution)
    p_yes, h_yes = _conditional_entropy(distribution, yes_groups)
    p_no, h_no = _conditional_entropy(distribution, no_groups)
    return before - (p_yes * h_yes + p_no * h_no)


def best_question(
    ranked: list[ScoredUnit],
    *,
    already_asked: list[str] | None = None,
    known_values: list[str] | None = None,
    min_gain: float = MIN_EXPECTED_GAIN,
) -> QuestionChoice | None:
    """The highest-utility discriminating question, or None.

    None means "no question is worth asking" -- either fewer than two distinct
    candidate episodes remain, or every remaining attribute would tell us
    almost nothing. The caller must treat that as this line of inquiry being
    exhausted, not as a reason to ask something generic.

    `known_values` are attribute values the user has already spoken about this
    session (their cues, either polarity); questions about those are skipped.
    `already_asked` only suppresses an identical question *string*, which is
    not enough: observed live, the system asked about Switzerland, the user
    replied "not Switzerland, I'm fairly sure it was with Markus", and the next
    round asked "Did this involve Markus?" -- re-asking the very thing it had
    just been told. That is the clearest case of a question that gains nothing
    while costing the user real effort.
    """
    # Imported here rather than at module level: discriminate.py imports this
    # module for its own selection, so a top-level import would be circular.
    from app.retrieval.discriminate import _ATTRIBUTE_TEMPLATES, values_for_field

    asked = set(already_asked or [])
    # Compared case-insensitively on trimmed text: a cue is free-text the user
    # typed, an attribute value came out of extraction, and they will not match
    # on exact bytes.
    known = {v.strip().casefold() for v in (known_values or []) if v and v.strip()}
    distribution = candidate_distribution(ranked)
    if len(distribution) < 2:
        return None
    before = entropy(distribution)

    groups = group_by_capture(ranked)
    best: QuestionChoice | None = None

    # Iteration order is the template dict's insertion order, and values are
    # sorted, so an exact gain tie resolves the same way on every run. An
    # unstable choice here would make a session's question sequence
    # irreproducible and the eval harness flaky.
    for field, template in _ATTRIBUTE_TEMPLATES.items():
        value_to_groups: dict[str, set[uuid.UUID]] = {}
        for capture_id, units in groups.items():
            for scored in units:
                for value in values_for_field(scored.unit.semantic or {}, field):
                    value_to_groups.setdefault(value, set()).add(capture_id)

        answerability = FIELD_ANSWERABILITY.get(field, DEFAULT_ANSWERABILITY)
        for value in sorted(value_to_groups):
            yes_groups = value_to_groups[value]
            question = template.format(value=value)
            if question in asked or value.strip().casefold() in known:
                continue
            gain = expected_gain(distribution, yes_groups)
            # The floor applies to the RAW gain: a question that genuinely
            # cannot separate the candidates is useless regardless of how
            # answerable its attribute is.
            if gain < min_gain:
                continue
            utility = gain * answerability
            if best is None or utility > best.utility:
                best = QuestionChoice(
                    question=question,
                    field=field,
                    value=value,
                    expected_gain=gain,
                    utility=utility,
                    entropy_before=before,
                    yes_group_ids=tuple(sorted(str(c) for c in yes_groups)),
                    no_group_ids=tuple(sorted(str(c) for c in set(distribution) - yes_groups)),
                )
    return best
