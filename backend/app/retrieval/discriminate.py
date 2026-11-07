import re
import uuid

from app.retrieval.pipeline import ScoredUnit

# Heuristic, not a formal Bayesian active-learning framework (spec section
# 13: "a simple heuristic... is sufficient"). Two distinct origin-group
# candidates whose top scores are this close are treated as genuinely
# ambiguous rather than one being "the" answer just because it ranked
# first (spec section 12/32: a high score is not proof of uniqueness).
AMBIGUITY_SCORE_MARGIN = 0.15

# The fused score is a rough rank signal, not a calibrated probability
# (see docs/DECISIONS.md "Improve ambiguity detection") -- a small margin
# between two candidates only means genuine ambiguity if the runner-up
# itself clears a real relevance bar. Two weak, barely-above-the-noise-
# floor matches that happen to be close to each other are "nothing solid
# was found", not "two strong contenders to disambiguate between";
# without this, a query with no good answer at all could still trigger
# a pointless clarifying question just because its two weakest matches
# scored similarly low.
AMBIGUITY_MIN_RUNNER_UP_SCORE = 0.45

# One question per distinguishing attribute; phrased as a concrete yes/no
# question about a specific value rather than an open-ended "tell me more"
# (spec section 13/40). "relations"/"definitions" were added after the
# interactive-recall benchmark traced a real bug here: an entity's own
# name frequently lands in one of these two fields instead of
# people/organizations/locations/aliases/topics (e.g. a project name
# extracted only as the object of a "worked on" relation, or as a
# definition's term) -- every function in this module that inspects
# semantic metadata must look at the same field set, or a cue naming an
# entity by name silently fails to match depending on which field the
# enrichment call happened to file it under. See docs/DECISIONS.md.
_ATTRIBUTE_TEMPLATES = {
    "people": "Did this involve {value}?",
    "locations": "Did this happen at {value}?",
    "organizations": "Was this related to {value}?",
    "topics": "Was this about {value}?",
    "event_date": "Was this around {value}?",
    # Distinct wording from every field above is deliberate, not just
    # style: callers that reverse-map a generated question back to the
    # field it came from (e.g. eval/run_interactive_benchmark.py's
    # simulated-user oracle) match on the template text itself, so two
    # fields sharing a template is a silent correctness bug, not a
    # cosmetic duplicate.
    "relations": "Was {value} involved in this?",
    "definitions": "Was this specifically about {value}?",
}

# Fields apply_negative_cues/apply_positive_cue_boost match a cue's text
# against. Kept as a plain list (not reusing _ATTRIBUTE_TEMPLATES's keys
# directly) since cue-matching and question-generation are conceptually
# separate even though they currently cover the same fields.
_CUE_MATCH_FIELDS = ["people", "organizations", "locations", "aliases", "topics", "relations", "definitions"]


def _clean_value(value: str) -> str:
    """Strip a parenthetical gloss from an extracted attribute value.

    The extraction model annotates its own output: `semantic.people` really
    does contain entries like `"Markus (Data provider/Colleague/Friend)"`.
    Those strings are used for two things -- phrasing a question to the user,
    and matching the user's cues against candidates -- and the gloss hurts
    both. It produced the live question "Did this involve Markus (Data
    provider/Colleague/Friend)?", and it meant a cue of "Markus" failed to
    match a candidate whose stored value carried the annotation.

    Only the display/match form is cleaned; `semantic` itself is untouched, so
    nothing authoritative or regenerable changes. Mirrors
    `app/memory/entities.py::display_name_for`, which does the same for entity
    nodes.
    """
    cleaned = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", value)
    cleaned = " ".join(cleaned.split()).strip(" ,;:-")
    return cleaned or " ".join(value.split())


def values_for_field(semantic: dict, field: str) -> list[str]:
    """Extract plain strings worth matching/asking about for one semantic
    field, handling the two fields that aren't flat string lists:
    `relations` (list of {subject, predicate, object}) and `definitions`
    (list of {term, definition}). An entity's own name is as likely to
    turn up as a relation's subject/object or a definition's term as in
    any of the flat list fields, so callers must not skip these.

    Values come back with any parenthetical gloss removed (see
    `_clean_value`), deduplicated, and in a stable order.
    """
    if field == "event_date":
        v = semantic.get("event_date")
        return [_clean_value(v)] if isinstance(v, str) and v.strip() else []

    raw_values: list[str] = []
    if field == "relations":
        for r in semantic.get("relations") or []:
            if not isinstance(r, dict):
                continue
            for key in ("subject", "object"):
                v = r.get(key)
                if isinstance(v, str) and v.strip():
                    raw_values.append(v)
    elif field == "definitions":
        for d in semantic.get("definitions") or []:
            if isinstance(d, dict):
                term = d.get("term")
                if isinstance(term, str) and term.strip():
                    raw_values.append(term)
    else:
        raw = semantic.get(field)
        if not isinstance(raw, list):
            return []
        raw_values = [v for v in raw if isinstance(v, str) and v.strip()]

    # Cleaning can collapse two spellings onto one value ("Markus" and
    # "Markus (Colleague)"), so dedupe afterwards -- otherwise a question's
    # candidate-splitting counts would double-count one attribute.
    seen: dict[str, None] = {}
    for value in raw_values:
        cleaned = _clean_value(value)
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def _group_by_capture(ranked: list[ScoredUnit]) -> dict[uuid.UUID, list[ScoredUnit]]:
    groups: dict[uuid.UUID, list[ScoredUnit]] = {}
    for s in ranked:
        groups.setdefault(s.unit.capture_id, []).append(s)
    return groups


def is_ambiguous(ranked: list[ScoredUnit]) -> bool:
    """Candidates from two or more distinct origin captures (i.e. likely
    different real-world events, not sibling units of the same episode)
    with comparably high scores mean the top result isn't yet reliably
    "the" answer -- a discriminating question is warranted before
    answering (spec section 12/32).
    """
    groups = _group_by_capture(ranked)
    if len(groups) < 2:
        return False
    top_scores = sorted((max(s.fused_score for s in g) for g in groups.values()), reverse=True)
    if top_scores[1] < AMBIGUITY_MIN_RUNNER_UP_SCORE:
        return False
    return (top_scores[0] - top_scores[1]) < AMBIGUITY_SCORE_MARGIN


def _matches_cue(unit, cue_lower: str) -> bool:
    """A cue matches a candidate if it names a value the unit's own
    structured semantic fields already extracted (exact match -- a
    strong, precise signal), OR -- the unified fallback for a cue naming
    something the extraction step didn't happen to pull out into any
    field, rather than chasing it with yet another field-specific
    special case -- if it appears as a normalized substring of the
    unit's own retrieval_text at all (see docs/DECISIONS.md "Simplify
    cue-to-candidate matching"). The substring path is skipped for very
    short cues (<3 chars) where it would mostly produce noise rather
    than signal.
    """
    semantic = unit.semantic or {}
    values = {v.lower() for field in _CUE_MATCH_FIELDS for v in values_for_field(semantic, field)}
    if cue_lower in values:
        return True
    if len(cue_lower) < 3:
        return False
    # Word-boundary match, not a raw substring check -- a plain `in`
    # here would let a short cue like "Ann" match inside unrelated
    # words ("annual", "banner"), wrongly including or excluding
    # candidates that have nothing to do with the cued entity.
    text = (getattr(unit, "retrieval_text", "") or "").lower()
    return re.search(rf"\b{re.escape(cue_lower)}\b", text) is not None


def apply_negative_cues(ranked: list[ScoredUnit], negative_cue_texts: list[str]) -> list[ScoredUnit]:
    """Demote a candidate the user has explicitly ruled out (spec section
    5/37/38: "No, it wasn't Anna") by excluding any candidate whose
    semantic representation (structured fields, or its retrieval_text as
    a fallback -- see _matches_cue) names that value. This only changes
    which candidates this session currently considers plausible -- it
    never touches the underlying MemoryUnit.
    """
    if not negative_cue_texts:
        return ranked
    negatives = {c.strip().lower() for c in negative_cue_texts if c.strip()}
    if not negatives:
        return ranked

    return [s for s in ranked if not any(_matches_cue(s.unit, n) for n in negatives)]


# Flat bonus applied when a positive cue's text exactly names an
# attribute a candidate's own semantic metadata already has (see
# apply_positive_cue_boost). Deliberately large relative to
# AMBIGUITY_SCORE_MARGIN: a user explicitly confirming "it was at ETH"
# in answer to exactly that discriminating question should decisively
# resolve ambiguity, not just nudge a generic re-ranking score that's
# dominated by unrelated lexical/vector overlap between the original
# query and both candidates (observed empirically: re-searching with the
# cue as a bare one-or-two-word query scores *lower* on the fused metric
# than the original full-sentence query already did for both candidates,
# so merging via max() alone never resolves anything -- see docs/TODO.md).
POSITIVE_CUE_BOOST = 0.3


def apply_positive_cue_boost(ranked: list[ScoredUnit], positive_cue_texts: list[str]) -> list[ScoredUnit]:
    """Reward a candidate whose own structured metadata matches a cue the
    user just confirmed (spec section 14: the question asked about a
    specific attribute value; a "yes" should count as real evidence for
    the candidate that has it, not just another noisy search term).
    Returns new ScoredUnit instances; never mutates the input or the
    underlying MemoryUnit.
    """
    if not positive_cue_texts:
        return ranked
    cues = {c.strip().lower() for c in positive_cue_texts if c.strip()}
    if not cues:
        return ranked

    boosted: list[ScoredUnit] = []
    for s in ranked:
        matches = any(_matches_cue(s.unit, c) for c in cues)
        new_score = min(1.0, s.fused_score + POSITIVE_CUE_BOOST) if matches else s.fused_score
        boosted.append(ScoredUnit(unit=s.unit, vector_score=s.vector_score, fts_score=s.fts_score, fused_score=new_score))
    return boosted


def narrow_to_dominant_group(ranked: list[ScoredUnit]) -> list[ScoredUnit]:
    """Once `is_ambiguous` has resolved to False, keep only the winning
    origin-group's units as evidence. A second, clearly-separated group
    that merely cleared the basic relevance bar (e.g. shares one keyword
    with the query) shouldn't keep cluttering the answer-generation
    evidence block once there's a clear winner -- unlike the ambiguous
    case (spec section 12), there is no longer a real decision for the
    answering step to make between groups.
    """
    groups = _group_by_capture(ranked)
    if len(groups) <= 1:
        return ranked
    best_capture_id = max(groups, key=lambda cid: max(s.fused_score for s in groups[cid]))
    return groups[best_capture_id]


def generate_discriminating_question(ranked: list[ScoredUnit], *, already_asked: list[str]) -> str | None:
    """Pick the semantic attribute that best discriminates between the current
    candidates and phrase one concrete question about it (spec section 13/14).
    Returns None if no attribute distinguishes the remaining candidates
    usefully without repeating a question already asked -- the caller should
    treat that as this angle being exhausted, not a reason to ask something
    generic (spec section 40: a poor question is worse than none).

    **Selection criterion, changed:** this used to score each candidate
    question by how evenly it bisected the candidate set *by count*. It now
    maximises expected information gain over a plausibility-weighted
    distribution (`app/retrieval/infogain.py`), which is a strict
    generalisation -- for equally-scored candidates the maximum-gain split is
    the even bisection, so this reduces to the old behaviour in that case --
    and a real improvement when one candidate is much more likely than the
    others, where an even split by count can be nearly uninformative.

    The question *phrasings* (`_ATTRIBUTE_TEMPLATES`) are deliberately
    unchanged: eval/run_interactive_benchmark.py's simulated-user oracle
    reverse-maps a generated question back to its field by matching the
    template text, so changing the wording here would silently break the
    benchmark rather than fail loudly.
    """
    choice = select_question(ranked, already_asked=already_asked)
    return choice.question if choice is not None else None


def select_question(
    ranked: list[ScoredUnit], *, already_asked: list[str], known_values: list[str] | None = None
):
    """Same selection as `generate_discriminating_question`, returning the full
    `QuestionChoice` (field, value, expected gain, utility, which candidates it
    separates) instead of just the text -- for the recall session's trail,
    which has to be able to show *why* this question was asked.

    `known_values` are the cue texts the user has already given this session,
    so the selector never asks about something they just told it.
    """
    from app.retrieval.infogain import best_question

    return best_question(ranked, already_asked=already_asked, known_values=known_values)
