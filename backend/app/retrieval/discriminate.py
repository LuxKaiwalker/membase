import uuid

from app.retrieval.pipeline import ScoredUnit

# Heuristic, not a formal Bayesian active-learning framework (spec section
# 13: "a simple heuristic... is sufficient"). Two distinct origin-group
# candidates whose top scores are this close are treated as genuinely
# ambiguous rather than one being "the" answer just because it ranked
# first (spec section 12/32: a high score is not proof of uniqueness).
AMBIGUITY_SCORE_MARGIN = 0.15

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


def values_for_field(semantic: dict, field: str) -> list[str]:
    """Extract plain strings worth matching/asking about for one semantic
    field, handling the two fields that aren't flat string lists:
    `relations` (list of {subject, predicate, object}) and `definitions`
    (list of {term, definition}). An entity's own name is as likely to
    turn up as a relation's subject/object or a definition's term as in
    any of the flat list fields, so callers must not skip these.
    """
    if field == "event_date":
        v = semantic.get("event_date")
        return [v.strip()] if isinstance(v, str) and v.strip() else []
    if field == "relations":
        out: list[str] = []
        for r in semantic.get("relations") or []:
            if not isinstance(r, dict):
                continue
            for key in ("subject", "object"):
                v = r.get(key)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
        return out
    if field == "definitions":
        out = []
        for d in semantic.get("definitions") or []:
            if isinstance(d, dict):
                term = d.get("term")
                if isinstance(term, str) and term.strip():
                    out.append(term.strip())
        return out
    raw = semantic.get(field)
    if not isinstance(raw, list):
        return []
    return [v.strip() for v in raw if isinstance(v, str) and v.strip()]


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
    return (top_scores[0] - top_scores[1]) < AMBIGUITY_SCORE_MARGIN


def apply_negative_cues(ranked: list[ScoredUnit], negative_cue_texts: list[str]) -> list[ScoredUnit]:
    """Demote a candidate the user has explicitly ruled out (spec section
    5/37/38: "No, it wasn't Anna") by excluding any candidate whose
    structured metadata (not just a loose text match) names that value.
    This only changes which candidates this session currently considers
    plausible -- it never touches the underlying MemoryUnit.
    """
    if not negative_cue_texts:
        return ranked
    negatives = {c.strip().lower() for c in negative_cue_texts if c.strip()}
    if not negatives:
        return ranked

    kept = []
    for s in ranked:
        semantic = s.unit.semantic or {}
        values = [v for field in _CUE_MATCH_FIELDS for v in values_for_field(semantic, field)]
        if any(v.lower() in negatives for v in values):
            continue
        kept.append(s)
    return kept


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
        semantic = s.unit.semantic or {}
        values = [v for field in _CUE_MATCH_FIELDS for v in values_for_field(semantic, field)]
        matches = any(v.lower() in cues for v in values)
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
    """Pick the semantic attribute that best bisects the current
    candidate set and phrase one concrete question about it (spec section
    13/14). Returns None if no attribute distinguishes the remaining
    candidates without repeating a question already asked -- the caller
    should treat that as this angle being exhausted, not a reason to ask
    something generic (spec section 40: a poor question is worse than
    none).
    """
    groups = _group_by_capture(ranked)
    candidates = [max(g, key=lambda s: s.fused_score) for g in groups.values()]
    if len(candidates) < 2:
        return None

    asked = set(already_asked)
    best_question: str | None = None
    best_split_quality = -1.0

    for field, template in _ATTRIBUTE_TEMPLATES.items():
        value_to_candidates: dict[str, list[ScoredUnit]] = {}
        for s in candidates:
            semantic = s.unit.semantic or {}
            for v in values_for_field(semantic, field):
                value_to_candidates.setdefault(v, []).append(s)

        for value, matched in value_to_candidates.items():
            if len(matched) == len(candidates):
                continue  # shared by every remaining candidate -- doesn't discriminate
            question = template.format(value=value)
            if question in asked:
                continue
            # How close this value's presence/absence comes to bisecting
            # the candidate set in half (a perfect 50/50 split scores 1.0).
            split_quality = 1.0 - abs(len(matched) - (len(candidates) - len(matched))) / len(candidates)
            if split_quality > best_split_quality:
                best_split_quality = split_quality
                best_question = question

    return best_question
