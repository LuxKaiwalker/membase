"""Deriving episodes: grouping memories into the experiences they belong to.

An episode is what people actually recall by -- "that trip where we hiked to
the lake" -- as opposed to a `Capture` (one submission) or a `MemoryUnit` (one
retrievable fact). This module builds them automatically from information
already present, using three rules in descending order of confidence:

1. **Same capture.** Units the user submitted together are one experience.
   `origin="extracted"`, `confidence="certain"` -- nothing is guessed; they
   really were written in one go.
2. **Temporally compatible and entity-overlapping.** A dated memory joins an
   existing episode when its date fits that episode's span *given how
   precisely each is known* (see app/memory/temporal.py::dates_compatible) AND
   it shares at least one non-generic entity with the episode's members.
   `origin="inferred"`, `confidence="likely"`.
3. **Strong entity overlap with no usable date.** An undated memory joins only
   on a stronger entity requirement, and never widens the episode's dates.
   `origin="inferred"`, `confidence="uncertain"`.

If no rule fires, the memory seeds a new single-member episode.

**Why requiring BOTH time and entity overlap matters.** Time alone groups
everything that happened in one month into "an episode", which is a calendar,
not a memory. Entities alone group every memory mentioning a recurring person
into one undated blob. The conjunction is what makes a cluster correspond to
something a person would recognise as one experience.

**No LLM call, anywhere in this module**, including for titles. A title is
composed from the episode's dominant entities and its date span ("Lake Bled,
Markus - July 2023"). A model-written title would read better, but it would
cost one generation call per episode and have to be re-run every time
membership changed -- an unbounded cost for a cosmetic gain, against a
codebase rule that every LLM-driven loop be explicitly bounded for a stated
reason. `Episode.title_source` records that the title is derived, so a
model-written one could be added later and told apart.

**Never invents a date or a boundary.** An episode whose members carry no
resolvable date has NULL dates and `boundary_confidence="uncertain"`. It is
not filed under the day the memories were typed.
"""

import datetime as dt
import logging
import uuid
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.memory.temporal import (
    PRECISION_UNKNOWN,
    dates_compatible,
    describe,
    looser_precision,
    parse_event_date,
)
from app.models import (
    ORIGIN_EXPLICIT,
    ORIGIN_EXTRACTED,
    ORIGIN_INFERRED,
    Episode,
    EpisodeMembership,
    MemoryEntity,
    MemoryUnit,
    UnitEntityLink,
)

logger = logging.getLogger(__name__)

RULE_SAME_CAPTURE = "same_capture"
RULE_TEMPORAL_ENTITY = "temporal_and_entity_overlap"
RULE_ENTITY_ONLY = "entity_overlap_without_a_date"
RULE_SEED = "seeded_a_new_episode"

CONFIDENCE_CERTAIN = "certain"
CONFIDENCE_LIKELY = "likely"
CONFIDENCE_UNCERTAIN = "uncertain"

BOUNDARY_ASSERTED = "asserted"
BOUNDARY_INFERRED = "inferred"
BOUNDARY_UNCERTAIN = "uncertain"

# Entity kinds specific enough to anchor an episode. `topic` is excluded on
# purpose: "hiking" is shared by every hike ever recorded and would merge
# unrelated trips years apart into one episode. A topic still contributes to
# the association graph, just not to episode identity.
ANCHOR_KINDS = ("person", "organization", "location", "object")

# How many anchor entities a *dated* memory must share with an episode. One is
# enough when the dates already agree -- two independent signals pointing the
# same way.
MIN_SHARED_ANCHORS_WITH_DATE = 1
# Without a date, entity overlap is the only evidence, so it has to be
# stronger. Two shared anchors ("Markus" AND "Lake Bled") is a real
# coincidence; one ("Markus") is just a recurring person.
MIN_SHARED_ANCHORS_WITHOUT_DATE = 2

# How many entities to name in a composed title before summarising.
TITLE_ENTITY_LIMIT = 2


@dataclass
class EpisodeAssignment:
    episode_id: uuid.UUID
    rule: str
    created: bool


def _anchor_entities(db: Session, unit_ids: list[uuid.UUID]) -> dict[uuid.UUID, set[uuid.UUID]]:
    """Anchor-kind entity ids per unit."""
    if not unit_ids:
        return {}
    rows = db.execute(
        select(UnitEntityLink.unit_id, UnitEntityLink.entity_id)
        .join(MemoryEntity, MemoryEntity.id == UnitEntityLink.entity_id)
        .where(UnitEntityLink.unit_id.in_(unit_ids), MemoryEntity.kind.in_(ANCHOR_KINDS))
    ).all()
    out: dict[uuid.UUID, set[uuid.UUID]] = {}
    for unit_id, entity_id in rows:
        out.setdefault(unit_id, set()).add(entity_id)
    return out


def _unit_date(unit: MemoryUnit) -> tuple[dt.date | None, str, bool]:
    """`(date, precision, user_stated)` for one memory.

    A date in `explicit_metadata` is the user's own; one in `semantic` was
    extracted by the model. The distinction decides the episode's
    `boundary_confidence`, so a span the user actually stated is never
    presented with the same authority as one inferred from extraction.
    """
    explicit = (unit.explicit_metadata or {}).get("date")
    if isinstance(explicit, str) and explicit.strip():
        parsed, precision = parse_event_date(explicit)
        if parsed is not None:
            return parsed, precision, True
    parsed, precision = parse_event_date((unit.semantic or {}).get("event_date"))
    return parsed, precision, False


# Public alias. app/api/episodes.py renders each member's own date and has to
# use the exact rule the episode's span was computed from: reading
# `semantic.event_date` alone there made a memory whose date the USER typed
# (`explicit_metadata["date"]`) display as "date unknown" inside an episode
# whose span came from precisely that date -- the authority hierarchy
# (docs/ARCHITECTURE.md: explicit_metadata > semantic) inverted in the one
# place the user actually sees it.
unit_date = _unit_date


def _episode_member_units(db: Session, episode_id: uuid.UUID) -> list[MemoryUnit]:
    return list(
        db.execute(
            select(MemoryUnit)
            .join(EpisodeMembership, EpisodeMembership.unit_id == MemoryUnit.id)
            .where(EpisodeMembership.episode_id == episode_id, MemoryUnit.status == "ready")
        ).scalars()
    )


def _compose_title(db: Session, units: list[MemoryUnit], span: tuple) -> str:
    """Deterministic title from dominant anchor entities plus the date span."""
    started, _ended, precision, _confidence = span
    entity_rows = db.execute(
        select(MemoryEntity.display_name, func.count(UnitEntityLink.id))
        .join(UnitEntityLink, UnitEntityLink.entity_id == MemoryEntity.id)
        .where(
            UnitEntityLink.unit_id.in_([u.id for u in units]),
            MemoryEntity.kind.in_(ANCHOR_KINDS),
        )
        .group_by(MemoryEntity.display_name)
        .order_by(func.count(UnitEntityLink.id).desc(), MemoryEntity.display_name)
        .limit(TITLE_ENTITY_LIMIT)
    ).all()
    names = [row[0] for row in entity_rows]

    when = describe(started, precision) if started is not None else ""
    if names and when:
        return f"{', '.join(names)} — {when}"
    if names:
        return ", ".join(names)
    if when:
        return f"Memories from {when}"
    # Nothing nameable and no date: fall back to the seed memory's own words,
    # which is still more useful than "Untitled episode".
    first = units[0].authoritative_content.strip() if units else ""
    return (first[:60].rstrip() + "...") if len(first) > 60 else (first or "Untitled episode")


def _compose_summary(units: list[MemoryUnit], span: tuple) -> str:
    started, ended, precision, confidence = span
    count = len(units)
    noun = "memory" if count == 1 else "memories"
    if started is None:
        return f"{count} {noun}, with no recorded date."
    start_text = describe(started, precision)
    end_text = describe(ended, precision) if ended is not None else start_text
    # Compare the RENDERED endpoints, not the raw dates. Two dates a month
    # apart both render as "2021" at year precision, and the range form then
    # read "from 2021 to 2021" -- a span the episode cannot actually claim to
    # know, stated twice. `_span_display` in app/api/episodes.py already
    # collapses this; the summary has to agree with it.
    if ended is not None and end_text != start_text:
        return (
            f"{count} {noun} from {start_text} to {end_text}"
            + (" (boundaries approximate)." if confidence == BOUNDARY_UNCERTAIN else ".")
        )
    return (
        f"{count} {noun} from {start_text}"
        + (" (boundaries approximate)." if confidence == BOUNDARY_UNCERTAIN else ".")
    )


def _compute_span(units: list[MemoryUnit]) -> tuple[dt.date | None, dt.date | None, str, str]:
    """`(started_on, ended_on, precision, boundary_confidence)` for a member set.

    The span covers only members that actually have a date. If any member has
    none, `boundary_confidence` drops to "uncertain" -- the real episode may
    extend past what is recorded, and saying so is more useful than a
    confident-looking span derived from a subset.
    """
    dated = []
    undated = 0
    all_user_stated = True
    for unit in units:
        parsed, precision, user_stated = _unit_date(unit)
        if parsed is None or precision == PRECISION_UNKNOWN:
            undated += 1
            continue
        dated.append((parsed, precision))
        all_user_stated = all_user_stated and user_stated

    if not dated:
        return None, None, PRECISION_UNKNOWN, BOUNDARY_UNCERTAIN

    started = min(d for d, _p in dated)
    ended = max(d for d, _p in dated)
    precision = PRECISION_UNKNOWN
    for _d, p in dated:
        precision = looser_precision(precision, p) if precision != PRECISION_UNKNOWN else p
    for _d, p in dated:
        precision = looser_precision(precision, p)

    if undated:
        confidence = BOUNDARY_UNCERTAIN
    elif all_user_stated:
        confidence = BOUNDARY_ASSERTED
    else:
        confidence = BOUNDARY_INFERRED
    return started, ended, precision, confidence


def refresh_episode(db: Session, episode_id: uuid.UUID) -> bool:
    """Recompute an episode's span, title, and summary from its current members.

    Returns False and deletes the episode if it has no ready members left --
    an empty episode is not a memory of anything, and leaving it behind would
    put a phantom entry on the timeline. Does not commit.
    """
    episode = db.get(Episode, episode_id)
    if episode is None:
        return False
    units = _episode_member_units(db, episode_id)
    if not units:
        db.delete(episode)
        return False

    span = _compute_span(units)
    episode.started_on, episode.ended_on, episode.date_precision, episode.boundary_confidence = span
    episode.title = _compose_title(db, units, span)
    episode.title_source = "derived"
    episode.summary = _compose_summary(units, span)
    # An episode is only as user-asserted as its weakest grouping: if any
    # member was grouped by inference, the episode as a whole is inferred.
    origins = {m.origin for m in episode.memberships}
    if origins == {ORIGIN_EXPLICIT}:
        episode.origin = ORIGIN_EXPLICIT
    elif ORIGIN_INFERRED in origins:
        episode.origin = ORIGIN_INFERRED
    else:
        episode.origin = ORIGIN_EXTRACTED
    return True


def _candidate_episodes(db: Session, unit: MemoryUnit, anchors: set[uuid.UUID]) -> list[Episode]:
    """Episodes worth testing for this memory: those with a member sharing at
    least one anchor entity, plus those already holding a sibling from the same
    capture. Keeps the comparison bounded to plausibly-related episodes instead
    of scanning every episode in the store."""
    episode_ids: set[uuid.UUID] = set()

    sibling_rows = db.execute(
        select(EpisodeMembership.episode_id)
        .join(MemoryUnit, MemoryUnit.id == EpisodeMembership.unit_id)
        .where(MemoryUnit.capture_id == unit.capture_id, MemoryUnit.id != unit.id)
    ).scalars()
    episode_ids.update(sibling_rows)

    if anchors:
        anchor_rows = db.execute(
            select(EpisodeMembership.episode_id)
            .join(UnitEntityLink, UnitEntityLink.unit_id == EpisodeMembership.unit_id)
            .where(UnitEntityLink.entity_id.in_(list(anchors)))
        ).scalars()
        episode_ids.update(anchor_rows)

    if not episode_ids:
        return []
    return list(
        db.execute(
            select(Episode).where(Episode.id.in_(list(episode_ids))).order_by(Episode.created_at)
        ).scalars()
    )


def _shared_anchor_count(
    anchors: set[uuid.UUID], member_anchors: dict[uuid.UUID, set[uuid.UUID]]
) -> tuple[int, set[uuid.UUID]]:
    union: set[uuid.UUID] = set()
    for member_set in member_anchors.values():
        union |= member_set
    shared = anchors & union
    return len(shared), shared


def assign_episode(db: Session, unit: MemoryUnit) -> EpisodeAssignment | None:
    """Place one ready memory into an episode, creating one if needed.

    Greedy and order-dependent by design: a memory joins the first (oldest)
    candidate episode it is compatible with, rather than the globally optimal
    one. Global clustering would mean recomputing every episode on every new
    memory, and re-running this over the corpus
    (`scripts/rebuild_graph.py --episodes`) converges to a stable grouping, so
    the greedy result is a cheap approximation that can be corrected rather
    than a permanent mistake.

    Does not commit -- the caller owns the transaction, same contract as
    app/memory/graph.py::project_unit.
    """
    if unit.status != "ready":
        return None

    anchors = _anchor_entities(db, [unit.id]).get(unit.id, set())
    unit_date, unit_precision, _user_stated = _unit_date(unit)
    has_date = unit_date is not None and unit_precision != PRECISION_UNKNOWN

    for episode in _candidate_episodes(db, unit, anchors):
        members = _episode_member_units(db, episode.id)
        if not members:
            continue
        member_ids = [m.id for m in members]

        # Rule 1: a sibling from the same submission is already here.
        if any(m.capture_id == unit.capture_id for m in members):
            _add_membership(
                db,
                episode,
                unit,
                rule=RULE_SAME_CAPTURE,
                origin=ORIGIN_EXTRACTED,
                confidence=CONFIDENCE_CERTAIN,
                shared=set(),
                explanation="You wrote this in the same note as another memory in this episode.",
            )
            refresh_episode(db, episode.id)
            return EpisodeAssignment(episode_id=episode.id, rule=RULE_SAME_CAPTURE, created=False)

        member_anchors = _anchor_entities(db, member_ids)
        shared_count, shared = _shared_anchor_count(anchors, member_anchors)

        # Rule 2: the dates agree (at the precision each is actually known)
        # AND there is entity overlap.
        if has_date and shared_count >= MIN_SHARED_ANCHORS_WITH_DATE:
            compatible = episode.started_on is not None and (
                dates_compatible(unit_date, unit_precision, episode.started_on, episode.date_precision)
                or (
                    episode.ended_on is not None
                    and dates_compatible(
                        unit_date, unit_precision, episode.ended_on, episode.date_precision
                    )
                )
            )
            if compatible:
                names = _entity_names(db, shared)
                _add_membership(
                    db,
                    episode,
                    unit,
                    rule=RULE_TEMPORAL_ENTITY,
                    origin=ORIGIN_INFERRED,
                    confidence=CONFIDENCE_LIKELY,
                    shared=shared,
                    explanation=(
                        f"Its date fits this episode and it shares {', '.join(names)} with it."
                    ),
                )
                refresh_episode(db, episode.id)
                return EpisodeAssignment(
                    episode_id=episode.id, rule=RULE_TEMPORAL_ENTITY, created=False
                )

        # Rule 3: no usable date, so entity overlap alone must be strong.
        if not has_date and shared_count >= MIN_SHARED_ANCHORS_WITHOUT_DATE:
            names = _entity_names(db, shared)
            _add_membership(
                db,
                episode,
                unit,
                rule=RULE_ENTITY_ONLY,
                origin=ORIGIN_INFERRED,
                confidence=CONFIDENCE_UNCERTAIN,
                shared=shared,
                explanation=(
                    f"It has no recorded date, but shares {', '.join(names)} with this episode."
                ),
            )
            refresh_episode(db, episode.id)
            return EpisodeAssignment(episode_id=episode.id, rule=RULE_ENTITY_ONLY, created=False)

    # Nothing matched: this memory starts its own episode.
    episode = Episode(
        title="",
        title_source="derived",
        summary="",
        origin=ORIGIN_INFERRED,
        boundary_confidence=BOUNDARY_UNCERTAIN,
        date_precision=PRECISION_UNKNOWN,
        provenance={"seed_unit_id": str(unit.id), "rule": RULE_SEED},
    )
    db.add(episode)
    db.flush()
    _add_membership(
        db,
        episode,
        unit,
        rule=RULE_SEED,
        origin=ORIGIN_EXTRACTED,
        confidence=CONFIDENCE_CERTAIN,
        shared=set(),
        explanation="This memory started the episode.",
    )
    refresh_episode(db, episode.id)
    return EpisodeAssignment(episode_id=episode.id, rule=RULE_SEED, created=True)


def _entity_names(db: Session, entity_ids: set[uuid.UUID]) -> list[str]:
    if not entity_ids:
        return []
    return [
        row[0]
        for row in db.execute(
            select(MemoryEntity.display_name)
            .where(MemoryEntity.id.in_(list(entity_ids)))
            .order_by(MemoryEntity.display_name)
            .limit(3)
        ).all()
    ]


def _add_membership(
    db: Session,
    episode: Episode,
    unit: MemoryUnit,
    *,
    rule: str,
    origin: str,
    confidence: str,
    shared: set[uuid.UUID],
    explanation: str,
) -> None:
    existing = db.execute(
        select(EpisodeMembership).where(
            EpisodeMembership.episode_id == episode.id, EpisodeMembership.unit_id == unit.id
        )
    ).scalar_one_or_none()
    if existing is not None:
        return
    db.add(
        EpisodeMembership(
            episode_id=episode.id,
            unit_id=unit.id,
            origin=origin,
            confidence=confidence,
            provenance={
                "rule": rule,
                "shared_entity_ids": [str(e) for e in sorted(shared, key=str)],
                "explanation": explanation,
            },
        )
    )
    db.flush()


def remove_unit_from_episodes(db: Session, unit_id: uuid.UUID) -> None:
    """Drop a memory's episode memberships and tidy up any episode left empty.
    Does not commit."""
    affected = list(
        db.execute(
            select(EpisodeMembership.episode_id).where(EpisodeMembership.unit_id == unit_id)
        ).scalars()
    )
    db.execute(delete(EpisodeMembership).where(EpisodeMembership.unit_id == unit_id))
    db.flush()
    for episode_id in set(affected):
        refresh_episode(db, episode_id)
    db.flush()


def timeline(
    db: Session,
    *,
    limit: int = 50,
    offset: int = 0,
    include_undated: bool = True,
) -> tuple[list[Episode], int]:
    """Episodes in chronological order, newest first, undated ones last.

    Undated episodes sort last rather than being hidden or given a fabricated
    position: they are real memories whose date is genuinely unknown, and the
    timeline should say so instead of guessing where they go.
    """
    query = select(Episode)
    if not include_undated:
        query = query.where(Episode.started_on.is_not(None))
    total = int(
        db.execute(select(func.count()).select_from(query.subquery())).scalar() or 0
    )
    episodes = list(
        db.execute(
            query.order_by(
                Episode.started_on.is_(None),  # False (dated) sorts before True
                Episode.started_on.desc(),
                Episode.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        ).scalars()
    )
    return episodes, total


def episodes_for_unit(db: Session, unit_id: uuid.UUID) -> list[tuple[Episode, EpisodeMembership]]:
    rows = db.execute(
        select(Episode, EpisodeMembership)
        .join(EpisodeMembership, EpisodeMembership.episode_id == Episode.id)
        .where(EpisodeMembership.unit_id == unit_id)
        .order_by(Episode.started_on.desc().nullslast())
    ).all()
    return [(episode, membership) for episode, membership in rows]


def overlapping_episodes(db: Session, episode: Episode, *, limit: int = 8) -> list[Episode]:
    """Other episodes whose span overlaps this one's, for "related episodes"
    navigation. Returns [] for an undated episode -- an unknown span cannot be
    said to overlap anything, and listing arbitrary neighbours would imply a
    temporal relationship that is not known to exist."""
    if episode.started_on is None:
        return []
    ended = episode.ended_on or episode.started_on
    return list(
        db.execute(
            select(Episode)
            .where(
                Episode.id != episode.id,
                Episode.started_on.is_not(None),
                Episode.started_on <= ended,
                func.coalesce(Episode.ended_on, Episode.started_on) >= episode.started_on,
            )
            .order_by(Episode.started_on.desc())
            .limit(limit)
        ).scalars()
    )


def anchor_counts(db: Session, episode_id: uuid.UUID) -> Counter:
    """Anchor-entity frequency within one episode, for display.

    Grouped by normalized name, not by display name. Entity identity is
    (kind, normalized_name), so the same string can legitimately exist as two
    entities of different kinds -- and grouping on the display name then listed
    both spellings, producing anchors like `["Markus", "The department", "the
    department"]`. For a reader they are one thing; the first-seen
    capitalisation wins.
    """
    rows = db.execute(
        select(
            MemoryEntity.normalized_name,
            func.min(MemoryEntity.display_name),
            func.count(UnitEntityLink.id),
        )
        .join(UnitEntityLink, UnitEntityLink.entity_id == MemoryEntity.id)
        .join(EpisodeMembership, EpisodeMembership.unit_id == UnitEntityLink.unit_id)
        .where(EpisodeMembership.episode_id == episode_id, MemoryEntity.kind.in_(ANCHOR_KINDS))
        .group_by(MemoryEntity.normalized_name)
    ).all()
    return Counter({row[1]: int(row[2]) for row in rows})
