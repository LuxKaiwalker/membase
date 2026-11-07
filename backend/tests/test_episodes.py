import datetime as dt
import json
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from app.api.episodes import get_episode, get_episodes_for_unit, get_timeline
from app.ingestion.pipeline import delete_capture
from app.memory.episodes import (
    BOUNDARY_ASSERTED,
    BOUNDARY_INFERRED,
    BOUNDARY_UNCERTAIN,
    CONFIDENCE_CERTAIN,
    CONFIDENCE_LIKELY,
    CONFIDENCE_UNCERTAIN,
    RULE_ENTITY_ONLY,
    RULE_SAME_CAPTURE,
    RULE_SEED,
    RULE_TEMPORAL_ENTITY,
    anchor_counts,
    assign_episode,
    episodes_for_unit,
    overlapping_episodes,
    refresh_episode,
    remove_unit_from_episodes,
    timeline,
)
from app.memory.graph import project_unit
from app.models import (
    ORIGIN_EXTRACTED,
    ORIGIN_INFERRED,
    Capture,
    Episode,
    EpisodeMembership,
    MemoryUnit,
)

# Episodes are a *projection* over MemoryUnit dates and entity links (see
# app/memory/episodes.py): deterministic, LLM-free. These tests therefore build
# the projection's input directly -- hand-written ready units whose
# `semantic.event_date`, `explicit_metadata["date"]`, and entities are exactly
# what the clustering rules are being asked about. Where a test needs a
# *particular span* rather than a particular member set (timeline ordering,
# overlap), the Episode row is built directly, since timeline() and
# overlapping_episodes() read nothing but those columns.


def _ready_unit(db, *, text, semantic=None, explicit_metadata=None, capture=None, status="ready"):
    """A capture plus one unit, written straight to the DB."""
    metadata = explicit_metadata or {}
    if capture is None:
        capture = Capture(raw_text=text, explicit_metadata=metadata)
        db.add(capture)
        db.flush()
    unit = MemoryUnit(
        capture_id=capture.id,
        authoritative_content=text,
        status=status,
        semantic=semantic or {},
        explicit_metadata=metadata,
        retrieval_text=text,
        searchable_text=text,
    )
    db.add(unit)
    db.commit()
    db.refresh(unit)
    return unit


def _capture(db, text="one submission"):
    capture = Capture(raw_text=text, explicit_metadata={})
    db.add(capture)
    db.commit()
    return capture


def _project(db, unit):
    """`project_unit` plus the commit the caller owns.

    Committing per projection matters for more than durability: episodes are
    compared in `created_at` order, and rows created inside one transaction all
    share that transaction's timestamp.
    """
    result = project_unit(db, unit)
    db.commit()
    return result


def _episode_row(
    db,
    *,
    title="an episode",
    started=None,
    ended=None,
    precision="unknown",
    boundary=BOUNDARY_UNCERTAIN,
):
    episode = Episode(
        title=title,
        title_source="derived",
        summary="",
        started_on=started,
        ended_on=ended,
        date_precision=precision,
        boundary_confidence=boundary,
        origin=ORIGIN_INFERRED,
        provenance={},
    )
    db.add(episode)
    db.commit()
    db.refresh(episode)
    return episode


def _membership_row(db, episode, unit, *, origin=ORIGIN_INFERRED, confidence=CONFIDENCE_UNCERTAIN):
    membership = EpisodeMembership(
        episode_id=episode.id,
        unit_id=unit.id,
        origin=origin,
        confidence=confidence,
        provenance={"rule": RULE_SEED, "shared_entity_ids": [], "explanation": "built by a test"},
    )
    db.add(membership)
    db.commit()
    return membership


def _count(db, column, *conditions):
    query = select(func.count(column))
    for condition in conditions:
        query = query.where(condition)
    return int(db.execute(query).scalar() or 0)


def _episodes(db):
    return list(db.execute(select(Episode).order_by(Episode.created_at)).scalars())


def _membership(db, unit_id):
    return db.execute(
        select(EpisodeMembership).where(EpisodeMembership.unit_id == unit_id)
    ).scalar_one()


def _episode_of(db, unit_id):
    return db.get(Episode, _membership(db, unit_id).episode_id)


# --------------------------------------------------------------------------
# Clustering rules
# --------------------------------------------------------------------------


def test_two_memories_from_one_capture_land_in_the_same_episode_as_certain(db):
    capture = _capture(db, "Hiked to the lake with Markus. The hut served rösti.")
    first = _ready_unit(
        db,
        text="Hiked to the lake with Markus.",
        semantic={"people": ["Markus"], "event_date": "2023-07-10"},
        capture=capture,
    )
    second = _ready_unit(
        db,
        text="The hut served rösti.",
        semantic={"topics": ["food"]},
        capture=capture,
    )

    _project(db, first)
    result = _project(db, second)

    assert result.episode_rule == RULE_SAME_CAPTURE
    assert result.episode_created is False
    assert _count(db, Episode.id) == 1
    membership = _membership(db, second.id)
    # "extracted"/"certain": the user really did write these in one go, so
    # nothing about this grouping was guessed.
    assert membership.origin == ORIGIN_EXTRACTED
    assert membership.confidence == CONFIDENCE_CERTAIN
    assert membership.provenance["rule"] == RULE_SAME_CAPTURE
    assert membership.provenance["explanation"].strip() != ""


def test_a_dated_memory_sharing_an_anchor_joins_a_temporally_compatible_episode(db):
    seed = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"], "event_date": "2023-07-10"},
    )
    _project(db, seed)
    later = _ready_unit(
        db,
        text="Markus drove us back that evening.",
        semantic={"people": ["Markus"], "event_date": "2023-07-10"},
    )

    result = _project(db, later)

    assert result.episode_rule == RULE_TEMPORAL_ENTITY
    assert _count(db, Episode.id) == 1
    assert _episode_of(db, later.id).id == _episode_of(db, seed.id).id
    membership = _membership(db, later.id)
    # Two independent signals (a compatible date AND a shared person) make this
    # "likely" -- but it is still the system's inference, not the user's claim.
    assert membership.origin == ORIGIN_INFERRED
    assert membership.confidence == CONFIDENCE_LIKELY
    assert "Markus" in membership.provenance["explanation"]
    assert membership.provenance["shared_entity_ids"] != []


def test_a_shared_anchor_with_an_incompatible_date_seeds_its_own_episode(db):
    """The rule that stops a recurring person collapsing years of memories into
    one episode: "Markus" is shared, but two years apart is not one
    experience."""
    first = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "event_date": "2021-07-10"},
    )
    _project(db, first)
    second = _ready_unit(
        db,
        text="Markus came over for dinner.",
        semantic={"people": ["Markus"], "event_date": "2023-07-10"},
    )

    result = _project(db, second)

    assert result.episode_rule == RULE_SEED
    assert result.episode_created is True
    assert _count(db, Episode.id) == 2
    assert _episode_of(db, second.id).id != _episode_of(db, first.id).id


def test_an_undated_memory_sharing_only_one_anchor_does_not_join(db):
    seed = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"], "event_date": "2023-07-10"},
    )
    _project(db, seed)
    undated = _ready_unit(db, text="Markus recommended a book.", semantic={"people": ["Markus"]})

    result = _project(db, undated)

    # With no date, entity overlap is the only evidence, and one recurring
    # person is not enough of a coincidence (MIN_SHARED_ANCHORS_WITHOUT_DATE).
    assert result.episode_rule == RULE_SEED
    assert _count(db, Episode.id) == 2
    assert _episode_of(db, undated.id).id != _episode_of(db, seed.id).id


def test_an_undated_memory_sharing_two_anchors_joins_as_uncertain(db):
    seed = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"], "event_date": "2023-07-10"},
    )
    _project(db, seed)
    undated = _ready_unit(
        db,
        text="Markus took a photo of the boats at Lake Bled.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"]},
    )

    result = _project(db, undated)

    assert result.episode_rule == RULE_ENTITY_ONLY
    assert _count(db, Episode.id) == 1
    membership = _membership(db, undated.id)
    assert membership.origin == ORIGIN_INFERRED
    # Grouped on entity overlap alone, so the weakest confidence the schema has.
    assert membership.confidence == CONFIDENCE_UNCERTAIN
    assert "no recorded date" in membership.provenance["explanation"]
    episode = _episode_of(db, undated.id)
    # An undated member never widens the span, but it does make the boundaries
    # openly uncertain.
    assert episode.started_on == dt.date(2023, 7, 10)
    assert episode.ended_on == dt.date(2023, 7, 10)
    assert episode.boundary_confidence == BOUNDARY_UNCERTAIN


def test_sharing_only_a_topic_never_groups_two_memories(db):
    """`topic` is excluded from ANCHOR_KINDS on purpose: "hiking" is shared by
    every hike ever recorded, so grouping on it would merge unrelated trips."""
    first = _ready_unit(db, text="Hiked up the Uetliberg.", semantic={"topics": ["hiking"]})
    _project(db, first)
    second = _ready_unit(db, text="Hiked around Oeschinensee.", semantic={"topics": ["hiking"]})

    _project(db, second)

    assert _count(db, Episode.id) == 2
    assert _episode_of(db, second.id).id != _episode_of(db, first.id).id


def test_a_unit_that_is_not_ready_is_never_assigned_to_an_episode(db):
    unit = _ready_unit(
        db,
        text="Half-finished note about Markus.",
        semantic={"people": ["Markus"], "event_date": "2023-07-10"},
        status="awaiting_clarification",
    )

    assert assign_episode(db, unit) is None
    db.commit()
    # `awaiting_clarification` content is invisible to retrieval; an episode
    # holding it would be a backdoor around that status gate.
    assert _count(db, Episode.id) == 0
    assert _count(db, EpisodeMembership.id) == 0


def test_assigning_the_same_memory_twice_creates_no_second_episode_or_membership(db):
    unit = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"], "event_date": "2023-07-10"},
    )
    first = _project(db, unit)

    second = assign_episode(db, unit)
    db.commit()

    assert second is not None
    assert str(second.episode_id) == first.episode_id
    assert second.created is False
    assert _count(db, Episode.id) == 1
    assert _count(db, EpisodeMembership.id, EpisodeMembership.unit_id == unit.id) == 1


# --------------------------------------------------------------------------
# Dates and boundaries -- the honesty requirements
# --------------------------------------------------------------------------


def test_an_episode_with_no_dated_member_is_never_back_filled_from_created_at(db):
    unit = _ready_unit(db, text="Repotted the monstera.", semantic={"topics": ["gardening"]})

    _project(db, unit)

    episode = _episode_of(db, unit.id)
    # The central promise of the module: "I don't know when this was" and "this
    # happened the day it was written down" are different claims, and the
    # second one is never made up. The row was created today; the span is NULL.
    assert episode.started_on is None
    assert episode.ended_on is None
    assert episode.created_at.date() == dt.date.today()
    assert episode.date_precision == "unknown"
    assert episode.boundary_confidence == BOUNDARY_UNCERTAIN
    assert "no recorded date" in episode.summary


def test_boundary_confidence_is_asserted_when_every_date_came_from_the_user(db):
    capture = _capture(db, "Two notes from the same submission.")
    first = _ready_unit(
        db,
        text="Arrived at the hut.",
        explicit_metadata={"date": "2023-07-10"},
        capture=capture,
    )
    second = _ready_unit(
        db,
        text="Walked back down.",
        explicit_metadata={"date": "2023-07-12"},
        capture=capture,
    )

    _project(db, first)
    _project(db, second)

    episode = _episode_of(db, second.id)
    # Both dates are the user's own (explicit_metadata), so the span is the
    # user's claim rather than the system's reading of their words.
    assert episode.boundary_confidence == BOUNDARY_ASSERTED
    assert episode.started_on == dt.date(2023, 7, 10)
    assert episode.ended_on == dt.date(2023, 7, 12)


def test_boundary_confidence_is_inferred_when_the_date_came_from_extraction(db):
    unit = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "event_date": "2023-07-10"},
    )

    _project(db, unit)

    episode = _episode_of(db, unit.id)
    # Same date, weaker claim: a model extracted it from the user's words.
    assert episode.boundary_confidence == BOUNDARY_INFERRED
    assert episode.started_on == dt.date(2023, 7, 10)


def test_boundary_confidence_is_uncertain_when_one_member_has_no_date_at_all(db):
    capture = _capture(db, "One dated note and one undated one.")
    dated = _ready_unit(
        db, text="Arrived at the hut.", semantic={"event_date": "2023-07-10"}, capture=capture
    )
    undated = _ready_unit(db, text="The rösti was excellent.", semantic={}, capture=capture)

    _project(db, dated)
    _project(db, undated)

    episode = _episode_of(db, undated.id)
    # The real episode may extend past what is recorded; saying so beats a
    # confident-looking span computed from a subset of its members.
    assert episode.boundary_confidence == BOUNDARY_UNCERTAIN
    assert episode.started_on == dt.date(2023, 7, 10)


def test_date_precision_is_the_loosest_among_the_members(db):
    capture = _capture(db, "A precise note and a vague one.")
    precise = _ready_unit(
        db, text="Arrived on the tenth.", semantic={"event_date": "2023-07-10"}, capture=capture
    )
    vague = _ready_unit(
        db, text="Sometime that year we went back.", semantic={"event_date": "2023"}, capture=capture
    )

    _project(db, precise)
    _project(db, vague)

    episode = _episode_of(db, vague.id)
    # A span is only as precise as its vaguest endpoint -- claiming "day" here
    # would present a guessed anchor (1 January) as a known date.
    assert episode.date_precision == "year"


def test_the_span_covers_only_members_that_actually_have_a_date(db):
    capture = _capture(db, "Three notes, one of them undated.")
    early = _ready_unit(db, text="Set off.", semantic={"event_date": "2023-07-10"}, capture=capture)
    late = _ready_unit(db, text="Came home.", semantic={"event_date": "2023-07-20"}, capture=capture)
    undated = _ready_unit(db, text="Someone lost a glove.", semantic={}, capture=capture)

    for unit in (early, late, undated):
        _project(db, unit)

    episode = _episode_of(db, undated.id)
    assert episode.started_on == dt.date(2023, 7, 10)
    assert episode.ended_on == dt.date(2023, 7, 20)
    assert _count(db, EpisodeMembership.id, EpisodeMembership.episode_id == episode.id) == 3


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_refresh_episode_deletes_an_episode_with_no_ready_members_left(db):
    unit = _ready_unit(db, text="Repotted the monstera.", semantic={"topics": ["gardening"]})
    _project(db, unit)
    episode_id = _episode_of(db, unit.id).id

    unit.status = "discarded"
    db.commit()
    kept = refresh_episode(db, episode_id)
    db.commit()

    # An empty episode is not a memory of anything, and leaving it behind would
    # put a phantom entry on the timeline.
    assert kept is False
    assert db.get(Episode, episode_id) is None
    assert _count(db, EpisodeMembership.id) == 0


def test_remove_unit_from_episodes_drops_the_membership_and_tidies_the_episode(db):
    unit = _ready_unit(db, text="Repotted the monstera.", semantic={"topics": ["gardening"]})
    _project(db, unit)
    assert _count(db, EpisodeMembership.id) == 1

    remove_unit_from_episodes(db, unit.id)
    db.commit()

    assert _count(db, EpisodeMembership.id) == 0
    assert _count(db, Episode.id) == 0


def test_discarding_a_memory_and_reprojecting_removes_it_from_its_episode(db):
    capture = _capture(db, "Two notes written together.")
    kept_unit = _ready_unit(db, text="Arrived at the hut.", semantic={}, capture=capture)
    cancelled = _ready_unit(db, text="A note I cancelled.", semantic={}, capture=capture)
    _project(db, kept_unit)
    _project(db, cancelled)
    episode_id = _episode_of(db, cancelled.id).id

    cancelled.status = "discarded"
    db.commit()
    _project(db, cancelled)

    assert _count(db, EpisodeMembership.id, EpisodeMembership.unit_id == cancelled.id) == 0
    # The sibling is still ready, so the episode itself survives.
    assert db.get(Episode, episode_id) is not None
    assert _count(db, EpisodeMembership.id, EpisodeMembership.episode_id == episode_id) == 1


def test_deleting_the_capture_cascades_episode_memberships_away(db):
    unit = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "event_date": "2023-07-10"},
    )
    _project(db, unit)
    capture = db.get(Capture, unit.capture_id)
    assert _count(db, EpisodeMembership.id) == 1

    delete_capture(db, capture)

    # Explicit user deletion is the only way a memory goes away, and it has to
    # take every derived row with it.
    assert _count(db, EpisodeMembership.id) == 0
    # NOTE: the Episode row itself survives this with zero members, because
    # `delete_capture` (app/ingestion/pipeline.py) never calls
    # `refresh_episode`. Deliberately not asserted either way here -- it is a
    # real gap in a file this change does not own, not the behaviour of the
    # cascade under test.


# --------------------------------------------------------------------------
# Title and summary composition
# --------------------------------------------------------------------------


def test_the_title_names_the_dominant_anchor_entities_and_the_date_span(db):
    unit = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"], "event_date": "2023-07-10"},
    )

    _project(db, unit)

    episode = _episode_of(db, unit.id)
    assert "Markus" in episode.title
    assert "Lake Bled" in episode.title
    assert "2023-07-10" in episode.title
    # Reserved so a future model-written title can be told apart from a
    # composed one rather than silently replacing it.
    assert episode.title_source == "derived"


def test_a_year_precision_title_never_renders_a_specific_day(db):
    unit = _ready_unit(
        db,
        text="Somewhere in that year we hiked to Lake Bled.",
        semantic={"locations": ["Lake Bled"], "event_date": "2023"},
    )

    _project(db, unit)

    episode = _episode_of(db, unit.id)
    assert "Lake Bled" in episode.title
    assert "2023" in episode.title
    # The anchor date is 1 January 2023, a sortable representative and not a
    # claim about the day -- so neither spelling of it may reach the title.
    assert "2023-01-01" not in episode.title
    assert "January" not in episode.title


def test_an_episode_with_no_entities_and_no_date_uses_the_seed_memorys_own_words(db):
    unit = _ready_unit(
        db, text="Repotted the monstera.", semantic={"topics": ["gardening"]}
    )

    _project(db, unit)

    episode = _episode_of(db, unit.id)
    assert episode.title == "Repotted the monstera."
    assert episode.title != "Untitled episode"


def test_a_long_seed_memory_is_truncated_rather_than_used_whole_as_a_title(db):
    text = (
        "Repotted the monstera into the big terracotta pot and it finally "
        "looks happy again after months of sulking."
    )
    unit = _ready_unit(db, text=text, semantic={"topics": ["gardening"]})

    _project(db, unit)

    episode = _episode_of(db, unit.id)
    assert episode.title.endswith("...")
    assert len(episode.title) <= 63
    assert episode.title.startswith("Repotted the monstera")


# --------------------------------------------------------------------------
# Timeline and query functions
# --------------------------------------------------------------------------


def test_timeline_orders_dated_episodes_newest_first_with_undated_ones_last(db):
    undated = _episode_row(db, title="date unknown")
    older = _episode_row(db, title="older", started=dt.date(2023, 1, 5), precision="day")
    newer = _episode_row(db, title="newer", started=dt.date(2024, 6, 1), precision="day")

    episodes, total = timeline(db)

    # Undated episodes sort last rather than being hidden or given a
    # fabricated position among the dated ones.
    assert [e.id for e in episodes] == [newer.id, older.id, undated.id]
    assert total == 3


def test_timeline_respects_limit_and_offset_and_still_reports_the_full_total(db):
    older = _episode_row(db, title="older", started=dt.date(2023, 1, 5), precision="day")
    newer = _episode_row(db, title="newer", started=dt.date(2024, 6, 1), precision="day")
    _episode_row(db, title="date unknown")

    first_page, total = timeline(db, limit=1)
    second_page, total_again = timeline(db, limit=1, offset=1)

    assert [e.id for e in first_page] == [newer.id]
    assert [e.id for e in second_page] == [older.id]
    assert total == total_again == 3


def test_timeline_can_exclude_undated_episodes_entirely(db):
    dated = _episode_row(db, title="dated", started=dt.date(2024, 6, 1), precision="day")
    undated = _episode_row(db, title="date unknown")

    episodes, total = timeline(db, include_undated=False)

    assert [e.id for e in episodes] == [dated.id]
    assert undated.id not in [e.id for e in episodes]
    assert total == 1


def test_episodes_for_unit_returns_every_episode_the_memory_belongs_to(db):
    unit = _ready_unit(db, text="A talk I gave in Berlin.", semantic={"locations": ["Berlin"]})
    conference = _episode_row(
        db, title="the conference", started=dt.date(2024, 5, 2), precision="day"
    )
    trip = _episode_row(db, title="the Berlin trip", started=dt.date(2024, 5, 1), precision="day")
    _membership_row(db, conference, unit)
    _membership_row(db, trip, unit)

    rows = episodes_for_unit(db, unit.id)

    # Multi-membership is deliberate: a talk belongs to both "the conference"
    # and "the Berlin trip", and collapsing that to one would discard a real
    # relationship.
    assert [episode.id for episode, _membership in rows] == [conference.id, trip.id]
    assert all(membership.unit_id == unit.id for _episode, membership in rows)


def test_overlapping_episodes_returns_only_spans_that_actually_overlap(db):
    subject = _episode_row(
        db, title="first week", started=dt.date(2023, 7, 1), ended=dt.date(2023, 7, 10), precision="day"
    )
    overlapping = _episode_row(
        db, title="second week", started=dt.date(2023, 7, 5), ended=dt.date(2023, 7, 20), precision="day"
    )
    _episode_row(db, title="next year", started=dt.date(2024, 1, 1), precision="day")
    _episode_row(db, title="date unknown")

    found = overlapping_episodes(db, subject)

    assert [e.id for e in found] == [overlapping.id]


def test_overlapping_episodes_of_an_undated_episode_is_empty(db):
    undated = _episode_row(db, title="date unknown")
    _episode_row(db, title="dated", started=dt.date(2023, 7, 5), precision="day")

    # An unknown span cannot be said to overlap anything; listing arbitrary
    # neighbours would imply a temporal relationship that is not known to exist.
    assert overlapping_episodes(db, undated) == []


def test_anchor_counts_reports_per_entity_frequency_within_the_episode(db):
    capture = _capture(db, "Two notes from the same trip.")
    first = _ready_unit(
        db,
        text="Hiked to Lake Bled with Markus.",
        semantic={"people": ["Markus"], "locations": ["Lake Bled"]},
        capture=capture,
    )
    second = _ready_unit(
        db, text="Markus drove us back.", semantic={"people": ["Markus"]}, capture=capture
    )
    _project(db, first)
    _project(db, second)

    counts = anchor_counts(db, _episode_of(db, first.id).id)

    assert counts["Markus"] == 2
    assert counts["Lake Bled"] == 1
    assert counts.most_common(1)[0][0] == "Markus"


# --------------------------------------------------------------------------
# API layer -- route functions called directly (this repo has no TestClient)
# --------------------------------------------------------------------------


def test_get_timeline_returns_a_timeline_with_the_undated_count(db):
    older = _episode_row(db, title="older", started=dt.date(2023, 1, 5), precision="day")
    newer = _episode_row(db, title="newer", started=dt.date(2024, 6, 1), precision="day")
    undated = _episode_row(db, title="date unknown")

    payload = get_timeline(db=db, limit=50, offset=0, include_undated=True)

    assert [e.id for e in payload.episodes] == [newer.id, older.id, undated.id]
    assert payload.total == 3
    # Counted separately so the UI can say "N memories whose date isn't known"
    # instead of hiding them.
    assert payload.undated_count == 1
    assert payload.episodes[2].date_display == "date unknown"
    assert json.dumps(payload.model_dump(mode="json"))


def test_get_timeline_excludes_undated_episodes_but_still_counts_them(db):
    _episode_row(db, title="dated", started=dt.date(2024, 6, 1), precision="day")
    _episode_row(db, title="date unknown")

    payload = get_timeline(db=db, limit=50, offset=0, include_undated=False)

    assert len(payload.episodes) == 1
    assert payload.total == 1
    assert payload.undated_count == 1


def test_get_episode_sorts_dated_members_first_and_renders_dates_at_known_precision(db):
    capture = _capture(db, "Three notes from one submission.")
    vague = _ready_unit(
        db, text="Sometime that year.", semantic={"event_date": "2023"}, capture=capture
    )
    precise = _ready_unit(
        db, text="Walked back down.", semantic={"event_date": "2023-07-20"}, capture=capture
    )
    undated = _ready_unit(db, text="Someone lost a glove.", semantic={}, capture=capture)
    for unit in (vague, precise, undated):
        _project(db, unit)
    episode = _episode_of(db, vague.id)
    neighbour = _episode_row(
        db, title="overlapping", started=dt.date(2023, 3, 1), precision="day"
    )

    detail = get_episode(episode.id, db=db)

    assert [m.unit_id for m in detail.members] == [vague.id, precise.id, undated.id]
    # A year-precision member must not display as 1 January.
    assert detail.members[0].date_display == "2023"
    assert detail.members[1].date_display == "2023-07-20"
    assert detail.members[2].date_display == "date unknown"
    assert detail.members[2].occurred_on is None
    assert detail.episode.member_count == 3
    assert neighbour.id in [e.id for e in detail.overlapping]
    assert json.dumps(detail.model_dump(mode="json"))


def test_get_episode_shows_a_member_date_the_user_stated_themselves(db):
    unit = _ready_unit(
        db, text="Arrived at the hut.", explicit_metadata={"date": "2023-07-10"}
    )
    _project(db, unit)

    detail = get_episode(_episode_of(db, unit.id).id, db=db)

    # The episode's own span is computed from this date (authority hierarchy:
    # user-typed beats model-extracted), so the member it came from cannot be
    # displayed as undated.
    assert detail.members[0].occurred_on == "2023-07-10"
    assert detail.members[0].date_precision == "day"
    assert detail.members[0].date_display == "2023-07-10"


def test_get_episode_omits_a_member_that_is_no_longer_ready(db):
    capture = _capture(db, "Two notes written together.")
    kept_unit = _ready_unit(db, text="Arrived at the hut.", semantic={}, capture=capture)
    hidden = _ready_unit(db, text="A half-finished note.", semantic={}, capture=capture)
    _project(db, kept_unit)
    _project(db, hidden)
    episode_id = _episode_of(db, hidden.id).id
    hidden.status = "awaiting_clarification"
    db.commit()

    detail = get_episode(episode_id, db=db)

    assert [m.unit_id for m in detail.members] == [kept_unit.id]


def test_get_episode_raises_404_for_an_episode_that_does_not_exist(db):
    with pytest.raises(HTTPException) as caught:
        get_episode(uuid.uuid4(), db=db)

    assert caught.value.status_code == 404


def test_get_episodes_for_unit_returns_each_episode_as_json_serializable_output(db):
    unit = _ready_unit(db, text="A talk I gave in Berlin.", semantic={"locations": ["Berlin"]})
    conference = _episode_row(
        db, title="the conference", started=dt.date(2024, 5, 2), precision="day"
    )
    trip = _episode_row(db, title="the Berlin trip", started=dt.date(2024, 5, 1), precision="day")
    _membership_row(db, conference, unit)
    _membership_row(db, trip, unit)

    out = get_episodes_for_unit(unit.id, db=db)

    assert [e.id for e in out] == [conference.id, trip.id]
    assert json.dumps([e.model_dump(mode="json") for e in out])


def test_get_episodes_for_unit_raises_404_for_a_memory_that_does_not_exist(db):
    with pytest.raises(HTTPException) as caught:
        get_episodes_for_unit(uuid.uuid4(), db=db)

    assert caught.value.status_code == 404
