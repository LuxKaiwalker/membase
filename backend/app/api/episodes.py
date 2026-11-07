"""Read-only HTTP surface over episodic memory.

Like the graph API, every route here is a GET and there is no way to create,
rename, or re-assign an episode. Episodes are derived from member dates and
entity overlap (app/memory/episodes.py); a hand-edited one would be the single
row a re-derivation destroys. The proposal's "the timeline should be generated
from available information, not require users to manually construct it" is
enforced by there being no construction API.

Two things every response here is careful about:

- **A date is never rendered more precisely than it is known.** Each payload
  carries both the raw ISO date and a `date_display` string produced by
  `app/memory/temporal.py::describe`, so a year-precision episode cannot be
  shown as a specific day by a client that forgot to check `date_precision`.
- **Undated episodes are present, counted, and labelled**, not hidden and not
  given an invented position. "I don't know when this was" is a real state of
  a memory and the timeline says so.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.memory import episodes as episodes_module
from app.memory.temporal import describe
from app.models import Capture, Episode, EpisodeMembership, MemoryUnit
from app.schemas import (
    AttachmentOut,
    EpisodeDetailOut,
    EpisodeMemberOut,
    EpisodeOut,
    TimelineOut,
)

router = APIRouter(prefix="/api/episodes", tags=["episodes"])


def _episode_out(db: Session, episode: Episode) -> EpisodeOut:
    member_count = int(
        db.execute(
            select(func.count(EpisodeMembership.id)).where(
                EpisodeMembership.episode_id == episode.id
            )
        ).scalar()
        or 0
    )
    anchors = [name for name, _count in episodes_module.anchor_counts(db, episode.id).most_common(6)]
    return EpisodeOut(
        id=episode.id,
        title=episode.title,
        title_source=episode.title_source,
        summary=episode.summary,
        started_on=episode.started_on.isoformat() if episode.started_on else None,
        ended_on=episode.ended_on.isoformat() if episode.ended_on else None,
        date_precision=episode.date_precision,
        boundary_confidence=episode.boundary_confidence,
        origin=episode.origin,
        date_display=_span_display(episode),
        member_count=member_count,
        anchors=anchors,
    )


def _span_display(episode: Episode) -> str:
    if episode.started_on is None:
        return "date unknown"
    start = describe(episode.started_on, episode.date_precision)
    if episode.ended_on and episode.ended_on != episode.started_on:
        end = describe(episode.ended_on, episode.date_precision)
        if end != start:
            return f"{start} – {end}"
    return start


@router.get("", response_model=TimelineOut)
def get_timeline(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_undated: bool = Query(True),
):
    """Episodes newest-first, undated ones last.

    This is the timeline. Ordering and the undated-last rule live in
    `app/memory/episodes.py::timeline` so the API and any other consumer
    cannot disagree about chronology.
    """
    episodes, total = episodes_module.timeline(
        db, limit=limit, offset=offset, include_undated=include_undated
    )
    undated_count = int(
        db.execute(select(func.count(Episode.id)).where(Episode.started_on.is_(None))).scalar() or 0
    )
    return TimelineOut(
        episodes=[_episode_out(db, e) for e in episodes],
        total=total,
        undated_count=undated_count,
    )


@router.get("/{episode_id}", response_model=EpisodeDetailOut)
def get_episode(episode_id: uuid.UUID, db: Session = Depends(get_db)):
    episode = db.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    rows = db.execute(
        select(EpisodeMembership, MemoryUnit)
        .join(MemoryUnit, MemoryUnit.id == EpisodeMembership.unit_id)
        .options(selectinload(MemoryUnit.capture).selectinload(Capture.attachments))
        .where(EpisodeMembership.episode_id == episode_id, MemoryUnit.status == "ready")
    ).all()

    members = []
    for membership, unit in rows:
        # Same date rule the episode's span is computed from (see
        # app/memory/episodes.py::unit_date), so a date the user typed
        # themselves is shown rather than reported as unknown next to a span
        # derived from it.
        occurred_on, precision, _user_stated = episodes_module.unit_date(unit)
        members.append(
            EpisodeMemberOut(
                unit_id=unit.id,
                capture_id=unit.capture_id,
                content=unit.authoritative_content,
                meaning=(unit.semantic or {}).get("meaning", "") or "",
                occurred_on=occurred_on.isoformat() if occurred_on else None,
                date_precision=precision,
                date_display=describe(occurred_on, precision),
                origin=membership.origin,
                confidence=membership.confidence,
                explanation=(membership.provenance or {}).get("explanation", ""),
                attachments=[AttachmentOut.model_validate(a) for a in unit.capture.attachments],
            )
        )
    # Dated members first and in order; undated last. Sorting here rather than
    # in SQL because the date lives inside a JSONB field and has to be parsed
    # to be ordered meaningfully.
    members.sort(key=lambda m: (m.occurred_on is None, m.occurred_on or ""))

    overlapping = episodes_module.overlapping_episodes(db, episode)
    return EpisodeDetailOut(
        episode=_episode_out(db, episode),
        members=members,
        overlapping=[_episode_out(db, e) for e in overlapping],
    )


@router.get("/for-unit/{unit_id}", response_model=list[EpisodeOut])
def get_episodes_for_unit(unit_id: uuid.UUID, db: Session = Depends(get_db)):
    """Which episodes a given memory belongs to.

    Returns a list, not a single episode: a memory can legitimately belong to
    several overlapping contexts (a talk belongs to both "the conference" and
    "the Berlin trip"), and collapsing that to one would discard a real
    relationship.
    """
    unit = db.get(MemoryUnit, unit_id)
    if unit is None:
        raise HTTPException(status_code=404, detail="Memory unit not found")
    return [
        _episode_out(db, episode) for episode, _membership in episodes_module.episodes_for_unit(db, unit_id)
    ]
