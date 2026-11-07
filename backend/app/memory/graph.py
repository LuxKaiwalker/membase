"""Projecting memories into the associative graph, and walking it.

**Projection** (`project_unit`) reads one unit's `semantic` and
`explicit_metadata` and writes the `memory_entities` / `unit_entity_links` /
`entity_relations` / `unit_associations` rows that follow from them. It is
deterministic, LLM-free, and idempotent: it deletes everything it previously
wrote for that unit and rebuilds. That property is what makes it safe to call
from several pipeline stages (after the fast pipeline finalizes a unit, again
after the structurer deepens it, again after a user correction) without
tracking what changed.

**Traversal** (`neighbors`, `subgraph`) is plain SQL over those tables. No
graph database, no recursive CTE: hops are bounded to 2 and done as two
batched queries, which at this corpus size is both faster and far easier to
reason about than a recursive query whose cost is hard to bound.

Why a projection rather than extracting the graph directly at ingestion time:
the authoritative text and the semantic extraction are the system's real
assets. The graph is an index over them. Making it a rebuildable projection
means a bug in this file is fixable with a backfill
(`scripts/rebuild_graph.py`) instead of requiring re-ingestion -- which, at
~20-25s of model time per memory, would otherwise make every graph bug
permanent in practice.
"""

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.memory.entities import (
    FIELD_TO_KIND,
    display_name_for,
    is_alias_of,
    is_usable_entity_name,
    normalize_name,
)
from app.memory.temporal import parse_event_date
from app.models import (
    ORIGIN_EXPLICIT,
    ORIGIN_EXTRACTED,
    ORIGIN_INFERRED,
    EntityRelation,
    MemoryEntity,
    MemoryUnit,
    UnitAssociation,
    UnitEntityLink,
)

logger = logging.getLogger(__name__)

# How much one mention is worth, by the semantic field it came from. These are
# not probabilities; they are a fixed, documented ranking of how strongly each
# field indicates "this memory is really *about* that thing":
#
#   A named person/place/organization is a strong handle on a memory.
#   A relation endpoint is strong too -- it is a thing the memory asserts
#     something about.
#   A definition's term is a thing the memory explains.
#   A topic is a theme, not a handle: "hiking" is shared by every hike.
#
# Used both as the link's own `weight` and (via the association formula below)
# as part of how much two memories sharing that entity counts for.
ROLE_WEIGHTS = {
    "people": 1.0,
    "organizations": 0.9,
    "locations": 1.0,
    "relation_subject": 0.8,
    "relation_object": 0.8,
    "definition_term": 0.7,
    "visual_concepts": 0.7,
    "topics": 0.5,
}

# Per-kind multiplier when deciding how much a *shared* entity justifies an
# association between two memories. Mirrors ROLE_WEIGHTS' reasoning at the
# node level: two memories sharing a person are far more likely to be about
# the same episode than two sharing the topic "work".
KIND_ASSOCIATION_WEIGHTS = {
    "person": 1.0,
    "location": 1.0,
    "organization": 0.9,
    "concept": 0.8,
    "object": 0.7,
    "topic": 0.5,
}

ASSOC_SHARED_ENTITY = "shared_entity"
ASSOC_SAME_CAPTURE = "same_capture"
ASSOC_STRUCTURER = "structurer_link"

# Fixed weights for the two association kinds that are not computed from
# entity overlap. Both are below what a strong shared-entity edge reaches, on
# purpose: "the user typed these together" and "the structurer thought these
# were related" are real signals but weaker evidence of being about the same
# thing than "both name the same person and the same lake".
SAME_CAPTURE_WEIGHT = 0.5
STRUCTURER_LINK_WEIGHT = 0.45


def _ordered_pair(a: uuid.UUID, b: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    """Canonical ordering for a symmetric edge, so one unordered pair is one
    row and the unique constraint actually prevents duplicates."""
    return (a, b) if str(a) < str(b) else (b, a)


# --------------------------------------------------------------------------
# Entity resolution against the database
# --------------------------------------------------------------------------


def _find_entity(db: Session, kind: str, normalized: str) -> MemoryEntity | None:
    """Look up an entity by its normalized name, then by any alias folded into
    it. The alias path is what makes "ETH" and "ETH Zurich" one node when some
    memory's `aliases` field connected them."""
    existing = db.execute(
        select(MemoryEntity).where(
            MemoryEntity.kind == kind, MemoryEntity.normalized_name == normalized
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    return db.execute(
        select(MemoryEntity)
        .where(MemoryEntity.kind == kind, MemoryEntity.aliases.contains([normalized]))
        .order_by(MemoryEntity.created_at)
        .limit(1)
    ).scalar_one_or_none()


def resolve_entity(
    db: Session, kind: str, surface: str, *, aliases: list[str] | None = None
) -> MemoryEntity | None:
    """Get or create the entity for `surface`, folding in `aliases`.

    Returns None for an unusable name (pronoun, single character, bare number
    -- see app/memory/entities.py), which callers treat as "skip this mention"
    rather than as an error: a memory that mentions "someone" is still a
    perfectly good memory, it just contributes no node.

    Flushes but does not commit -- the caller owns the transaction boundary.
    """
    if not is_usable_entity_name(surface):
        return None
    normalized = normalize_name(surface)
    entity = _find_entity(db, kind, normalized)
    if entity is None:
        entity = MemoryEntity(
            kind=kind,
            normalized_name=normalized,
            display_name=display_name_for(surface),
            aliases=[],
            mention_count=0,
        )
        db.add(entity)
        db.flush()
    else:
        # Heal a display name that still carries a parenthetical gloss. These
        # exist for real -- the live model emitted entities literally named
        # "Slovenia (Implicit location of Lake Bled)" -- and `display_name` is
        # otherwise only ever written at creation, so a row stored before
        # `display_name_for` stripped glosses would keep the machine debris
        # forever and surface it in episode titles and association
        # explanations. One-way and idempotent (the cleaned form has no gloss
        # to strip), so this cannot flap between spellings the way "always
        # take the latest surface" would.
        cleaned = display_name_for(entity.display_name)
        if cleaned and cleaned != entity.display_name:
            entity.display_name = cleaned

    if aliases:
        normalized_aliases = {
            normalize_name(a)
            for a in aliases
            # `is_alias_of` is the load-bearing filter here: `semantic.aliases`
            # describes the whole MEMORY, not this entity, so only the entries
            # that are actually surface variants of *this* name may be folded
            # in. See app/memory/entities.py::is_alias_of for the real
            # corruption this prevents.
            if is_usable_entity_name(a) and is_alias_of(a, surface)
        }
        normalized_aliases.discard(normalized)
        # Never fold in a string that is already some *other* entity's primary
        # name: that would quietly merge two distinct nodes on the strength of
        # one memory's alias guess, and there is no un-merge.
        additions = sorted(
            a
            for a in normalized_aliases - set(entity.aliases)
            if _find_entity(db, kind, a) in (None, entity)
        )
        if additions:
            entity.aliases = [*entity.aliases, *additions]
    return entity


# --------------------------------------------------------------------------
# Mentions: semantic fields -> (kind, surface, role, origin)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mention:
    kind: str
    surface: str
    role: str
    origin: str
    excerpt: str = ""
    predicate: str | None = None


# Which explicit_metadata keys correspond to which semantic field. A value the
# user typed themselves gets origin="explicit", which is the whole point of
# keeping the two apart: at retrieval time a user-stated location can be
# trusted over a model-inferred one.
_EXPLICIT_METADATA_FIELDS = {
    "people": "people",
    "organizations": "organizations",
    "topics": "topics",
    "location": "locations",
}


# Role -> the explicit_metadata field name it should be checked against.
# Only these four roles have a user-supplied counterpart; every other role
# therefore never resolves as origin="explicit".
ROLE_TO_EXPLICIT_FIELD = {
    "people": "people",
    "organizations": "organizations",
    "locations": "locations",
    "topics": "topics",
}


def _explicit_values(explicit_metadata: dict) -> dict[str, set[str]]:
    """Normalized set of values the user supplied per semantic field, used to
    decide each mention's origin."""
    out: dict[str, set[str]] = defaultdict(set)
    for meta_key, field_name in _EXPLICIT_METADATA_FIELDS.items():
        raw = (explicit_metadata or {}).get(meta_key)
        values = raw if isinstance(raw, list) else [raw] if isinstance(raw, str) else []
        for value in values:
            normalized = normalize_name(value) if isinstance(value, str) else ""
            if normalized:
                out[field_name].add(normalized)
    return out


def mentions_for(semantic: dict, explicit_metadata: dict | None = None) -> list[Mention]:
    """Every entity mention a unit's semantic representation implies.

    Pure function of its inputs -- no DB, no LLM -- so the mapping from
    extraction output to graph nodes is directly testable.
    """
    semantic = semantic or {}
    explicit = _explicit_values(explicit_metadata or {})
    mentions: list[Mention] = []
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, surface: str, role: str, *, predicate: str | None = None, excerpt: str = "") -> None:
        if not is_usable_entity_name(surface):
            return
        normalized = normalize_name(surface)
        key = (kind, normalized, role)
        if key in seen:
            return
        seen.add(key)
        field_name = ROLE_TO_EXPLICIT_FIELD.get(role, role)
        origin = ORIGIN_EXPLICIT if normalized in explicit.get(field_name, ()) else ORIGIN_EXTRACTED
        mentions.append(
            Mention(
                kind=kind,
                surface=surface,
                role=role,
                origin=origin,
                excerpt=excerpt,
                predicate=predicate,
            )
        )

    # Specifically-typed fields first: a person, organization, or location is
    # a definite kind, and these establish the kind every later mention of the
    # same name resolves against.
    for field_name in ("people", "organizations", "locations"):
        for value in semantic.get(field_name) or []:
            if isinstance(value, str):
                add(FIELD_TO_KIND[field_name], value, field_name)

    # `topics` is the generic bucket, so it resolves against the specific
    # kinds above rather than establishing its own. The live model routinely
    # lists the same string in both -- "Switzerland" as a `locations` entry
    # AND a `topics` entry, "ETH Zurich" as an `organizations` entry AND a
    # `topics` entry -- which without this produced two nodes for one real
    # thing and split the association weight between memories that mention it.
    # A name that is also a named entity should BE that entity.
    def _established(value: str) -> str | None:
        return next(
            (m.kind for m in mentions if normalize_name(m.surface) == normalize_name(value)), None
        )

    for value in semantic.get("topics") or []:
        if isinstance(value, str):
            add(_established(value) or "topic", value, "topics")

    # Everything after this point is untyped: a definition term, a visual
    # concept, and a relation endpoint are all just strings. Each reuses the
    # kind a typed field already established for that name and falls back to
    # its own default otherwise. Without this, one real thing fragments into
    # several nodes by role -- the live model emitted "Lake Bled" as both a
    # `locations` entry and a `definitions` term, which produced a location
    # node and a separate concept node for the same lake, halving the
    # association weight between memories that mention it.
    kind_by_normalized: dict[str, str] = {}
    for m in mentions:
        kind_by_normalized.setdefault(normalize_name(m.surface), m.kind)

    def established_kind(value: str, fallback: str) -> str:
        return kind_by_normalized.get(normalize_name(value), fallback)

    for definition in semantic.get("definitions") or []:
        if isinstance(definition, dict) and isinstance(definition.get("term"), str):
            add(
                established_kind(definition["term"], "concept"),
                definition["term"],
                "definition_term",
                excerpt=definition.get("definition", "") if isinstance(definition.get("definition"), str) else "",
            )

    # `visual_concepts` is written by the image path (see
    # app/ingestion/pipeline.py) -- objects actually visible in a photograph,
    # which is what makes "a lake with mountains behind it" a retrievable
    # association rather than only a string inside a caption.
    for value in semantic.get("visual_concepts") or []:
        if isinstance(value, str):
            add(established_kind(value, "object"), value, "visual_concepts")

    # Relation endpoints last, so they can resolve against definitions and
    # visual concepts too, not only the typed fields.
    for m in mentions:
        kind_by_normalized.setdefault(normalize_name(m.surface), m.kind)

    for relation in semantic.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        predicate = relation.get("predicate") if isinstance(relation.get("predicate"), str) else ""
        for key, role in (("subject", "relation_subject"), ("object", "relation_object")):
            value = relation.get(key)
            if not isinstance(value, str):
                continue
            kind = kind_by_normalized.get(normalize_name(value), "concept")
            add(kind, value, role, predicate=predicate)

    return mentions


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


def remove_unit_from_graph(db: Session, unit_id: uuid.UUID) -> None:
    """Delete everything projection wrote for one unit, including associations
    where it is the *target* rather than the source, and its episode
    memberships. Does not commit."""
    # Local import: app/memory/episodes.py imports nothing from this module,
    # but keeping the dependency one-directional at module load time avoids
    # making the two mutually importable as they grow.
    from app.memory.episodes import remove_unit_from_episodes

    db.execute(delete(UnitEntityLink).where(UnitEntityLink.unit_id == unit_id))
    db.execute(delete(EntityRelation).where(EntityRelation.unit_id == unit_id))
    db.execute(
        delete(UnitAssociation).where(
            (UnitAssociation.source_unit_id == unit_id) | (UnitAssociation.target_unit_id == unit_id)
        )
    )
    remove_unit_from_episodes(db, unit_id)


def _entity_document_frequency(db: Session, entity_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    """How many distinct ready units link to each of these entities."""
    if not entity_ids:
        return {}
    rows = db.execute(
        select(UnitEntityLink.entity_id, func.count(func.distinct(UnitEntityLink.unit_id)))
        .join(MemoryUnit, MemoryUnit.id == UnitEntityLink.unit_id)
        .where(UnitEntityLink.entity_id.in_(entity_ids), MemoryUnit.status == "ready")
        .group_by(UnitEntityLink.entity_id)
    ).all()
    return {row[0]: int(row[1]) for row in rows}


def _ready_unit_count(db: Session) -> int:
    return int(
        db.execute(select(func.count(MemoryUnit.id)).where(MemoryUnit.status == "ready")).scalar() or 0
    )


def entity_specificity(document_frequency: int, total_units: int) -> float:
    """How much sharing this entity tells you, in (0, 1].

    Smoothed inverse document frequency: `log(1 + N/df) / log(1 + N)`. An
    entity in 1 memory out of 200 scores ~1.0; one in half of them ~0.2. This
    is what stops a recurring entity (your own city, your employer) from
    making every memory an associate of every other -- the most likely way an
    automatically-built graph degenerates into uselessness.

    The `1 +` smoothing is not cosmetic. Plain IDF (`log(N/df)/log(N)`) is
    exactly **zero** when an entity appears in every ready memory, and on a
    small corpus that is the normal case, not an edge case: with two stored
    memories that both mention the same person, df == N == 2, specificity is
    0, and the single most informative connection in the store produces no
    edge at all. Caught by running the real model over a four-memory corpus --
    the two memories sharing "Markus" were left unconnected while a much
    weaker pair sharing "Switzerland" got an edge, purely because the latter
    happened to be projected when a third memory existed. Smoothing keeps the
    ordering (more common is still less specific) while never collapsing a
    genuine shared entity to no signal.
    """
    if total_units <= 1 or document_frequency <= 0:
        return 1.0
    df = min(document_frequency, total_units)
    return max(0.0, min(1.0, math.log1p(total_units / df) / math.log1p(total_units)))


@dataclass
class ProjectionResult:
    # Rows actually written, not mentions considered -- several mentions can
    # fold onto one entity+role and produce a single link, so counting
    # mentions here would over-report what is in the database.
    entity_links: int = 0
    relations: int = 0
    associations: int = 0
    # Mention surfaces that were dropped as unusable (pronouns, self-reference,
    # dates, bare quantities). Useful when diagnosing "why is this memory not
    # in the graph?" -- an empty graph footprint with a long list here means
    # the extraction produced nothing nameable, which is a different problem
    # from projection failing.
    skipped_entities: list[str] = field(default_factory=list)
    # Which episode this memory ended up in, and by which clustering rule --
    # see app/memory/episodes.py. None when the unit was not ready (and so was
    # removed from the graph rather than projected into it).
    episode_id: str | None = None
    episode_rule: str | None = None
    episode_created: bool = False


def _link_mentions(
    db: Session, unit: MemoryUnit, mentions: list[Mention]
) -> tuple[dict[uuid.UUID, UnitEntityLink], int, list[str]]:
    """Create UnitEntityLink rows for these mentions, resolving entities.

    Returns `(strongest_link_per_entity, links_written, skipped_surfaces)`.
    """
    aliases = [a for a in (unit.semantic or {}).get("aliases") or [] if isinstance(a, str)]
    links: dict[uuid.UUID, UnitEntityLink] = {}
    # Dedup key is the RESOLVED entity, not the surface string. `mentions_for`
    # already deduped by normalized surface, but that is not enough: two
    # different surfaces can resolve to one entity through alias folding (a
    # memory listing both "Glaciology" and "glaciology research" as topics,
    # where one is an alias of the other), which would then violate
    # uq_unit_entity_links_unit_entity_role. Found for real against the live
    # model, not hypothesised.
    seen_entity_roles: set[tuple[uuid.UUID, str]] = set()
    written = 0
    skipped: list[str] = []
    for mention in mentions:
        # Aliases are folded in only for the typed-field mentions, not for
        # relation endpoints: an alias list belongs to what the memory is
        # about, and attaching it to an incidental relation object is how
        # unrelated entities get wrongly merged.
        entity = resolve_entity(
            db,
            mention.kind,
            mention.surface,
            aliases=aliases if mention.role in ROLE_TO_EXPLICIT_FIELD else None,
        )
        if entity is None:
            skipped.append(mention.surface)
            continue
        if (entity.id, mention.role) in seen_entity_roles:
            continue
        seen_entity_roles.add((entity.id, mention.role))
        link = UnitEntityLink(
            unit_id=unit.id,
            entity_id=entity.id,
            role=mention.role,
            origin=mention.origin,
            weight=ROLE_WEIGHTS.get(mention.role, 0.5),
            provenance={
                "field": mention.role,
                "surface": display_name_for(mention.surface),
                "excerpt": mention.excerpt[:300],
                "predicate": mention.predicate,
                "capture_id": str(unit.capture_id),
            },
        )
        db.add(link)
        # Keep the strongest link per entity for association weighting -- an
        # entity named both as a person and as a relation subject should count
        # once, at its best role.
        prior = links.get(entity.id)
        if prior is None or link.weight > prior.weight:
            links[entity.id] = link
        written += 1
    db.flush()
    return links, written, skipped


def _project_relations(db: Session, unit: MemoryUnit) -> int:
    """Turn `semantic.relations` triples into entity_relations rows."""
    semantic = unit.semantic or {}
    occurred_on, precision = parse_event_date(semantic.get("event_date"))
    created = 0
    seen: set[tuple[uuid.UUID, uuid.UUID, str]] = set()

    mention_kinds: dict[str, str] = {}
    for mention in mentions_for(semantic, unit.explicit_metadata):
        mention_kinds.setdefault(normalize_name(mention.surface), mention.kind)

    for relation in semantic.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        subject, predicate, obj = (
            relation.get("subject"), relation.get("predicate"), relation.get("object")
        )
        if not all(isinstance(v, str) and v.strip() for v in (subject, predicate, obj)):
            continue
        subject_entity = resolve_entity(db, mention_kinds.get(normalize_name(subject), "concept"), subject)
        object_entity = resolve_entity(db, mention_kinds.get(normalize_name(obj), "concept"), obj)
        # A triple whose subject or object is a pronoun ("I met Anna" ->
        # subject "I") still contributes its usable half as a plain mention
        # above; it just cannot be an *edge*, which needs two real endpoints.
        if subject_entity is None or object_entity is None or subject_entity.id == object_entity.id:
            continue
        normalized_predicate = normalize_name(predicate)
        key = (subject_entity.id, object_entity.id, normalized_predicate)
        if key in seen:
            continue
        seen.add(key)
        db.add(
            EntityRelation(
                unit_id=unit.id,
                subject_entity_id=subject_entity.id,
                object_entity_id=object_entity.id,
                predicate=predicate.strip(),
                normalized_predicate=normalized_predicate,
                origin=ORIGIN_EXTRACTED,
                weight=1.0,
                occurred_on=occurred_on,
                date_precision=precision,
                provenance={
                    "capture_id": str(unit.capture_id),
                    "event_date_raw": semantic.get("event_date") or "",
                    "excerpt": (unit.authoritative_content or "")[:300],
                },
            )
        )
        created += 1
    db.flush()
    return created


def _describe_shared(entities: list[MemoryEntity]) -> str:
    """Deterministic 'why are these connected' text."""
    if not entities:
        return ""
    parts = [f"{e.display_name} ({e.kind})" for e in entities[:4]]
    more = len(entities) - len(parts)
    joined = ", ".join(parts)
    suffix = f", and {more} more" if more > 0 else ""
    return f"Both mention {joined}{suffix}."


def _project_shared_entity_associations(
    db: Session, unit: MemoryUnit, links: dict[uuid.UUID, UnitEntityLink]
) -> int:
    """Create weighted edges to every other ready unit sharing entities.

    Weight is the summed specificity of the shared entities, each scaled by
    kind and by how strongly each side's link role indicated aboutness, then
    clamped to 1.0. Edges below the configured floor are not written at all --
    a graph where everything is weakly connected to everything is the same as
    no graph, just more expensive to query.
    """
    settings = get_settings()
    if not links:
        return 0

    entity_ids = list(links)
    total_units = _ready_unit_count(db)
    document_frequency = _entity_document_frequency(db, entity_ids)

    # Entities that appear in too large a share of the corpus are excluded
    # from association weighting entirely (their link rows are still kept as
    # real provenance -- this only stops them generating edges).
    #
    # The ceiling is a FRACTION, which is meaningless on a small corpus, and
    # the first version of this gated only on `total_units <= 2`. That was
    # wrong in a way the real-model smoke test caught immediately: with three
    # stored memories, an entity shared by two of them sits at 0.67, over the
    # 0.5 ceiling, so the single most informative entity in the corpus
    # (a person named in two memories) was excluded and NO shared-entity
    # association was created at all. The feature was effectively disabled
    # until the store grew large. Both guards below have to hold before the
    # ceiling is allowed to exclude anything; below that, the smooth
    # specificity weighting in `entity_specificity` is the only discount
    # applied, which is the right behaviour for a small store.
    ceiling = settings.membase_graph_entity_df_ceiling
    min_corpus = settings.membase_graph_entity_df_min_units
    min_df = settings.membase_graph_entity_df_min_mentions
    usable_entity_ids = [
        eid
        for eid in entity_ids
        if not (
            total_units >= min_corpus
            and document_frequency.get(eid, 0) >= min_df
            and (document_frequency.get(eid, 0) / max(total_units, 1)) > ceiling
        )
    ]
    if not usable_entity_ids:
        return 0

    rows = db.execute(
        select(UnitEntityLink.unit_id, UnitEntityLink.entity_id, UnitEntityLink.weight)
        .join(MemoryUnit, MemoryUnit.id == UnitEntityLink.unit_id)
        .where(
            UnitEntityLink.entity_id.in_(usable_entity_ids),
            UnitEntityLink.unit_id != unit.id,
            MemoryUnit.status == "ready",
        )
    ).all()

    per_unit: dict[uuid.UUID, dict[uuid.UUID, float]] = defaultdict(dict)
    for other_unit_id, entity_id, other_weight in rows:
        prior = per_unit[other_unit_id].get(entity_id, 0.0)
        per_unit[other_unit_id][entity_id] = max(prior, float(other_weight))

    entities = {
        e.id: e
        for e in db.execute(select(MemoryEntity).where(MemoryEntity.id.in_(usable_entity_ids))).scalars()
    }

    scored: list[tuple[float, uuid.UUID, list[uuid.UUID]]] = []
    for other_unit_id, shared in per_unit.items():
        total = 0.0
        for entity_id, other_weight in shared.items():
            entity = entities.get(entity_id)
            if entity is None:
                continue
            specificity = entity_specificity(document_frequency.get(entity_id, 1), total_units)
            kind_weight = KIND_ASSOCIATION_WEIGHTS.get(entity.kind, 0.5)
            own_weight = links[entity_id].weight
            total += specificity * kind_weight * min(own_weight, other_weight)
        weight = min(1.0, total)
        if weight >= settings.membase_graph_min_association_weight:
            ordered_shared = sorted(shared, key=lambda eid: -document_frequency.get(eid, 0))
            scored.append((weight, other_unit_id, ordered_shared))

    # Cap per unit, keeping the strongest -- bounds both storage and the
    # fan-out any traversal has to consider.
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    scored = scored[: settings.membase_graph_max_associations_per_unit]

    created = 0
    for weight, other_unit_id, shared_ids in scored:
        source_id, target_id = _ordered_pair(unit.id, other_unit_id)
        existing = db.execute(
            select(UnitAssociation).where(
                UnitAssociation.source_unit_id == source_id,
                UnitAssociation.target_unit_id == target_id,
                UnitAssociation.kind == ASSOC_SHARED_ENTITY,
            )
        ).scalar_one_or_none()
        shared_entities = [entities[eid] for eid in shared_ids if eid in entities]
        values = {
            "origin": ORIGIN_INFERRED,
            "weight": weight,
            "shared_entity_ids": [str(eid) for eid in shared_ids],
            "explanation": _describe_shared(shared_entities),
        }
        if existing is not None:
            # Re-projecting the other side of an existing pair: refresh the
            # weight rather than skipping, since document frequencies (and so
            # specificity) move as the corpus grows.
            for key, value in values.items():
                setattr(existing, key, value)
        else:
            db.add(
                UnitAssociation(
                    source_unit_id=source_id, target_unit_id=target_id, kind=ASSOC_SHARED_ENTITY, **values
                )
            )
            created += 1
    db.flush()
    return created


def _project_structural_associations(db: Session, unit: MemoryUnit) -> int:
    """Sibling (same-capture) and structurer-judged edges.

    These need no weighting: the first is a fact about how the user submitted
    the memory, the second is a judgement the structurer already made and
    stored in `related_unit_ids` (which this reads and does not modify).
    """
    created = 0

    sibling_ids = db.execute(
        select(MemoryUnit.id).where(
            MemoryUnit.capture_id == unit.capture_id,
            MemoryUnit.id != unit.id,
            MemoryUnit.status == "ready",
        )
    ).scalars().all()
    for sibling_id in sibling_ids:
        source_id, target_id = _ordered_pair(unit.id, sibling_id)
        if _association_exists(db, source_id, target_id, ASSOC_SAME_CAPTURE):
            continue
        db.add(
            UnitAssociation(
                source_unit_id=source_id,
                target_unit_id=target_id,
                kind=ASSOC_SAME_CAPTURE,
                # "extracted", not "inferred": the user genuinely did write
                # these in one submission. Nothing was guessed.
                origin=ORIGIN_EXTRACTED,
                weight=SAME_CAPTURE_WEIGHT,
                shared_entity_ids=[],
                explanation="Written in the same note.",
            )
        )
        created += 1

    for raw_id in unit.related_unit_ids or []:
        try:
            related_id = uuid.UUID(str(raw_id))
        except (ValueError, TypeError):
            continue
        if related_id == unit.id:
            continue
        related = db.get(MemoryUnit, related_id)
        if related is None or related.status != "ready":
            continue
        source_id, target_id = _ordered_pair(unit.id, related_id)
        if _association_exists(db, source_id, target_id, ASSOC_STRUCTURER):
            continue
        db.add(
            UnitAssociation(
                source_unit_id=source_id,
                target_unit_id=target_id,
                kind=ASSOC_STRUCTURER,
                origin=ORIGIN_INFERRED,
                weight=STRUCTURER_LINK_WEIGHT,
                shared_entity_ids=[],
                explanation="The deep structuring pass judged these related.",
            )
        )
        created += 1

    db.flush()
    return created


def _association_exists(db: Session, source_id: uuid.UUID, target_id: uuid.UUID, kind: str) -> bool:
    return (
        db.execute(
            select(UnitAssociation.id).where(
                UnitAssociation.source_unit_id == source_id,
                UnitAssociation.target_unit_id == target_id,
                UnitAssociation.kind == kind,
            )
        ).first()
        is not None
    )


def _refresh_mention_counts(db: Session, entity_ids: list[uuid.UUID]) -> None:
    if not entity_ids:
        return
    counts = _entity_document_frequency(db, entity_ids)
    for entity in db.execute(select(MemoryEntity).where(MemoryEntity.id.in_(entity_ids))).scalars():
        entity.mention_count = counts.get(entity.id, 0)


def project_unit(db: Session, unit: MemoryUnit) -> ProjectionResult:
    """Rebuild one unit's entire graph footprint from its current semantics.

    Idempotent (delete-then-rebuild). Does **not** commit: the caller decides
    the transaction boundary, which matters because projection runs inside
    pipeline steps that have their own lease-gated commit discipline.

    A unit that is not `ready` is removed from the graph and nothing is
    rebuilt -- `awaiting_clarification` content is deliberately invisible to
    retrieval (see docs/ARCHITECTURE.md "Status gate"), and a graph node for
    it would be a backdoor around that gate.
    """
    remove_unit_from_graph(db, unit.id)
    if unit.status != "ready":
        return ProjectionResult()

    mentions = mentions_for(unit.semantic, unit.explicit_metadata)
    links, links_written, skipped = _link_mentions(db, unit, mentions)
    relations = _project_relations(db, unit)
    associations = _project_shared_entity_associations(db, unit, links)
    associations += _project_structural_associations(db, unit)
    _refresh_mention_counts(db, list(links))

    # Episode assignment runs last, inside the same projection, because it
    # depends on the entity links written above -- a memory's anchor entities
    # are what let it join an existing episode. Bundling it here also means
    # every caller of projection (the pipeline's four commit points and
    # scripts/rebuild_graph.py) maintains episodes for free, with no second
    # backfill path to keep in sync.
    from app.memory.episodes import assign_episode

    episode = assign_episode(db, unit)

    return ProjectionResult(
        entity_links=links_written,
        relations=relations,
        associations=associations,
        skipped_entities=skipped,
        episode_id=str(episode.episode_id) if episode else None,
        episode_rule=episode.rule if episode else None,
        episode_created=bool(episode and episode.created),
    )


def project_unit_safely(db: Session, unit: MemoryUnit) -> ProjectionResult:
    """`project_unit` with a broad guard, for the pipeline call sites.

    The graph is an index, not a memory: a bug here must never be able to fail
    the ingestion step that already durably stored the user's words.

    Projection runs inside a SAVEPOINT, which is what makes that guarantee
    real rather than merely intended. A plain try/except is not enough: a
    constraint violation raised during a flush leaves the whole session in a
    pending-rollback state, so the *caller's* next statement fails too --
    observed for real, where a duplicate-link bug here propagated out of
    `process_capture_fully` and marked an otherwise perfectly-ingested capture
    as `processing_status="failed"`. A savepoint confines the damage to the
    projection, leaving the surrounding transaction usable.

    Every call site commits the memory itself *before* calling this, so a
    rolled-back projection never costs anything the user wrote. The unit stays
    fully stored and fully searchable through every existing (non-graph)
    retrieval path; it just has no graph rows until the next projection or a
    `scripts/rebuild_graph.py` run.
    """
    try:
        with db.begin_nested():
            result = project_unit(db, unit)
        db.commit()
        return result
    except Exception:
        logger.exception("Graph projection failed for unit %s; memory itself is unaffected", unit.id)
        try:
            db.rollback()
        except Exception:
            logger.exception("Rollback after a failed graph projection also failed")
        return ProjectionResult()


# --------------------------------------------------------------------------
# Traversal
# --------------------------------------------------------------------------
#
# Bounded to MAX_HOPS as two batched queries rather than a recursive CTE. At
# this corpus size that is both cheaper and much easier to bound: each hop is
# one `WHERE unit_id IN (...)`, and the fan-out cap
# (`membase_graph_max_associations_per_unit`) makes the worst case a product
# of two known constants instead of a query whose cost depends on graph shape.

MAX_HOPS = 2

# Multiplier applied per hop, so a memory two hops away carries visibly less
# weight than a direct associate. Not calibrated against an eval set -- an
# engineering default, flagged in docs/TODO.md; the benchmark in eval/ is what
# would justify changing it.
HOP_DECAY = 0.55


@dataclass
class AssociationStep:
    """One edge traversed, kept so a retrieval result can explain its own
    provenance all the way back to a directly-matched memory."""

    from_unit_id: uuid.UUID
    to_unit_id: uuid.UUID
    kind: str
    origin: str
    weight: float
    explanation: str


@dataclass
class AssociatedUnit:
    unit: MemoryUnit
    # Strength of the best path found, after per-hop decay, in (0, 1].
    strength: float
    hops: int
    # The actual path, nearest-seed first.
    path: list[AssociationStep] = field(default_factory=list)

    @property
    def why(self) -> str:
        """Deterministic human-readable justification for the whole path."""
        return " -> ".join(step.explanation for step in self.path if step.explanation)


def _edges_from(
    db: Session, unit_ids: list[uuid.UUID], *, kinds: tuple[str, ...] | None = None
) -> list[tuple[uuid.UUID, uuid.UUID, str, str, float, str]]:
    """Every association touching any of `unit_ids`, normalized so the first
    element is always the queried side. Associations are stored once per
    unordered pair (see `_ordered_pair`), so both columns must be checked."""
    if not unit_ids:
        return []
    query = select(
        UnitAssociation.source_unit_id,
        UnitAssociation.target_unit_id,
        UnitAssociation.kind,
        UnitAssociation.origin,
        UnitAssociation.weight,
        UnitAssociation.explanation,
    ).where(
        UnitAssociation.source_unit_id.in_(unit_ids) | UnitAssociation.target_unit_id.in_(unit_ids)
    )
    if kinds:
        query = query.where(UnitAssociation.kind.in_(kinds))

    wanted = set(unit_ids)
    out = []
    for source_id, target_id, kind, origin, weight, explanation in db.execute(query).all():
        if source_id in wanted:
            out.append((source_id, target_id, kind, origin, float(weight), explanation))
        if target_id in wanted:
            out.append((target_id, source_id, kind, origin, float(weight), explanation))
    return out


def neighbors(
    db: Session,
    seed_scores: dict[uuid.UUID, float],
    *,
    max_hops: int = MAX_HOPS,
    limit: int = 20,
    min_strength: float = 0.05,
    kinds: tuple[str, ...] | None = None,
) -> list[AssociatedUnit]:
    """Walk outward from scored seed units and return the ready units reached.

    `seed_scores` maps a directly-retrieved unit id to its retrieval score;
    a neighbour's strength is `seed_score * edge_weight * HOP_DECAY**hop`, and
    the best path wins when several lead to the same unit. Seeds themselves
    are never returned -- this answers "what else is connected to what I
    already found", not "what did I find".

    Only `status='ready'` units come back, which keeps the status gate
    (docs/ARCHITECTURE.md) intact through the graph as well as through
    `hybrid_search`.
    """
    if not seed_scores or max_hops < 1:
        return []

    # unit id -> (strength, hops, path) for the best path found so far
    best: dict[uuid.UUID, tuple[float, int, list[AssociationStep]]] = {}
    # strength and path of the frontier, keyed by unit id
    frontier: dict[uuid.UUID, tuple[float, list[AssociationStep]]] = {
        uid: (max(0.0, min(1.0, score)), []) for uid, score in seed_scores.items()
    }
    visited: set[uuid.UUID] = set(seed_scores)

    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        edges = _edges_from(db, list(frontier), kinds=kinds)
        next_frontier: dict[uuid.UUID, tuple[float, list[AssociationStep]]] = {}
        for from_id, to_id, kind, origin, weight, explanation in edges:
            if to_id in seed_scores:
                continue
            base_strength, base_path = frontier[from_id]
            strength = base_strength * weight * (HOP_DECAY ** (hop - 1))
            if strength < min_strength:
                continue
            step = AssociationStep(
                from_unit_id=from_id,
                to_unit_id=to_id,
                kind=kind,
                origin=origin,
                weight=weight,
                explanation=explanation,
            )
            path = [*base_path, step]
            prior = best.get(to_id)
            if prior is None or strength > prior[0]:
                best[to_id] = (strength, hop, path)
            prior_next = next_frontier.get(to_id)
            if to_id not in visited and (prior_next is None or strength > prior_next[0]):
                next_frontier[to_id] = (strength, path)
        visited |= set(next_frontier)
        frontier = next_frontier

    if not best:
        return []

    units = {
        u.id: u
        for u in db.execute(
            select(MemoryUnit).where(MemoryUnit.id.in_(list(best)), MemoryUnit.status == "ready")
        ).scalars()
    }
    resolved = [
        AssociatedUnit(unit=units[unit_id], strength=strength, hops=hops, path=path)
        for unit_id, (strength, hops, path) in best.items()
        if unit_id in units
    ]
    # Deterministic tie-break on id: two equally-weighted associates must come
    # back in the same order across identical calls, or the evidence block an
    # answer is generated from varies run to run.
    resolved.sort(key=lambda a: (-a.strength, str(a.unit.id)))
    return resolved[:limit]


def entity_bridges(
    db: Session, unit_id_a: uuid.UUID, unit_id_b: uuid.UUID
) -> list[MemoryEntity]:
    """Entities both units mention -- the concrete reason a `shared_entity`
    edge exists, recomputed live so the UI can show it for any pair, not only
    for pairs that happen to have a stored edge."""
    rows = db.execute(
        select(UnitEntityLink.entity_id)
        .where(UnitEntityLink.unit_id == unit_id_a)
        .intersect(select(UnitEntityLink.entity_id).where(UnitEntityLink.unit_id == unit_id_b))
    ).scalars().all()
    if not rows:
        return []
    return list(
        db.execute(
            select(MemoryEntity)
            .where(MemoryEntity.id.in_(rows))
            .order_by(MemoryEntity.mention_count, MemoryEntity.display_name)
        ).scalars()
    )


# --------------------------------------------------------------------------
# Snapshot for the explorer UI
# --------------------------------------------------------------------------


@dataclass
class GraphNode:
    id: str
    # "unit" | "entity" -- the explorer draws a bipartite graph: memories and
    # the things they are about. Memory-to-memory association edges are
    # included too, so a user can follow either kind of connection.
    node_type: str
    label: str
    kind: str
    detail: str = ""
    degree: int = 0
    occurred_on: str | None = None
    date_precision: str = "unknown"


@dataclass
class GraphEdge:
    source: str
    target: str
    # "mentions" | "association" | "relation"
    edge_type: str
    kind: str
    origin: str
    weight: float
    label: str = ""
    explanation: str = ""


@dataclass
class GraphSnapshot:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False
    total_units: int = 0
    total_entities: int = 0


def _unit_label(unit: MemoryUnit) -> str:
    content = (unit.authoritative_content or "").strip()
    return content if len(content) <= 90 else content[:87].rstrip() + "..."


def subgraph(
    db: Session,
    *,
    center_unit_id: uuid.UUID | None = None,
    entity_kinds: tuple[str, ...] | None = None,
    association_kinds: tuple[str, ...] | None = None,
    hops: int = 1,
    max_nodes: int = 250,
) -> GraphSnapshot:
    """Assemble a renderable node/edge view.

    With `center_unit_id`, returns that memory's neighbourhood out to `hops`.
    Without one, returns the whole graph capped at `max_nodes` memories, taking
    the most-connected ones first -- so the default explorer view opens on the
    dense, interesting part of the graph rather than on whichever memories
    happen to be newest.

    `truncated` is set whenever the cap actually dropped something, so the UI
    can say so instead of silently presenting a partial graph as complete.
    """
    total_units = _ready_unit_count(db)
    total_entities = int(db.execute(select(func.count(MemoryEntity.id))).scalar() or 0)

    if center_unit_id is not None:
        center = db.get(MemoryUnit, center_unit_id)
        if center is None or center.status != "ready":
            return GraphSnapshot(nodes=[], edges=[], total_units=total_units, total_entities=total_entities)
        associated = neighbors(
            db,
            {center_unit_id: 1.0},
            max_hops=max(1, min(hops, MAX_HOPS)),
            limit=max_nodes,
            kinds=association_kinds,
        )
        unit_ids = [center_unit_id, *(a.unit.id for a in associated)]
        truncated = len(associated) >= max_nodes
    else:
        degree_rows = db.execute(
            select(UnitEntityLink.unit_id, func.count(UnitEntityLink.id))
            .join(MemoryUnit, MemoryUnit.id == UnitEntityLink.unit_id)
            .where(MemoryUnit.status == "ready")
            .group_by(UnitEntityLink.unit_id)
            .order_by(func.count(UnitEntityLink.id).desc(), UnitEntityLink.unit_id)
            .limit(max_nodes)
        ).all()
        unit_ids = [row[0] for row in degree_rows]
        if not unit_ids:
            # No projection has run yet (or no memory has any usable entity):
            # fall back to the newest ready memories so the explorer shows the
            # store rather than an empty canvas that looks like a bug.
            unit_ids = list(
                db.execute(
                    select(MemoryUnit.id)
                    .where(MemoryUnit.status == "ready")
                    .order_by(MemoryUnit.created_at.desc())
                    .limit(max_nodes)
                ).scalars()
            )
        # `truncated` must mean "the cap dropped something", not merely "some
        # ready unit is absent": a unit with no entity links at all is legitimately
        # not in a graph view, and reporting that as truncation would make the
        # UI warn about incompleteness on every corpus containing one
        # unconnected memory.
        truncated = len(unit_ids) >= max_nodes and total_units > len(unit_ids)

    if not unit_ids:
        return GraphSnapshot(nodes=[], edges=[], total_units=total_units, total_entities=total_entities)

    units = {
        u.id: u
        for u in db.execute(select(MemoryUnit).where(MemoryUnit.id.in_(unit_ids))).scalars()
    }

    link_query = select(UnitEntityLink).where(UnitEntityLink.unit_id.in_(unit_ids))
    links = list(db.execute(link_query).scalars())
    entity_ids = {link.entity_id for link in links}
    entities = {
        e.id: e
        for e in db.execute(select(MemoryEntity).where(MemoryEntity.id.in_(entity_ids))).scalars()
    }
    if entity_kinds:
        entities = {eid: e for eid, e in entities.items() if e.kind in entity_kinds}
        links = [link for link in links if link.entity_id in entities]

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    entity_degree: dict[uuid.UUID, int] = defaultdict(int)
    unit_degree: dict[uuid.UUID, int] = defaultdict(int)

    for link in links:
        entity = entities.get(link.entity_id)
        if entity is None:
            continue
        entity_degree[entity.id] += 1
        unit_degree[link.unit_id] += 1
        edges.append(
            GraphEdge(
                source=f"unit:{link.unit_id}",
                target=f"entity:{entity.id}",
                edge_type="mentions",
                kind=entity.kind,
                origin=link.origin,
                weight=link.weight,
                label=link.role,
                explanation=(
                    f"This memory names {entity.display_name} "
                    f"({'stated by you' if link.origin == ORIGIN_EXPLICIT else 'extracted from your words'})."
                ),
            )
        )

    for source_id, target_id, kind, origin, weight, explanation in _edges_from(
        db, unit_ids, kinds=association_kinds
    ):
        # _edges_from normalizes both directions; keep one row per pair.
        if str(source_id) > str(target_id) or target_id not in units:
            continue
        unit_degree[source_id] += 1
        unit_degree[target_id] += 1
        edges.append(
            GraphEdge(
                source=f"unit:{source_id}",
                target=f"unit:{target_id}",
                edge_type="association",
                kind=kind,
                origin=origin,
                weight=weight,
                label=kind.replace("_", " "),
                explanation=explanation,
            )
        )

    for relation in db.execute(
        select(EntityRelation).where(EntityRelation.unit_id.in_(unit_ids))
    ).scalars():
        if relation.subject_entity_id not in entities or relation.object_entity_id not in entities:
            continue
        edges.append(
            GraphEdge(
                source=f"entity:{relation.subject_entity_id}",
                target=f"entity:{relation.object_entity_id}",
                edge_type="relation",
                kind="relation",
                origin=relation.origin,
                weight=relation.weight,
                label=relation.predicate,
                explanation=(
                    f"'{entities[relation.subject_entity_id].display_name} {relation.predicate} "
                    f"{entities[relation.object_entity_id].display_name}' comes from one stored memory."
                ),
            )
        )

    for unit_id, unit in units.items():
        occurred_on, precision = parse_event_date((unit.semantic or {}).get("event_date"))
        nodes.append(
            GraphNode(
                id=f"unit:{unit_id}",
                node_type="unit",
                label=_unit_label(unit),
                kind="memory",
                detail=(unit.semantic or {}).get("meaning", "") or "",
                degree=unit_degree.get(unit_id, 0),
                occurred_on=occurred_on.isoformat() if occurred_on else None,
                date_precision=precision,
            )
        )
    for entity_id, entity in entities.items():
        nodes.append(
            GraphNode(
                id=f"entity:{entity_id}",
                node_type="entity",
                label=entity.display_name,
                kind=entity.kind,
                detail=", ".join(entity.aliases[:3]),
                degree=entity_degree.get(entity_id, 0),
            )
        )

    return GraphSnapshot(
        nodes=nodes,
        edges=edges,
        truncated=truncated,
        total_units=total_units,
        total_entities=total_entities,
    )
