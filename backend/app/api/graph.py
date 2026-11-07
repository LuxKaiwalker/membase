"""Read-only HTTP surface over the associative memory graph.

Every route here is a GET. There is deliberately no endpoint to create, edit,
or delete an edge: the graph is a projection of what ingestion extracted (see
app/memory/graph.py), and a hand-edited edge would be the one row in it that a
re-projection silently destroys. The proposal's "do not turn this into a
manually maintained knowledge graph" is enforced by there being no write API,
not by convention.

Mutating a memory (`POST /api/memory-units/{id}/correct`) is how a user
changes the graph, and the projection follows automatically.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.memory import graph as memory_graph
from app.models import (
    EntityRelation,
    MemoryEntity,
    MemoryUnit,
    UnitAssociation,
    UnitEntityLink,
)
from app.schemas import (
    AssociatedUnitOut,
    AssociationStepOut,
    EntityDetailOut,
    EntityMentionOut,
    EntityOut,
    EntityRelationOut,
    GraphEdgeOut,
    GraphNodeOut,
    GraphOut,
    GraphStatsOut,
)

router = APIRouter(prefix="/api/graph", tags=["graph"])

# Hard ceiling on how much graph one request can ask for. The explorer renders
# in a canvas on the client; past a few hundred nodes a force layout stops
# being readable before it stops being fast, so this is a usability bound as
# much as a cost one.
MAX_NODES_CEILING = 600
DEFAULT_MAX_NODES = 250


def _csv_param(raw: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated filter list. An empty/absent value means "no
    filter", which is distinct from "filter to nothing"."""
    if not raw:
        return None
    values = tuple(v.strip() for v in raw.split(",") if v.strip())
    return values or None


@router.get("", response_model=GraphOut)
def get_graph(
    db: Session = Depends(get_db),
    center_unit_id: uuid.UUID | None = Query(
        None, description="Centre the view on one memory's neighbourhood instead of the whole graph"
    ),
    entity_kinds: str | None = Query(
        None, description="Comma-separated entity kinds to include (person, location, ...)"
    ),
    association_kinds: str | None = Query(
        None, description="Comma-separated association kinds (shared_entity, same_capture, structurer_link)"
    ),
    hops: int = Query(1, ge=1, le=memory_graph.MAX_HOPS),
    max_nodes: int = Query(DEFAULT_MAX_NODES, ge=1, le=MAX_NODES_CEILING),
):
    snapshot = memory_graph.subgraph(
        db,
        center_unit_id=center_unit_id,
        entity_kinds=_csv_param(entity_kinds),
        association_kinds=_csv_param(association_kinds),
        hops=hops,
        max_nodes=max_nodes,
    )
    return GraphOut(
        nodes=[GraphNodeOut(**vars(n)) for n in snapshot.nodes],
        edges=[GraphEdgeOut(**vars(e)) for e in snapshot.edges],
        truncated=snapshot.truncated,
        total_units=snapshot.total_units,
        total_entities=snapshot.total_entities,
    )


@router.get("/stats", response_model=GraphStatsOut)
def get_stats(db: Session = Depends(get_db), top: int = Query(12, ge=1, le=50)):
    """Counts for the overview page, plus the most-mentioned entities.

    `mention_count` is read off `memory_entities` rather than aggregated here:
    projection maintains it precisely so this page is a cheap read (see
    app/memory/graph.py::_refresh_mention_counts).
    """
    ready_units = int(
        db.execute(select(func.count(MemoryUnit.id)).where(MemoryUnit.status == "ready")).scalar() or 0
    )
    entities = int(db.execute(select(func.count(MemoryEntity.id))).scalar() or 0)
    entity_links = int(db.execute(select(func.count(UnitEntityLink.id))).scalar() or 0)
    relations = int(db.execute(select(func.count(EntityRelation.id))).scalar() or 0)
    associations = int(db.execute(select(func.count(UnitAssociation.id))).scalar() or 0)

    entities_by_kind = {
        row[0]: int(row[1])
        for row in db.execute(
            select(MemoryEntity.kind, func.count(MemoryEntity.id)).group_by(MemoryEntity.kind)
        ).all()
    }
    associations_by_kind = {
        row[0]: int(row[1])
        for row in db.execute(
            select(UnitAssociation.kind, func.count(UnitAssociation.id)).group_by(UnitAssociation.kind)
        ).all()
    }
    top_entities = list(
        db.execute(
            select(MemoryEntity)
            .where(MemoryEntity.mention_count > 0)
            .order_by(MemoryEntity.mention_count.desc(), MemoryEntity.display_name)
            .limit(top)
        ).scalars()
    )
    return GraphStatsOut(
        ready_units=ready_units,
        entities=entities,
        entity_links=entity_links,
        relations=relations,
        associations=associations,
        entities_by_kind=entities_by_kind,
        associations_by_kind=associations_by_kind,
        top_entities=[EntityOut.model_validate(e) for e in top_entities],
    )


@router.get("/entities", response_model=list[EntityOut])
def list_entities(
    db: Session = Depends(get_db),
    q: str | None = Query(None, description="Case-insensitive prefix/substring match on the name"),
    kind: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    query = select(MemoryEntity)
    if kind:
        query = query.where(MemoryEntity.kind == kind)
    if q and q.strip():
        # Match against the normalized name so a query typed with different
        # capitalisation or a possessive still finds the entity -- the same
        # normalization entity resolution itself uses.
        from app.memory.entities import normalize_name

        needle = normalize_name(q)
        if needle:
            query = query.where(MemoryEntity.normalized_name.ilike(f"%{needle}%"))
    entities = list(
        db.execute(
            query.order_by(MemoryEntity.mention_count.desc(), MemoryEntity.display_name).limit(limit)
        ).scalars()
    )
    return [EntityOut.model_validate(e) for e in entities]


@router.get("/entities/{entity_id}", response_model=EntityDetailOut)
def get_entity(entity_id: uuid.UUID, db: Session = Depends(get_db)):
    entity = db.get(MemoryEntity, entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    mention_rows = db.execute(
        select(UnitEntityLink, MemoryUnit)
        .join(MemoryUnit, MemoryUnit.id == UnitEntityLink.unit_id)
        .where(UnitEntityLink.entity_id == entity_id, MemoryUnit.status == "ready")
        .order_by(MemoryUnit.created_at.desc())
    ).all()
    mentions = [
        EntityMentionOut(
            unit_id=unit.id,
            capture_id=unit.capture_id,
            content=unit.authoritative_content,
            role=link.role,
            origin=link.origin,
            weight=link.weight,
            provenance=link.provenance or {},
            created_at=unit.created_at,
        )
        for link, unit in mention_rows
    ]

    relation_rows = db.execute(
        select(EntityRelation).where(
            (EntityRelation.subject_entity_id == entity_id)
            | (EntityRelation.object_entity_id == entity_id)
        )
    ).scalars()
    relations = []
    name_cache: dict[uuid.UUID, str] = {entity.id: entity.display_name}

    def _name(eid: uuid.UUID) -> str:
        if eid not in name_cache:
            other = db.get(MemoryEntity, eid)
            name_cache[eid] = other.display_name if other else "(removed)"
        return name_cache[eid]

    for relation in relation_rows:
        relations.append(
            EntityRelationOut(
                id=relation.id,
                unit_id=relation.unit_id,
                subject=_name(relation.subject_entity_id),
                predicate=relation.predicate,
                object=_name(relation.object_entity_id),
                origin=relation.origin,
                weight=relation.weight,
                occurred_on=relation.occurred_on.isoformat() if relation.occurred_on else None,
                date_precision=relation.date_precision,
            )
        )

    return EntityDetailOut(
        entity=EntityOut.model_validate(entity), mentions=mentions, relations=relations
    )


@router.get("/units/{unit_id}/associations", response_model=list[AssociatedUnitOut])
def get_unit_associations(
    unit_id: uuid.UUID,
    db: Session = Depends(get_db),
    hops: int = Query(1, ge=1, le=memory_graph.MAX_HOPS),
    limit: int = Query(20, ge=1, le=100),
):
    """What else is connected to this memory, and why.

    This is the endpoint behind "navigate from a retrieved answer to its
    underlying memory network": the `why` field on each result is a
    deterministic explanation of the path, not a generated one.
    """
    unit = db.get(MemoryUnit, unit_id)
    if unit is None:
        raise HTTPException(status_code=404, detail="Memory unit not found")
    if unit.status != "ready":
        # Not a 404: the memory exists, it is just not part of the searchable
        # graph yet (awaiting clarification) or any more (discarded). An empty
        # list is the honest answer.
        return []

    associated = memory_graph.neighbors(db, {unit_id: 1.0}, max_hops=hops, limit=limit)
    return [
        AssociatedUnitOut(
            unit=a.unit,
            strength=a.strength,
            hops=a.hops,
            path=[AssociationStepOut(**vars(step)) for step in a.path],
            why=a.why,
        )
        for a in associated
    ]
