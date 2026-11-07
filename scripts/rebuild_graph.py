#!/usr/bin/env python3
"""Backfill/maintenance pass that re-projects stored memories into the
associative memory graph (`app/memory/graph.py`).

The graph -- `memory_entities`, `unit_entity_links`, `entity_relations`,
`unit_associations` -- is a *rebuildable projection* over each unit's
`semantic`/`explicit_metadata`, not primary data. Two consequences, both of
which this script exists to deliver:

  - Memories stored before the graph existed have no graph rows at all. They
    are perfectly good memories; they are simply absent from every
    graph-backed path until something projects them.
  - A bug fixed in the projection logic has to be applicable *without*
    re-running ingestion. Re-ingesting costs ~20-25s of real model time per
    memory (see CLAUDE.md), so without a backfill every graph bug would be
    permanent in practice. This script is what makes "the graph is just an
    index, we can always rebuild it" actually true.

No LLM calls. Projection is deterministic and LLM-free (entity resolution is
normalized-name matching, association weights are inverse-document-frequency
arithmetic), so this script never even constructs a provider -- it can run
while the GPU is busy serving live traffic.

Idempotent and safe to rerun: `project_unit` deletes everything it previously
wrote for a unit and rebuilds from current semantics, so a second run over an
already-projected corpus produces the same rows and the same totals.

Usage (from the repo root, inside `nix develop`):
    python3 scripts/rebuild_graph.py --dry-run       # report what would be projected
    python3 scripts/rebuild_graph.py                 # re-project every ready unit
    python3 scripts/rebuild_graph.py --unit-id UUID  # re-project exactly one unit
    python3 scripts/rebuild_graph.py --prune         # also drop rows of no-longer-ready units
    python3 scripts/rebuild_graph.py --batch-size 10 # commit every N units (default 50)
"""

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import func, or_, select  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import get_sessionmaker  # noqa: E402
from app.memory.graph import project_unit, remove_unit_from_graph  # noqa: E402
from app.models import (  # noqa: E402
    EntityRelation,
    MemoryEntity,
    MemoryUnit,
    UnitAssociation,
    UnitEntityLink,
)


def _report_database(db) -> str | None:
    """Print the database this run will touch, and verify it is the one
    DATABASE_URL names.

    Same `SELECT current_database()` idea as eval/_helpers.py's
    assert_eval_database(): ask the live connection, not an env var, so the
    check cannot be fooled by how the connection was established. Unlike the
    eval guard this does *not* pin a specific name -- this script is meant to
    run against the real `membase`/`membase_prod` database. It only refuses
    when the connection disagrees with the configured URL, which would mean
    you are about to rewrite graph rows in a database you did not name (see
    CLAUDE.md footguns #9/#10 -- `membase`, `membase_prod` and `membase_eval`
    are three different databases on the same cluster).

    Returns the live database name, or None if it mismatches.
    """
    url = make_url(get_settings().database_url)
    live = db.execute(sql_text("SELECT current_database()")).scalar()
    # The dev/prod/eval URLs connect over a Unix socket, so host/port live in
    # the query string rather than in the authority part of the URL.
    host = url.host or url.query.get("host") or "?"
    port = url.port or url.query.get("port") or "?"
    print(f"Database: {live}  (host {host}, port {port})")
    if live != url.database:
        print(
            f"ERROR: connected to database {live!r} but DATABASE_URL names "
            f"{url.database!r}. Refusing to rewrite graph rows in a database "
            f"this run did not ask for -- fix DATABASE_URL (or the shell you "
            f"are in) and rerun. See CLAUDE.md footguns #9/#10.",
            file=sys.stderr,
        )
        return None
    return live


def _units_with_graph_rows(db) -> set[uuid.UUID]:
    """Every unit id that currently owns at least one projected row, from any
    of the four tables -- including associations where the unit is the
    *target*, since `_ordered_pair` means either side can be the owner."""
    ids: set[uuid.UUID] = set()
    ids.update(db.execute(select(UnitEntityLink.unit_id).distinct()).scalars())
    ids.update(db.execute(select(EntityRelation.unit_id).distinct()).scalars())
    ids.update(db.execute(select(UnitAssociation.source_unit_id).distinct()).scalars())
    ids.update(db.execute(select(UnitAssociation.target_unit_id).distinct()).scalars())
    return ids


def _count(db, column) -> int:
    return int(db.execute(select(func.count(column))).scalar() or 0)


def _ready_units(db, unit_id: uuid.UUID | None):
    query = db.query(MemoryUnit).filter(MemoryUnit.status == "ready")
    if unit_id is not None:
        query = query.filter(MemoryUnit.id == unit_id)
    return query.order_by(MemoryUnit.created_at.asc()).all()


def _stale_graph_unit_ids(db) -> list[uuid.UUID]:
    """Units that own graph rows but are no longer `ready` -- status flipped to
    discarded/failed after a previous projection wrote them."""
    return list(
        db.execute(
            select(MemoryUnit.id)
            .where(
                MemoryUnit.status != "ready",
                or_(
                    MemoryUnit.id.in_(select(UnitEntityLink.unit_id)),
                    MemoryUnit.id.in_(select(EntityRelation.unit_id)),
                    MemoryUnit.id.in_(select(UnitAssociation.source_unit_id)),
                    MemoryUnit.id.in_(select(UnitAssociation.target_unit_id)),
                ),
            )
            .order_by(MemoryUnit.created_at.asc())
        ).scalars()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report what would be projected without changing anything")
    parser.add_argument("--unit-id", default=None, help="re-project exactly this unit (debugging)")
    parser.add_argument("--batch-size", type=int, default=50, help="commit every N units (default: 50)")
    parser.add_argument(
        "--prune", action="store_true",
        help="also delete graph rows owned by units that are no longer 'ready'",
    )
    args = parser.parse_args()

    unit_id = None
    if args.unit_id is not None:
        try:
            unit_id = uuid.UUID(str(args.unit_id))
        except (ValueError, TypeError):
            print(f"ERROR: --unit-id {args.unit_id!r} is not a UUID", file=sys.stderr)
            return 1
    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1", file=sys.stderr)
        return 1

    db = get_sessionmaker()()
    if _report_database(db) is None:
        db.close()
        return 1

    units = _ready_units(db, unit_id)
    if unit_id is not None and not units:
        print(f"No ready unit with id {unit_id} (missing, or status is not 'ready')")
        db.close()
        return 1

    projected_ids = _units_with_graph_rows(db)
    with_rows = sum(1 for u in units if u.id in projected_ids)
    print(f"{len(units)} ready unit(s) in scope")
    print(f"  {len(units) - with_rows} with no graph rows yet, {with_rows} already projected")
    print(
        f"Existing graph: {_count(db, MemoryEntity.id)} entities, "
        f"{_count(db, UnitEntityLink.id)} entity links, "
        f"{_count(db, EntityRelation.id)} relations, "
        f"{_count(db, UnitAssociation.id)} associations"
    )

    # Pruning is opt-in, not always-on, because deletion is the one thing in
    # this script that is not a rebuildable projection: re-projecting a unit
    # can always be undone by re-projecting it again, but dropping a non-ready
    # unit's rows cannot be undone without the unit coming back to 'ready'.
    # A status is also allowed to flip back (a `failed` unit is explicitly
    # retryable, see CLAUDE.md "Status gate"), so "not ready right now" is not
    # the same as "should not be in the graph forever". Keeping the delete
    # behind a flag means the default run of a maintenance script only ever
    # *adds* correctness, and never silently removes graph rows as a side
    # effect of a backfill someone ran for an unrelated reason.
    stale_ids = _stale_graph_unit_ids(db)
    if args.prune:
        print(f"--prune: {len(stale_ids)} non-ready unit(s) still own graph rows")
    elif stale_ids:
        print(f"{len(stale_ids)} non-ready unit(s) still own graph rows (pass --prune to remove them)")

    if args.dry_run:
        print("\n--dry-run: reporting only, no changes will be made")
        db.close()
        return 0

    processed = entity_links = relations = associations = failed = 0
    for i, unit in enumerate(units, start=1):
        # project_unit, not project_unit_safely: the _safely wrapper exists for
        # the ingestion call sites, where it swallows the error and commits on
        # the caller's behalf -- both wrong here, because a backfill must report
        # which units failed and must own its own batched commits. The guard is
        # instead a per-unit SAVEPOINT: one unit with unexpected semantics
        # rolls back only itself and the sweep continues, so a single bad row
        # can never abort a backfill over the whole corpus.
        try:
            with db.begin_nested():
                result = project_unit(db, unit)
        except Exception as exc:  # noqa: BLE001 -- see comment above
            print(f"  [{i}/{len(units)}] FAILED {unit.id}: {exc}")
            failed += 1
            continue
        processed += 1
        entity_links += result.entity_links
        relations += result.relations
        associations += result.associations
        if i % args.batch_size == 0 or i == len(units):
            db.commit()
            print(
                f"  [{i}/{len(units)}] committed  "
                f"(+{entity_links} links, +{relations} relations, +{associations} associations so far)"
            )

    pruned = 0
    if args.prune:
        for stale_id in stale_ids:
            try:
                with db.begin_nested():
                    remove_unit_from_graph(db, stale_id)
            except Exception as exc:  # noqa: BLE001 -- same rationale as above
                print(f"  prune FAILED {stale_id}: {exc}")
                failed += 1
                continue
            pruned += 1
        db.commit()
        print(f"  pruned {pruned} non-ready unit(s) from the graph")

    db.commit()
    # These are sums of per-unit ProjectionResults, so an association can be
    # counted twice over a full-corpus run: each endpoint deletes and rewrites
    # the pair it owns. The "Graph now" line below is the stable figure to
    # compare between runs when checking idempotence.
    print(
        f"\n{processed}/{len(units)} unit(s) projected: {entity_links} entity links, "
        f"{relations} relations, {associations} association edges written, {failed} failed"
    )
    print(
        f"Graph now: {_count(db, MemoryEntity.id)} entities, "
        f"{_count(db, UnitEntityLink.id)} entity links, "
        f"{_count(db, EntityRelation.id)} relations, "
        f"{_count(db, UnitAssociation.id)} associations"
    )
    db.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
