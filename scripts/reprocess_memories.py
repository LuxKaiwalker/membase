#!/usr/bin/env python3
"""Low-priority reprocessing sweep for memories derived by an older,
now-fixed pipeline version (see docs/DECISIONS.md "Reprocessing
pipeline" and app/ingestion/pipeline.py's CURRENT_PIPELINE_VERSION /
reprocess_unit()).

Finds every MemoryUnit with pipeline_version < CURRENT_PIPELINE_VERSION
and status "ready" or "awaiting_clarification", and re-derives it:

  - "ready" units are already searchable and stay that way -- only
    `semantic`/retrieval/embedding are silently refreshed, the same
    invariant the membase structurer already relies on (never make a
    unit less available than it already was).
  - "awaiting_clarification" units are re-asked under corrected inputs.
    If nothing is left to ask, the unit resolves itself straight to
    "ready" with no user involvement. If a question remains, it REPLACES
    the unit's single pending question in place -- a huge backlog can
    therefore never leave a user with more open questions than they
    already had, only fewer (auto-resolutions) or better-worded ones.

Meant to run manually or from a cron/systemd timer during idle periods
(see CLAUDE.md -- this process shares the one Ollama GPU with the live
backend, which serializes every request regardless). `--delay` paces
calls so a long unattended run leaves gaps for live traffic rather than
saturating the request queue back-to-back.

Usage (from the repo root, inside `nix develop`):
    python3 scripts/reprocess_memories.py                  # reprocess everything eligible
    python3 scripts/reprocess_memories.py --dry-run         # report what would be reprocessed
    python3 scripts/reprocess_memories.py --limit 20        # cap this run's batch size
    python3 scripts/reprocess_memories.py --only-with-attachments  # scope to attachment-bearing captures
    python3 scripts/reprocess_memories.py --delay 10        # seconds between units (default 5)
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import exists  # noqa: E402

from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import CURRENT_PIPELINE_VERSION, reprocess_unit  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.models import Attachment, MemoryUnit  # noqa: E402


def _eligible_units(db, *, only_with_attachments: bool, limit: int | None):
    query = db.query(MemoryUnit).filter(
        MemoryUnit.status.in_(["ready", "awaiting_clarification"]),
        MemoryUnit.pipeline_version < CURRENT_PIPELINE_VERSION,
    )
    if only_with_attachments:
        query = query.filter(exists().where(Attachment.capture_id == MemoryUnit.capture_id))
    query = query.order_by(MemoryUnit.created_at.asc())
    if limit is not None:
        query = query.limit(limit)
    return query.all()


def _describe_outcome(status_before: str, pending_before: dict | None, unit: MemoryUnit) -> str:
    if status_before == "awaiting_clarification" and unit.status == "ready":
        return "resolved -> ready (no further question after reprocessing)"
    if unit.status == "awaiting_clarification":
        field_before = (pending_before or {}).get("field")
        field_after = (unit.pending_clarification or {}).get("field")
        if field_before == field_after and (pending_before or {}).get("question") == (unit.pending_clarification or {}).get("question"):
            return f"still awaiting clarification (field: {field_after}, unchanged)"
        return f"still awaiting clarification (field: {field_after}, question refreshed)"
    return "re-enriched (stayed ready)"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report what would be reprocessed without calling the LLM")
    parser.add_argument("--limit", type=int, default=None, help="reprocess at most this many units this run")
    parser.add_argument(
        "--only-with-attachments", action="store_true",
        help="only reprocess units whose capture has at least one attachment",
    )
    parser.add_argument("--delay", type=float, default=5.0, help="seconds to sleep between units (default: 5)")
    args = parser.parse_args()

    db = get_sessionmaker()()
    llm = get_llm_provider()

    units = _eligible_units(db, only_with_attachments=args.only_with_attachments, limit=args.limit)
    print(f"pipeline_version target: {CURRENT_PIPELINE_VERSION}")
    print(f"{len(units)} unit(s) eligible for reprocessing")
    if args.dry_run:
        print("--dry-run: reporting only, no changes will be made\n")
        for unit in units:
            field = (unit.pending_clarification or {}).get("field")
            suffix = f" (pending: {field})" if unit.status == "awaiting_clarification" else ""
            print(f"  {unit.id}  status={unit.status}{suffix}  pipeline_version={unit.pipeline_version}  created_at={unit.created_at}")
        db.close()
        return 0

    resolved = refreshed = reenriched = failed = 0
    for i, unit in enumerate(units, start=1):
        status_before = unit.status
        pending_before = dict(unit.pending_clarification) if unit.pending_clarification else None
        try:
            unit = await reprocess_unit(db, llm, unit)
        except Exception as exc:  # noqa: BLE001 -- one unit's unexpected failure must not abort the whole sweep
            print(f"  [{i}/{len(units)}] FAILED {unit.id}: {exc}")
            failed += 1
        else:
            outcome = _describe_outcome(status_before, pending_before, unit)
            print(f"  [{i}/{len(units)}] {unit.id}: {outcome}")
            if status_before == "awaiting_clarification" and unit.status == "ready":
                resolved += 1
            elif unit.status == "awaiting_clarification":
                refreshed += 1
            else:
                reenriched += 1
        if i < len(units) and args.delay > 0:
            await asyncio.sleep(args.delay)

    db.close()
    print(f"\n{resolved} resolved, {refreshed} still awaiting clarification, {reenriched} re-enriched, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
