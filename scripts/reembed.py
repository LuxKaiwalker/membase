#!/usr/bin/env python3
"""Re-embedding/backfill tool (spec section 26).

Finds memory units and attachment chunks whose embedding is missing or
was produced by a different model/dimension than the currently
configured one, and regenerates just the embedding -- nothing else
(retrieval_text, semantic data, authoritative content are untouched).
Idempotent and safe to rerun: rows that already match the configured
model are skipped without an LLM call.

Usage (from the repo root, inside `nix develop`):
    python3 scripts/reembed.py            # backfill the main DATABASE_URL
    python3 scripts/reembed.py --dry-run  # report what would change, do nothing
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import get_settings  # noqa: E402
from app.db import get_sessionmaker  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.llm.base import LLMError  # noqa: E402
from app.models import EMBEDDING_DIM, AttachmentChunk, MemoryUnit  # noqa: E402


async def backfill_table(db, llm, model, table_label, rows, text_attr, dry_run: bool) -> tuple[int, int]:
    stale = [
        row
        for row in rows
        if row.embedding is None or row.embedding_model != model or row.embedding_dim != llm.embedding_dim
    ]
    print(f"{table_label}: {len(stale)} stale / {len(rows)} total")
    if dry_run or not stale:
        return len(stale), 0

    failed = 0
    for i, row in enumerate(stale, start=1):
        text_value = getattr(row, text_attr)
        try:
            [embedding] = await llm.embed([text_value])
        except LLMError as exc:
            print(f"  [{i}/{len(stale)}] FAILED {row.id}: {exc}")
            failed += 1
            continue
        row.embedding = embedding
        row.embedding_model = llm.embedding_model_name
        row.embedding_dim = llm.embedding_dim
        db.commit()
        print(f"  [{i}/{len(stale)}] re-embedded {row.id}")

    return len(stale), failed


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report staleness without changing anything")
    args = parser.parse_args()

    settings = get_settings()
    if settings.membase_embed_dim != EMBEDDING_DIM:
        print(
            f"ERROR: configured MEMBASE_EMBED_DIM={settings.membase_embed_dim} does not match the "
            f"schema's fixed vector({EMBEDDING_DIM}) column. Changing to a different-dimension "
            f"embedding model requires an explicit schema migration (ALTER the vector column's "
            f"dimension and rebuild both HNSW indexes), not just a config change -- this tool "
            f"deliberately refuses to proceed rather than attempt that automatically. "
            f"See docs/DECISIONS.md.",
            file=sys.stderr,
        )
        return 1

    db = get_sessionmaker()()
    llm = get_llm_provider()
    print(f"Configured embedding model: {llm.embedding_model_name} (dim {llm.embedding_dim})")
    if args.dry_run:
        print("--dry-run: reporting only, no changes will be made\n")

    units = db.query(MemoryUnit).filter(MemoryUnit.status == "ready").all()
    chunks = db.query(AttachmentChunk).all()

    unit_stale, unit_failed = await backfill_table(
        db, llm, llm.embedding_model_name, "memory_units", units, "retrieval_text", args.dry_run
    )
    chunk_stale, chunk_failed = await backfill_table(
        db, llm, llm.embedding_model_name, "attachment_chunks", chunks, "text", args.dry_run
    )

    db.close()

    total_stale = unit_stale + chunk_stale
    total_failed = unit_failed + chunk_failed
    print(f"\n{total_stale - total_failed}/{total_stale} re-embedded, {total_failed} failed")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
