#!/usr/bin/env python3
"""Record the recall sessions the public demo replays.

The published site is read-only (see docs/DECISIONS.md "Public demo as a
static snapshot"): a logged-out visitor cannot start a recall, because a
round costs tens of seconds of GPU time on a single 11GB card and
anonymous access to that would hand out the GPU. But recall is the part
of this system most worth looking at -- the evidence behind each
candidate, the confidence category, the point where it asks a question
instead of guessing -- so the sessions are recorded here, once, against
the SYNTHETIC demonstration store, and exported with the rest of the
snapshot. `app/static/pages/recall.js` renders them through exactly the
same code path the live page uses.

Each query below is chosen to show a different outcome, because a demo
that only shows successes is a brochure:

  - a vague query that needs the associative hop to get anywhere
  - a query several stored trips match, where the right behaviour is to
    ask one discriminating question rather than guess
  - a direct question that should simply be answered, confidently
  - a question about something that is genuinely not in the store, where
    the right behaviour is to decline

Usage (from the repo root, inside `nix develop`):

    DATABASE_URL=...membase_demo python3 scripts/seed_demo_sessions.py --dry-run
    DATABASE_URL=...membase_demo python3 scripts/seed_demo_sessions.py

Refuses any database but `membase_demo` (CLAUDE.md footgun #11): these
are real model calls that write rows, and they belong only in the store
whose contents are meant to be published.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402

from app.db import get_sessionmaker  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.models import MemoryUnit, RecallSession  # noqa: E402
from app.retrieval.session import start_session  # noqa: E402

DEMO_DATABASE = "membase_demo"

# (query, why this one is in the demo). The "why" is printed, not stored:
# the session row speaks for itself, and a hand-written label next to a
# real result is exactly the kind of thing that goes stale when the
# corpus changes.
DEMO_QUERIES = [
    (
        "I am trying to remember a photograph from a trip a few years ago. I think we were "
        "near a lake, possibly in Switzerland or Austria. There were mountains in the "
        "background, and I remember taking it with a friend.",
        "the scenario the demo corpus was built for: three trips match partially, so the "
        "honest move is one discriminating question",
    ),
    (
        "what was the name of the glaciologist friend again",
        "reachable only through the entity graph -- the memory naming him shares no wording "
        "with this query",
    ),
    (
        "which lake did I visit in 2022",
        "a direct question with a single answer; should answer, not ask",
    ),
    (
        "what did the doctor say about my knee last winter",
        "nothing in the store is about this at all; should decline rather than reach for "
        "the nearest loosely-related memory",
    ),
]


def _database_name(db) -> str:
    return db.execute(sql_text("SELECT current_database()")).scalar_one()


async def _record(llm, queries) -> int:
    """One session per query, each in its own DB session.

    Short-lived deliberately: a round takes tens of seconds of model
    time, and a Session held open across that can sit
    idle-in-transaction until Postgres terminates the connection (see
    docs/DECISIONS.md "One LLM call per open transaction" -- the same
    trap scripts/seed_demo.py fell into).
    """
    recorded = 0
    for index, (query, why) in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {query[:70]}...")
        print(f"      ({why})")
        started = time.monotonic()
        db = get_sessionmaker()()
        try:
            session = await start_session(db, llm, query)
        except Exception as exc:  # noqa: BLE001
            # One failed query must not cost the whole run: each session
            # is independent, and a partial set of recorded recalls is
            # still a usable demo.
            print(f"      FAILED after {time.monotonic() - started:.1f}s: {exc}")
            continue
        finally:
            detail = ""
            try:
                detail = session.final_confidence or session.abstention_reason or ""
                status = session.status
            except Exception:  # noqa: BLE001
                status = "unknown"
            db.close()
        print(
            f"      -> {status}{f' ({detail})' if detail else ''}"
            f" in {time.monotonic() - started:.1f}s"
        )
        recorded += 1
    return recorded


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="list the queries and exit")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="add to the sessions already recorded instead of replacing them",
    )
    args = parser.parse_args()

    db = get_sessionmaker()()
    live = _database_name(db)
    print(f"Database: {live}")
    if live != DEMO_DATABASE:
        print(
            f"ERROR: refusing to record demo recall sessions in {live!r}. These are real "
            f"model calls that write rows, and they belong only in the synthetic store whose "
            f"contents get published ({DEMO_DATABASE}). See CLAUDE.md footgun #11.",
            file=sys.stderr,
        )
        db.close()
        return 1

    ready = int(
        db.execute(select(func.count(MemoryUnit.id)).where(MemoryUnit.status == "ready")).scalar()
        or 0
    )
    existing = int(db.execute(select(func.count(RecallSession.id))).scalar() or 0)
    print(f"{ready} ready memory unit(s), {existing} session(s) already recorded")
    if ready == 0:
        print(
            "ERROR: the demo store is empty -- run scripts/seed_demo.py first.",
            file=sys.stderr,
        )
        db.close()
        return 1

    if args.dry_run:
        for query, why in DEMO_QUERIES:
            print(f"\n  {query}\n      ({why})")
        print("\n--dry-run: nothing recorded")
        db.close()
        return 0

    if existing and not args.keep_existing:
        # Re-running this script should replace its own output, not
        # accumulate near-duplicate sessions the public page would list
        # side by side with no way to tell them apart.
        db.query(RecallSession).delete()
        db.commit()
        print(f"removed {existing} previously recorded session(s) (--keep-existing to add instead)")

    llm = get_llm_provider()
    started = time.monotonic()
    db.close()
    recorded = asyncio.run(_record(llm, DEMO_QUERIES))
    db = get_sessionmaker()()
    total = int(db.execute(select(func.count(RecallSession.id))).scalar() or 0)
    print(
        f"\nRecorded {recorded}/{len(DEMO_QUERIES)} session(s) in "
        f"{(time.monotonic() - started) / 60:.1f} min; {total} in the store."
    )
    print("Next: python3 scripts/export_demo.py")
    db.close()
    return 0 if recorded else 1


if __name__ == "__main__":
    sys.exit(main())
