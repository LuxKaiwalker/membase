#!/usr/bin/env python3
"""Export the synthetic demonstration store as a static API snapshot.

This is what makes the published web app usable by someone who has no
credentials. The live backend stays entirely private: it runs on this
machine's loopback interface, reachable from the internet only through
one nginx location that applies `auth_basic` before it proxies
(see docs/DEPLOYMENT.md). Nothing here changes that. Instead, every
read-only API response the browse pages need is computed once, against
the SYNTHETIC `membase_demo` store (`scripts/seed_demo.py`), and written
to plain files published alongside the frontend. Logged out, the app
reads those files; signed in, it talks to the real backend exactly as
before. See docs/DECISIONS.md "Public demo as a static snapshot".

Why this shape rather than a second public backend:

  - A public read-only backend would need a new unauthenticated nginx
    location, a second uvicorn process, its own rate limiting, and it
    would only be up while this machine is. The snapshot needs none of
    that and does not go down.
  - Writes are impossible by construction rather than by policy. There
    is no endpoint to POST to; `GET` of a file is the only operation the
    public surface supports.
  - No LLM call is reachable. Recall costs tens of seconds of GPU time
    per round on one 11GB card (see CLAUDE.md); exposing it to anonymous
    traffic would hand out that GPU. Stored recall sessions are exported
    as data instead, so the reasoning is inspectable without running it.

Responses are produced by driving the real FastAPI app in-process over
ASGI, NOT by re-implementing any endpoint here. The snapshot is
therefore byte-identical to what the live API would have returned for
the same URL, including response_model field filtering, and it cannot
drift from the API as the schemas change -- a reimplementation would
silently start lying the first time a field was added.

Usage (from the repo root, inside `nix develop`):

    DATABASE_URL=...membase_demo python3 scripts/export_demo.py --dry-run
    DATABASE_URL=...membase_demo python3 scripts/export_demo.py

It refuses to run against any database other than `membase_demo`, asked
of the live connection rather than taken from an env var: exporting the
owner's real store into a directory destined for a public web server is
exactly the mistake worth making impossible. See CLAUDE.md footgun #10.
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import httpx  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402

from app.db import get_sessionmaker  # noqa: E402
from app.models import (  # noqa: E402
    Attachment,
    Capture,
    Episode,
    MemoryEntity,
    MemoryUnit,
    RecallSession,
)

DEMO_DATABASE = "membase_demo"
OUTPUT_DIR = REPO_ROOT / "public-demo"

# Mirrors of the constants the frontend builds its request URLs from. A
# static snapshot can only answer URLs it was asked to produce, so these
# have to agree with the frontend exactly -- hence the explicit
# cross-references. test_export_demo.py asserts they still match the
# values in app/static/, which is what stops this drifting silently when
# someone retunes a page.
EXPLORER_MAX_NODES = 400  # app/static/pages/explorer.js MAX_NODES
EXPLORER_ASSOC_LIMIT = 12  # app/static/pages/explorer.js, the associations fetch
ENTITY_LIST_LIMIT = 500  # the route's own ceiling: one file, filtered client-side
TIMELINE_PAGE_SIZES = (5, 10, 25, 50)  # app/templates/timeline.html's #tl-page-size
OVERVIEW_EPISODE_LIMIT = 5  # app/static/pages/overview.js RECENT_EPISODE_LIMIT
HOP_CHOICES = (1, 2)  # app/memory/graph.py MAX_HOPS


def _safe_segment(value: str) -> str:
    """Filesystem- and URL-safe form of a query string.

    Deliberately simple and total: every character outside the safe set
    becomes `_`. The same function exists in app/static/core.js
    (`snapshotKey`), and the two must agree character-for-character or
    the browser asks for a file this script never wrote. Kept this dumb
    precisely so it is reimplementable in six lines of JS without
    sharing a library.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def snapshot_path(url: str) -> str:
    """Map an API URL to its file within the snapshot.

    `/api/graph?hops=1&max_nodes=400` -> `api/graph~hops_1_max_nodes_400.json`
    `/api/graph/entities/<uuid>`      -> `api/graph/entities/<uuid>.json`

    Query parameters are sorted, so the file a request resolves to does
    not depend on the order the frontend happened to build them in.
    Values are NOT defaulted: a URL that omits a parameter is a
    different URL, and the exporter lists both forms where the frontend
    sends both (e.g. /api/episodes?limit=5 from the overview page vs.
    the timeline's fully-specified query).
    """
    path, _, query = url.partition("?")
    assert path.startswith("/api/"), url
    rel = path[len("/api/") :].strip("/")
    if query:
        canonical = "&".join(sorted(query.split("&")))
        return f"api/{rel}~{_safe_segment(canonical)}.json"
    return f"api/{rel}.json"


def _assert_demo_database(db) -> bool:
    live = db.execute(sql_text("SELECT current_database()")).scalar()
    print(f"Database: {live}")
    if live != DEMO_DATABASE:
        print(
            f"ERROR: refusing to export from {live!r}. This writes a directory "
            f"that gets published to a public web server, so it only ever runs "
            f"against the synthetic {DEMO_DATABASE!r} store -- never the "
            f"owner's real memories. Point DATABASE_URL at {DEMO_DATABASE} "
            f"(see scripts/seed_demo.py) and rerun.",
            file=sys.stderr,
        )
        return False
    return True


def _ids(db, column) -> list[str]:
    return [str(v) for v in db.execute(select(column)).scalars()]


def declared_urls(db) -> list[str]:
    """Every URL the published pages can ask for, in one place."""
    capture_ids = _ids(db, Capture.id)
    unit_ids = _ids(db, MemoryUnit.id)
    ready_unit_ids = [
        str(v)
        for v in db.execute(select(MemoryUnit.id).where(MemoryUnit.status == "ready")).scalars()
    ]
    entity_ids = _ids(db, MemoryEntity.id)
    episode_ids = _ids(db, Episode.id)
    session_ids = _ids(db, RecallSession.id)
    episode_total = len(episode_ids)

    urls: list[str] = [
        "/api/captures",
        "/api/graph/stats",
        f"/api/graph/entities?limit={ENTITY_LIST_LIMIT}",
        f"/api/episodes?limit={OVERVIEW_EPISODE_LIMIT}",
    ]

    # The explorer's whole-graph view, and one centred view per memory.
    # Entity-kind / association-kind filters are deliberately absent:
    # the server applies them BEFORE the max_nodes cap, so filtering an
    # already-capped snapshot client-side would quietly show a different
    # graph than the live API does for the same filter. Public mode hides
    # those controls instead of approximating them (see
    # app/static/pages/explorer.js).
    for hops in HOP_CHOICES:
        urls.append(f"/api/graph?max_nodes={EXPLORER_MAX_NODES}&hops={hops}")
        for unit_id in ready_unit_ids:
            urls.append(
                f"/api/graph?max_nodes={EXPLORER_MAX_NODES}&hops={hops}&center_unit_id={unit_id}"
            )

    for unit_id in unit_ids:
        urls.append(f"/api/memory-units/{unit_id}")
        urls.append(f"/api/episodes/for-unit/{unit_id}")
        for hops in HOP_CHOICES:
            # The explorer passes an explicit limit; the recall page's
            # "walk the graph" buttons don't. Two different URLs, both
            # really sent, so both are exported.
            urls.append(f"/api/graph/units/{unit_id}/associations?hops={hops}&limit={EXPLORER_ASSOC_LIMIT}")
            urls.append(f"/api/graph/units/{unit_id}/associations?hops={hops}")

    for capture_id in capture_ids:
        urls.append(f"/api/captures/{capture_id}")
    for entity_id in entity_ids:
        urls.append(f"/api/graph/entities/{entity_id}")
    for episode_id in episode_ids:
        urls.append(f"/api/episodes/{episode_id}")
    for session_id in session_ids:
        urls.append(f"/api/recall/sessions/{session_id}")

    # The timeline pages through episodes; export every page it can
    # actually reach rather than just the first, so paging works offline.
    for limit in TIMELINE_PAGE_SIZES:
        for include_undated in ("true", "false"):
            offset = 0
            while True:
                urls.append(
                    f"/api/episodes?limit={limit}&offset={offset}&include_undated={include_undated}"
                )
                offset += limit
                if offset >= max(episode_total, 1):
                    break

    # Dedupe while preserving order -- several pages legitimately ask for
    # the same URL.
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _session_summaries(db) -> list[dict]:
    """Index the public Recall page lists, newest first.

    Only the fields needed to choose a session; the session itself is a
    separate exported file rendered by the normal recall renderer.
    """
    sessions = list(
        db.execute(select(RecallSession).order_by(RecallSession.created_at.desc())).scalars()
    )
    return [
        {
            "id": str(s.id),
            "query": s.original_query,
            "status": s.status,
            "rounds": s.round_count,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sessions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be written")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help=f"snapshot directory (default: {OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    db = get_sessionmaker()()
    if not _assert_demo_database(db):
        db.close()
        return 1

    urls = declared_urls(db)
    attachments = list(db.execute(select(Attachment)).scalars())
    sessions = _session_summaries(db)
    print(f"{len(urls)} URL(s), {len(attachments)} attachment file(s), {len(sessions)} recall session(s)")

    if args.dry_run:
        for url in urls[:20]:
            print(f"  {url}  ->  {snapshot_path(url)}")
        if len(urls) > 20:
            print(f"  ... and {len(urls) - 20} more")
        print("\n--dry-run: nothing written")
        db.close()
        return 0

    out = Path(args.output)
    # Rebuilt from scratch every time: a stale file left behind from a
    # previous corpus would be served as current, and the snapshot is
    # supposed to be a faithful picture of one store at one moment.
    if out.exists():
        shutil.rmtree(out)
    (out / "api").mkdir(parents=True)

    from app.main import app  # noqa: PLC0415 -- after the database guard above

    failures: list[str] = []
    written = 0

    # httpx's ASGITransport is async-only (it awaits the app directly --
    # there is no WSGI-style sync path to an ASGI callable), so the
    # request loop runs under asyncio.run even though everything else
    # here is plain synchronous SQLAlchemy.
    async def fetch_all() -> tuple[int, list[str]]:
        count = 0
        errors: list[str] = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://snapshot", timeout=120.0
        ) as client:
            for url in urls:
                response = await client.get(url)
                if response.status_code != 200:
                    errors.append(f"{url} -> HTTP {response.status_code}")
                    continue
                target = out / snapshot_path(url)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                count += 1

            # Attachment bytes, served from the same relative path the
            # API uses so the frontend's existing <img src> construction
            # needs no special case beyond the base URL.
            for attachment in attachments:
                url = f"/api/attachments/{attachment.id}/file"
                response = await client.get(url)
                if response.status_code != 200:
                    errors.append(f"{url} -> HTTP {response.status_code}")
                    continue
                target = out / "api" / "attachments" / str(attachment.id) / "file"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                count += 1
        return count, errors

    written, failures = asyncio.run(fetch_all())

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_database": DEMO_DATABASE,
        "synthetic": True,
        "counts": {
            "captures": len(_ids(db, Capture.id)),
            "memory_units": len(_ids(db, MemoryUnit.id)),
            "entities": len(_ids(db, MemoryEntity.id)),
            "episodes": len(_ids(db, Episode.id)),
            "recall_sessions": len(sessions),
            "files": written,
        },
        "recall_sessions": sessions,
        # Read by core.js to decide whether a URL is answerable offline
        # before it tries: a missing file must become an explicit "not in
        # the snapshot" message, never a silent blank page.
        "urls": [snapshot_path(url) for url in urls],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Last line of defence before this directory is published. The
    # export is supposed to contain synthetic data and nothing else;
    # an absolute path from this machine appearing in it would mean a
    # schema started exposing `stored_path` (or similar) and nobody
    # noticed.
    leaked = []
    needles = [str(Path.home()), os.environ.get("UPLOADS_DIR", "") or "\0", "membase_prod"]
    for path in out.rglob("*.json"):
        blob = path.read_text(errors="ignore")
        for needle in needles:
            if needle and needle in blob:
                leaked.append(f"{path.relative_to(out)}: contains {needle!r}")
    db.close()

    print(f"Wrote {written} file(s) to {out}")
    if leaked:
        print("\nERROR: the snapshot contains host-specific or non-demo strings:", file=sys.stderr)
        for line in leaked:
            print(f"  {line}", file=sys.stderr)
        return 1
    if failures:
        print(f"\n{len(failures)} URL(s) failed:", file=sys.stderr)
        for line in failures[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
