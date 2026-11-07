#!/usr/bin/env python3
"""Seed a clearly-labelled SYNTHETIC demonstration dataset.

Everything this script stores is invented. No part of it is a real memory of
any real person, and the memories say so where it matters. It exists because
the demonstration scenario in the product brief needs a store with a specific
shape -- several similar trips that are genuinely hard to tell apart from a
vague query -- and an empty database or a handful of real notes cannot show
that.

The scenario it is built for:

    "I am trying to remember a photograph from a trip a few years ago. I think
     we were near a lake, possibly in Switzerland or Austria. There were
     mountains in the background, and I remember taking it with a friend."

That query is deliberately under-specified. Three stored trips match it
partially -- a lake, mountains, a friend, a few years ago -- and they differ in
exactly one attribute each (who was there, which country, which year), so the
system has to explore associations and then ask one discriminating question
rather than guessing. The supporting memories about each friend and place share
no wording with the query at all, which is what makes the associative hop
necessary rather than decorative.

The images are generated here with Pillow (already a dependency), not sourced
from anywhere: a flat-colour mountain-and-lake scene per trip. They are
obviously illustrations rather than photographs, and the captions the model
produces for them say so. That is the honest trade-off -- a synthetic image
still exercises the whole path (vision caption -> visual concepts -> entities
-> association -> retrieval) with no real personal data involved.

Usage (from the repo root, inside `nix develop`):

    python3 scripts/seed_demo.py --dry-run     # show what would be stored
    python3 scripts/seed_demo.py               # seed it

Ingestion runs the real pipeline against the real local model, so this takes a
while: roughly 20-25s of model time per memory (see docs/DECISIONS.md
"Benchmark: fast pipeline vs. structurer wall-clock time"), plus a vision call
per image. Expect several minutes.

This writes to whatever database `DATABASE_URL` points at and prints which one
before doing anything. It refuses to touch `membase_prod` without an explicit
flag -- see CLAUDE.md footgun #10 for why that database is protected.
"""

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from PIL import Image, ImageDraw  # noqa: E402
from sqlalchemy import func, select, text  # noqa: E402

from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import create_capture, process_capture_fully  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.models import Capture, Episode, MemoryEntity, MemoryUnit, UnitAssociation  # noqa: E402

PROTECTED_DATABASES = ("membase_prod",)


# --------------------------------------------------------------------------
# Synthetic images
# --------------------------------------------------------------------------


def _render_lake_scene(path: Path, *, sky: tuple, mountain: tuple, water: tuple, peaks: int) -> None:
    """A flat-colour lake-and-mountains scene.

    Deliberately simple and obviously not a photograph. What matters for the
    demonstration is that a vision model can see "a lake with mountains behind
    it", which drives `semantic.visual_concepts` and from there the entity
    graph -- not that the image is realistic.
    """
    width, height = 768, 512
    horizon = int(height * 0.55)
    image = Image.new("RGB", (width, height), sky)
    draw = ImageDraw.Draw(image)

    # Mountain silhouettes behind the horizon.
    for index in range(peaks):
        centre = int(width * (index + 0.5) / peaks)
        half_base = int(width / peaks * 0.85)
        peak_height = int(height * (0.22 + 0.10 * math.sin(index * 1.7 + 1)))
        shade = tuple(max(0, c - 14 * (index % 3)) for c in mountain)
        draw.polygon(
            [
                (centre - half_base, horizon),
                (centre, horizon - peak_height),
                (centre + half_base, horizon),
            ],
            fill=shade,
        )
        # A snow cap, so "snow" is a visible concept the model can report.
        cap = int(peak_height * 0.3)
        draw.polygon(
            [
                (centre - int(half_base * cap / peak_height), horizon - peak_height + cap),
                (centre, horizon - peak_height),
                (centre + int(half_base * cap / peak_height), horizon - peak_height + cap),
            ],
            fill=(242, 244, 248),
        )

    # The lake, with a few horizontal highlights so it reads as water.
    draw.rectangle([0, horizon, width, height], fill=water)
    for offset in range(horizon + 14, height, 26):
        lighter = tuple(min(255, c + 26) for c in water)
        draw.line([(int(width * 0.08), offset), (int(width * 0.92), offset)], fill=lighter, width=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=88)


IMAGES = {
    "bled_lake_view.jpg": {"sky": (168, 199, 224), "mountain": (92, 108, 120), "water": (56, 104, 122), "peaks": 3},
    "thun_lake_view.jpg": {"sky": (196, 214, 231), "mountain": (104, 114, 128), "water": (64, 118, 138), "peaks": 4},
    "hallstatt_lake_view.jpg": {"sky": (150, 176, 203), "mountain": (84, 96, 110), "water": (48, 90, 110), "peaks": 2},
}


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------
#
# Each entry is one capture: the text the "user" typed, plus any attachments.
# The `note` field is not stored -- it documents what each memory is for, so
# the dataset's structure is auditable rather than something to be inferred by
# reading the texts and guessing.

DEMO_MEMORIES = [
    # --- The three competing lake trips. All three match the demo query
    # partially; each differs from the others in who/where/when.
    {
        "text": (
            "Took this photo on the hike above Lake Bled in July 2023. Markus and I walked up to "
            "the viewpoint early so the water was still completely flat, with the mountains "
            "behind it."
        ),
        "images": ["bled_lake_view.jpg"],
        "note": "Target of the demo query. Lake + mountains + a friend + 'a few years ago'.",
    },
    {
        "text": (
            "A weekend at Lake Thun in September 2022 with Nadia. We rented a boat and I "
            "photographed the mountains from the middle of the lake."
        ),
        "images": ["thun_lake_view.jpg"],
        "note": "Competing candidate: also a lake, mountains, a friend. Switzerland, different year.",
    },
    {
        "text": (
            "Hallstatt in the spring of 2021. The lake was ringed by mountains and still half in "
            "shadow at midday. I went on my own that time."
        ),
        "images": ["hallstatt_lake_view.jpg"],
        "note": "Competing candidate: Austria, lake + mountains, but NO friend -- a 'did you go with someone' question separates it.",
    },
    # --- Supporting memories. These share almost no wording with the demo
    # query, so they are only reachable by association from a memory that did
    # match. They are what the clarifying question's answer is checked against.
    {
        "text": (
            "Markus is doing a doctorate in glaciology at ETH Zurich. He is the one who keeps "
            "sending me alpine routes I would never have found on my own."
        ),
        "note": "Multi-hop anchor for Markus. Shares 'Markus' with the Bled trip and nothing with the query.",
    },
    {
        "text": (
            "Nadia moved to Bern in 2021 for a job at a civil engineering practice. She is the "
            "reason I started going to the Bernese Oberland at all."
        ),
        "note": "Multi-hop anchor for Nadia, and a second route into the Thun trip.",
    },
    {
        "text": (
            "Lake Bled is in Slovenia, not Switzerland. I keep having to correct myself about "
            "this because the whole valley looks so much like the Alps around Interlaken."
        ),
        "note": (
            "Directly addresses the demo query's wrong guess ('possibly in Switzerland or "
            "Austria'). Retrieval has to surface this rather than silently accept the premise."
        ),
    },
    {
        "text": (
            "The viewpoint above Lake Bled is called Ojstrica. It is a short but very steep walk "
            "up from the lake, maybe twenty minutes."
        ),
        "note": "Second-hop detail about the target trip; reachable only via 'Lake Bled'.",
    },
    # --- Lexical distractors: share salient query words, about something else.
    {
        "text": (
            "Finally framed the mountain print I bought years ago and hung it above the desk. It "
            "is a photograph of a lake, but not one I have ever been to."
        ),
        "note": "Lexical distractor: contains 'mountain', 'photograph', 'lake' and is not a trip at all.",
    },
    {
        "text": (
            "Swam in the Zurich lake after work almost every evening in August. No mountains "
            "worth photographing, just the city on both sides."
        ),
        "note": "Lexical distractor: a lake, no mountains, no trip, no friend.",
    },
    # --- Temporal variety, including a memory with no resolvable date at all.
    {
        "text": "Bought proper hiking boots before the Slovenia trip. The old ones had no grip left on wet rock.",
        "note": "Undated on purpose: tests that the timeline reports 'date unknown' rather than inventing one.",
    },
    {
        "text": (
            "In February 2024 I gave a short talk about glacier retreat at the department, using "
            "some of Markus's figures."
        ),
        "note": "Same person, clearly different episode and year -- should NOT join the Bled episode.",
    },
    # --- A causal link and a contradiction pair, so the explainability and
    # conflict-surfacing paths have something real to show.
    {
        "text": (
            "We abandoned the second day of the Bled trip because the forecast turned and the "
            "path up Ojstrica was already slick."
        ),
        "note": "Causal relationship, same episode as the target trip.",
    },
    {
        "text": "My favourite of the three lake photographs is definitely the one from Hallstatt.",
        "note": "Contradiction pair, part 1.",
    },
    {
        "text": "Looking at them again, the Lake Bled photograph is clearly the best of the three.",
        "note": "Contradiction pair, part 2 -- the answer must state the disagreement, not pick a side.",
    },
    # --- A terse fragment and an unrelated fact, so the corpus is not uniformly
    # well-formed prose.
    {
        "text": "Ferry timetable for the lake changes in October.",
        "note": "Fragmentary recollection with minimal context.",
    },
    {
        "text": "The bike needs a new chain before spring.",
        "note": "Unrelated: should never appear in any lake-photograph result.",
    },
]


def _database_name(db) -> str:
    return db.execute(text("SELECT current_database()")).scalar_one()


async def _seed(db, llm, *, uploads: Path) -> None:
    total = len(DEMO_MEMORIES)
    started = time.monotonic()
    for index, entry in enumerate(DEMO_MEMORIES, start=1):
        files = []
        for name in entry.get("images", []):
            data = (uploads / name).read_bytes()
            files.append((name, data))
        label = entry["text"][:58].replace("\n", " ")
        print(f"[{index}/{total}] {label}...", flush=True)
        capture = await create_capture(db, text=entry["text"], uploads=files)
        await process_capture_fully(capture.id, llm=llm)
        db.expire_all()
    elapsed = time.monotonic() - started
    print(f"\nSeeded {total} memories in {elapsed / 60:.1f} min.")


def _report(db) -> None:
    ready = int(
        db.execute(select(func.count(MemoryUnit.id)).where(MemoryUnit.status == "ready")).scalar() or 0
    )
    entities = int(db.execute(select(func.count(MemoryEntity.id))).scalar() or 0)
    associations = int(db.execute(select(func.count(UnitAssociation.id))).scalar() or 0)
    episodes = int(db.execute(select(func.count(Episode.id))).scalar() or 0)
    captures = int(db.execute(select(func.count(Capture.id))).scalar() or 0)
    print(
        f"\nStore now: {captures} captures, {ready} ready memory units, {entities} entities, "
        f"{associations} association edges, {episodes} episodes."
    )
    print("\nTry this in Recall:")
    print(
        '  "I am trying to remember a photograph from a trip a few years ago. I think we were\n'
        '   near a lake, possibly in Switzerland or Austria. There were mountains in the\n'
        '   background, and I remember taking it with a friend."'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="describe the dataset and exit, storing nothing")
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="permit seeding the protected production database (see CLAUDE.md footgun #10)",
    )
    parser.add_argument(
        "--images-dir",
        default=str(REPO_ROOT / ".data" / "demo-images"),
        help="where to write the generated synthetic images",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(f"SYNTHETIC demonstration dataset: {len(DEMO_MEMORIES)} memories, {len(IMAGES)} generated images.\n")
        for index, entry in enumerate(DEMO_MEMORIES, start=1):
            attachments = ", ".join(entry.get("images", [])) or "-"
            print(f"{index:2}. {entry['text'][:72]}")
            print(f"     attachments: {attachments}")
            print(f"     purpose: {entry['note']}\n")
        print("--dry-run: nothing was stored.")
        return 0

    db = get_sessionmaker()()
    try:
        name = _database_name(db)
        print(f"Database: {name}")
        if name in PROTECTED_DATABASES and not args.allow_prod:
            print(
                f"\nERROR: refusing to seed synthetic demonstration data into '{name}'.\n"
                "That database holds real stored memories (CLAUDE.md footgun #10). Point\n"
                "DATABASE_URL at the dev database, or pass --allow-prod if you really mean it.",
                file=sys.stderr,
            )
            return 1

        uploads = Path(args.images_dir)
        for filename, spec in IMAGES.items():
            _render_lake_scene(uploads / filename, **spec)
        print(f"Generated {len(IMAGES)} synthetic images in {uploads}")

        llm = get_llm_provider()
        asyncio.run(_seed(db, llm, uploads=uploads))
        db.expire_all()
        _report(db)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
