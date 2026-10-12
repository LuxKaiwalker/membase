"""Retrieval-threshold calibration experiment (spec section 23).

Ingests every case in cases.yaml ONCE (the expensive part: segmentation +
enrichment + embedding, one real LLM call per step), then re-runs just
the cheap SQL-level hybrid_search() for each case's query under several
candidate VECTOR_SIM_THRESHOLD values, without re-ingesting or calling
the LLM again per threshold. For each threshold we measure:

  recall    = fraction of cases where at least one of that case's own
              ("memories") units was found
  precision-ish = fraction of found units that came from "memories"
              rather than "distractors" (a crude false-positive signal)

This is deliberately NOT a full answer-quality sweep (that would require
re-running the agent + LLM per threshold, which is the expensive part) --
it isolates exactly the one mechanism the threshold controls: whether a
candidate clears the SQL-level relevance bar before anything is even
shown to the LLM.
"""

import asyncio
import sys
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from _helpers import assert_eval_database, finish_processing  # noqa: E402
from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import create_capture, resolve_clarification  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
import app.retrieval.pipeline as retrieval_pipeline  # noqa: E402

CANDIDATE_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


async def ingest_all_cases(db, llm, cases: list[dict]) -> list[dict]:
    """Ingests every case into ONE shared database (not truncated between
    cases, unlike run_eval.py) so a single hybrid_search() pass per
    threshold covers the whole set at once. Returns a list of
    {case, own_unit_ids, distractor_unit_ids} for scoring.
    """
    results = []
    for case in cases:
        own_ids: set = set()
        own_captures = []
        for text in case.get("memories", []):
            capture = await create_capture(db, llm, text=text, uploads=[])
            own_captures.append(capture)
            for unit in capture.units:
                if case.get("clarification_answer") and unit.pending_clarification:
                    unit = await resolve_clarification(db, llm, unit, case["clarification_answer"])
                own_ids.add(unit.id)
        # Full enrichment is now a backgrounded step (see
        # docs/DECISIONS.md) -- without this, every unit stays
        # status="processing" and hybrid_search() (status="ready" only)
        # would score every threshold as zero recall.
        await finish_processing(db, llm, *own_captures)

        distractor_ids: set = set()
        distractor_captures = []
        for text in case.get("distractors", []):
            capture = await create_capture(db, llm, text=text, uploads=[])
            distractor_captures.append(capture)
            distractor_ids.update(u.id for u in capture.units)
        await finish_processing(db, llm, *distractor_captures)

        results.append({"case": case, "own_ids": own_ids, "distractor_ids": distractor_ids})
        print(f"  ingested: {case['name']} ({len(own_ids)} own units, {len(distractor_ids)} distractor units)")
    return results


async def score_threshold(db, llm, ingested: list[dict], threshold: float) -> tuple[float, float]:
    retrieval_pipeline.VECTOR_SIM_THRESHOLD = threshold
    found_any = 0
    total_hits = 0
    own_hits = 0
    for entry in ingested:
        case = entry["case"]
        if not entry["own_ids"]:
            continue  # no-answer cases: nothing to measure recall against here
        scored = await retrieval_pipeline.hybrid_search(db, llm, case["query"], limit=10)
        hit_ids = {s.unit.id for s in scored}
        if hit_ids & entry["own_ids"]:
            found_any += 1
        total_hits += len(hit_ids)
        own_hits += len(hit_ids & entry["own_ids"])

    measurable = sum(1 for e in ingested if e["own_ids"])
    recall = found_any / measurable if measurable else 0.0
    precision = own_hits / total_hits if total_hits else 0.0
    return recall, precision


async def main() -> int:
    cases = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text())
    db = get_sessionmaker()()
    assert_eval_database(db)
    llm = get_llm_provider()

    db.execute(sql_text("TRUNCATE captures, memory_units, attachments, attachment_chunks"))
    db.commit()

    print(f"Ingesting all {len(cases)} cases once...")
    ingested = await ingest_all_cases(db, llm, cases)

    original_threshold = retrieval_pipeline.VECTOR_SIM_THRESHOLD
    print(f"\nOriginal VECTOR_SIM_THRESHOLD = {original_threshold}\n")
    print(f"{'threshold':>10} | {'recall':>8} | {'precision':>9}")
    print("-" * 34)
    rows = []
    for threshold in CANDIDATE_THRESHOLDS:
        recall, precision = await score_threshold(db, llm, ingested, threshold)
        rows.append((threshold, recall, precision))
        print(f"{threshold:>10.2f} | {recall:>8.2%} | {precision:>9.2%}")

    retrieval_pipeline.VECTOR_SIM_THRESHOLD = original_threshold

    db.execute(sql_text("TRUNCATE captures, memory_units, attachments, attachment_chunks"))
    db.commit()
    db.close()

    best = max(rows, key=lambda r: (r[1], r[2]))
    print(f"\nHighest recall+precision: threshold={best[0]} (recall={best[1]:.2%}, precision={best[2]:.2%})")
    print("See docs/DECISIONS.md for the chosen value and reasoning.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
