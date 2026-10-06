"""Small evaluation harness (spec section 21/22). Not a substitute for the
pytest suite -- this exercises segmentation + clarification + retrieval +
grounding *quality* against the real local LLM, with hand-written cases
across the categories the spec calls out. Run via scripts/run-eval.sh,
which points it at a dedicated `membase_eval` database so it never
touches real stored memories.
"""

import asyncio
import sys
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import create_capture, resolve_clarification  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.retrieval.agent import run_agent  # noqa: E402


async def run_case(db, llm, case: dict) -> tuple[bool, str, str]:
    """Returns (ok, detail, segmentation_note)."""
    db.execute(sql_text("TRUNCATE captures, memory_units, attachments, attachment_chunks"))
    db.commit()

    own_units = []
    for text in case.get("memories", []):
        capture = await create_capture(db, llm, text=text, uploads=[])
        own_units.extend(capture.units)

    if case.get("clarification_answer"):
        for unit in own_units:
            if unit.pending_clarification:
                await resolve_clarification(db, llm, unit, case["clarification_answer"])
                break

    for text in case.get("distractors", []):
        await create_capture(db, llm, text=text, uploads=[])

    segmentation_note = f"{len(own_units)} unit(s)"
    expect_units = case.get("expect_units")
    if expect_units and not (expect_units[0] <= len(own_units) <= expect_units[1]):
        segmentation_note += f" (expected {expect_units[0]}-{expect_units[1]})"

    result = await run_agent(db, llm, case["query"])

    if case["expect"] == "abstain":
        return result.abstained, result.answer, segmentation_note

    expect_contains = case.get("expect_contains", [])
    ok = (not result.abstained) and all(
        phrase.lower() in result.answer.lower() for phrase in expect_contains
    )
    return ok, result.answer, segmentation_note


async def main() -> int:
    cases = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text())
    db = get_sessionmaker()()
    llm = get_llm_provider()

    passed = 0
    for case in cases:
        ok, detail, seg_note = await run_case(db, llm, case)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']} ({case['category']})")
        print(f"    query:  {case['query']}")
        print(f"    answer: {detail}")
        if case.get("expect_units"):
            print(f"    segmentation: {seg_note}")
        passed += int(ok)

    db.execute(sql_text("TRUNCATE captures, memory_units, attachments, attachment_chunks"))
    db.commit()
    db.close()

    total = len(cases)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
