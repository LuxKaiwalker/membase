"""Small evaluation harness (spec section 21). Not a substitute for the
pytest suite -- this exercises retrieval+grounding *quality* against the
real local LLM, with hand-written cases across the categories the spec
calls out. Run via scripts/run-eval.sh, which points it at a dedicated
`membase_eval` database so it never touches real stored memories.
"""

import asyncio
import sys
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import create_memory  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.retrieval.pipeline import answer  # noqa: E402


async def run_case(db, llm, case: dict) -> tuple[bool, str]:
    db.execute(sql_text("TRUNCATE memories, attachments"))
    db.commit()
    for caption in [*case.get("memories", []), *case.get("distractors", [])]:
        await create_memory(db, llm, caption=caption, uploads=[])

    result = await answer(db, llm, case["query"])

    if case["expect"] == "abstain":
        return result.abstained, result.answer

    expect_contains = case.get("expect_contains", [])
    ok = (not result.abstained) and all(
        phrase.lower() in result.answer.lower() for phrase in expect_contains
    )
    return ok, result.answer


async def main() -> int:
    cases = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text())
    db = get_sessionmaker()()
    llm = get_llm_provider()

    passed = 0
    for case in cases:
        ok, detail = await run_case(db, llm, case)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']} ({case['category']})")
        print(f"    query:  {case['query']}")
        print(f"    answer: {detail}")
        passed += int(ok)

    db.execute(sql_text("TRUNCATE memories, attachments"))
    db.commit()
    db.close()

    total = len(cases)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
