"""Iterative-recall evaluation harness (spec section 34/35): runs
eval/session_cases.yaml, a script of turns against a RecallSession per
case, checking the status transition sequence rather than a single
query -> answer round trip. Companion to eval/run_eval.py (which stays
focused on one-shot recall quality); run via scripts/run-session-eval.sh
against the same isolated `membase_eval` database.
"""

import asyncio
import sys
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import create_capture  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.retrieval.session import continue_session, start_session  # noqa: E402

TRUNCATE = "TRUNCATE captures, memory_units, attachments, attachment_chunks, recall_sessions"


async def run_case(db, llm, case: dict) -> tuple[bool, list[str]]:
    db.execute(sql_text(TRUNCATE))
    db.commit()

    for text in case.get("memories", []):
        await create_capture(db, llm, text=text, uploads=[])

    turns = case["turns"]
    notes: list[str] = []
    ok = True

    session = await start_session(db, llm, turns[0]["query"])
    for turn in turns:
        if turn is not turns[0]:
            session = await continue_session(db, llm, session, turn["message"])

        expect_status = turn.get("expect_status")
        if expect_status and session.status != expect_status:
            ok = False
            notes.append(f"expected status '{expect_status}', got '{session.status}'")

        expect_contains = turn.get("expect_contains")
        if expect_contains:
            answer = (session.final_answer or "").lower()
            missing = [p for p in expect_contains if p.lower() not in answer]
            if missing:
                ok = False
                notes.append(f"answer missing {missing}: got {session.final_answer!r}")

        expect_original_query = turn.get("expect_original_query")
        if expect_original_query and session.original_query != expect_original_query:
            ok = False
            notes.append(f"original_query drifted: {session.original_query!r}")

    notes.append(f"final status={session.status} answer={session.final_answer!r} question={session.current_question!r}")
    return ok, notes


async def main() -> int:
    cases = yaml.safe_load((Path(__file__).parent / "session_cases.yaml").read_text())
    db = get_sessionmaker()()
    llm = get_llm_provider()

    passed = 0
    for case in cases:
        ok, notes = await run_case(db, llm, case)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']} ({case['category']})")
        for note in notes:
            print(f"    {note}")
        passed += int(ok)

    db.execute(sql_text(TRUNCATE))
    db.commit()
    db.close()

    total = len(cases)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
