"""Instrumented re-run of the interactive-retrieval benchmark, for
visualization/debugging rather than scoring: captures the full
round-by-round internal state of mode C (candidate scores per origin
capture, entropy before/after, which semantic field each question and
simulated reply matched) and dumps it to eval/trace_output.json.

Reuses every helper from run_interactive_benchmark.py unmodified --
this script only adds observation, never changes retrieval behavior.
"""

import asyncio
import json
import sys
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.db import get_sessionmaker  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.retrieval.discriminate import _CUE_MATCH_FIELDS, apply_negative_cues, values_for_field  # noqa: E402
from app.retrieval.session import continue_session, start_session  # noqa: E402
from run_interactive_benchmark import (  # noqa: E402
    MAX_SIMULATED_TURNS,
    TRUNCATE,
    _TEMPLATE_PATTERNS,
    candidate_distribution,
    entropy,
    ingest_corpus,
    run_mode_a,
    run_mode_b,
    score,
    simulate_reply,
)


def _match_question_template(question: str) -> tuple[str | None, str | None]:
    for field, pattern in _TEMPLATE_PATTERNS.items():
        m = pattern.match(question)
        if m:
            return field, m.group("value")
    return None, None


def _distribution_rows(dist: dict[str, float], capture_to_memory: dict[str, str]) -> list[dict]:
    rows = [{"memory_id": capture_to_memory.get(cid, cid[:8]), "score": score} for cid, score in dist.items()]
    return sorted(rows, key=lambda r: r["score"], reverse=True)


async def run_mode_c_traced(db, llm, query: str, ground_truth_semantic: dict, capture_to_memory: dict) -> dict:
    session = await start_session(db, llm, query)
    rounds = []
    positive_cues: list[str] = []
    negative_cues: list[str] = []

    while session.status == "needs_user_cue" and len(rounds) < MAX_SIMULATED_TURNS:
        question = session.current_question
        field, value = _match_question_template(question)

        dist_before = await candidate_distribution(db, llm, query, positive_cues)
        h_before = entropy(dist_before)

        reply = simulate_reply(question, ground_truth_semantic)
        matched_truth = None
        if field is not None:
            truth_values = [v.lower() for v in values_for_field(ground_truth_semantic, field)]
            matched_truth = value.strip().lower() in truth_values

        session = await continue_session(db, llm, session, reply)

        positive_cues = [c["text"] for c in session.cues if c["polarity"] == "positive"]
        negative_cues = [c["text"] for c in session.cues if c["polarity"] == "negative"]

        dist_after = await candidate_distribution(db, llm, query, positive_cues)
        h_after = entropy(dist_after)

        rounds.append(
            {
                "round": len(rounds) + 1,
                "candidates_before": _distribution_rows(dist_before, capture_to_memory),
                "entropy_before": h_before,
                "question": question,
                "question_field": field,
                "question_value": value,
                "simulated_reply": reply,
                "reply_matched_ground_truth_field": matched_truth,
                "candidates_after": _distribution_rows(dist_after, capture_to_memory),
                "entropy_after": h_after,
                "info_gain": h_before - h_after,
                "session_status_after": session.status,
            }
        )

    return {
        "final_status": session.status,
        "final_answer": session.final_answer,
        "final_sources": session.final_sources,
        "rounds": rounds,
        "all_cues": session.cues,
        "positive_cues": positive_cues,
        "negative_cues": negative_cues,
    }


async def field_coverage_case(db, llm, capture_to_memory: dict) -> dict:
    """The Q3/Orion anomaly, captured directly: where does the target
    entity actually live across all semantic fields, vs. which fields
    apply_negative_cues/apply_positive_cue_boost actually check? Field
    lists come straight from discriminate.py rather than being
    hand-duplicated here, so this stays accurate if that module's
    coverage ever changes again (it was exactly this kind of silent
    drift -- a hardcoded copy of the field list going stale after a fix
    -- that let the original bug go unnoticed)."""
    from app.models import Capture

    orion_id = next(cid for cid, mid in capture_to_memory.items() if mid == "M2")
    capture = db.get(Capture, orion_id)
    unit = capture.units[0]
    semantic = unit.semantic

    checked_fields = list(_CUE_MATCH_FIELDS)
    unchecked_fields = ["meaning"]  # free text, never matched against by design

    target = "orion"
    all_fields = checked_fields + unchecked_fields
    presence = {}
    for f in all_fields:
        values = values_for_field(semantic, f) if f != "meaning" else ([semantic["meaning"]] if semantic.get("meaning") else [])
        presence[f] = {
            "values": values,
            "contains_target": any(target in str(v).lower() for v in values),
        }

    return {
        "memory_id": "M2",
        "text": capture.raw_text,
        "target_value": "Orion",
        "checked_by_negative_cues": checked_fields,
        "checked_by_positive_boost": checked_fields,
        "never_checked": unchecked_fields,
        "field_presence": presence,
    }


async def main() -> int:
    data = yaml.safe_load((Path(__file__).parent / "interactive_benchmark.yaml").read_text())
    db = get_sessionmaker()()
    llm = get_llm_provider()

    db.execute(sql_text(TRUNCATE))
    db.commit()

    print(f"Ingesting {len(data['memories'])} memories...", flush=True)
    capture_to_memory, semantics = await ingest_corpus(db, llm, data["memories"])

    coverage = await field_coverage_case(db, llm, capture_to_memory)
    print("Field coverage case captured.", flush=True)

    results = []
    for case in data["queries"]:
        query = case["question"]
        print(f"[{case['id']}] {query}", flush=True)

        res_a = await run_mode_a(db, llm, query)
        res_b = await run_mode_b(db, llm, query)
        ground_truth = semantics[case["ground_truth_memory"]]
        traced_c = await run_mode_c_traced(db, llm, query, ground_truth, capture_to_memory)

        ok_a = score(res_a, case, "a", capture_to_memory)
        ok_b = score(res_b, case, "b", capture_to_memory)
        res_c_for_scoring = {
            "answer": traced_c["final_answer"],
            "grounded": traced_c["final_status"] == "success",
            "sources": traced_c["final_sources"],  # list[dict], same shape as session.final_sources
        }
        ok_c = score(res_c_for_scoring, case, "c", capture_to_memory)

        print(f"    A={'OK' if ok_a else 'MISS'} B={'OK' if ok_b else 'MISS'} C={'OK' if ok_c else 'MISS'} ({len(traced_c['rounds'])} rounds)", flush=True)

        results.append(
            {
                "id": case["id"],
                "level": case["level"],
                "category": case["category"],
                "question": query,
                "ground_truth_memory": case["ground_truth_memory"],
                "candidates": case.get("candidates", []),
                "mode_a": {"answer": res_a["answer"], "correct": ok_a},
                "mode_b": {"answer": res_b["answer"], "correct": ok_b},
                "mode_c": {**traced_c, "correct": ok_c},
            }
        )

    output = {"memories": data["memories"], "results": results, "field_coverage_case": coverage}
    out_path = Path(__file__).parent / "trace_output.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote trace to {out_path}")

    db.execute(sql_text(TRUNCATE))
    db.commit()
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
