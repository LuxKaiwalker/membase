"""Interactive-retrieval benchmark: does interactive (ask-a-clarifying-
question) retrieval recover information that single-shot retrieval
cannot efficiently recover? Compares three modes against one shared,
deliberately ambiguous synthetic corpus (eval/interactive_benchmark.yaml):

  A. semantic baseline -- one hybrid_search() call, no reformulation,
     no reasoning about sufficiency.
  B. LLM-assisted       -- the existing bounded one-shot agent
     (run_agent): search -> evaluate -> reformulate -> answer, no user
     interaction.
  C. interactive        -- the existing RecallSession: search ->
     ambiguity check -> ask -> (simulated) answer -> search again -> ...

A and B reuse real production retrieval primitives unmodified (A is a
deliberately crippled baseline assembled here, not production code --
production has no reason to offer a non-iterating search path). C runs
the actual RecallSession controller; its clarifying questions are
answered by a *simulated* user (see simulate_reply below), whose replies
are still fed through the real extract_cues() LLM call, not a shortcut.

Run via scripts/run-interactive-benchmark.sh (isolated `membase_eval`
database, same convention as the other eval scripts).
"""

import asyncio
import math
import re
import sys
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import create_capture  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.llm.base import LLMProvider  # noqa: E402
from app.retrieval.agent import automatic_search, run_agent  # noqa: E402
from app.retrieval.discriminate import _ATTRIBUTE_TEMPLATES, apply_positive_cue_boost  # noqa: E402
from app.retrieval.pipeline import (  # noqa: E402
    ABSTENTION_MESSAGE,
    TOP_K_FOR_ANSWER,
    generate_answer,
    hybrid_search,
    sibling_context,
)
from app.retrieval.session import continue_session, start_session  # noqa: E402

TRUNCATE = "TRUNCATE captures, memory_units, attachments, attachment_chunks, recall_sessions"
MAX_SIMULATED_TURNS = 5

# ---------------------------------------------------------------------------
# Simulated user: answers a RecallSession's own generated clarifying
# question by checking whether its templated attribute value matches the
# *actual* (post-ingestion) semantic metadata of the query's designated
# ground-truth memory -- never a hand-authored Q&A script, so it works
# regardless of exactly how the heuristic phrases the question.
# ---------------------------------------------------------------------------

def _compile_template(template: str) -> re.Pattern:
    escaped = re.escape(template).replace(re.escape("{value}"), "(?P<value>.+)")
    return re.compile(f"^{escaped}$")


_TEMPLATE_PATTERNS = {field: _compile_template(t) for field, t in _ATTRIBUTE_TEMPLATES.items()}


def simulate_reply(question: str, ground_truth_semantic: dict) -> str:
    for field, pattern in _TEMPLATE_PATTERNS.items():
        m = pattern.match(question)
        if not m:
            continue
        value = m.group("value")
        raw = ground_truth_semantic.get(field)
        truth_values = [raw] if field == "event_date" else (raw or [])
        truth_values = [v.lower() for v in truth_values if isinstance(v, str)]
        if value.strip().lower() in truth_values:
            return f"Yes, that's right -- it was about {value}."
        return f"No, that's not it. It had nothing to do with {value}."
    # Heuristic produced something outside the known templates (shouldn't
    # happen given discriminate.py's fixed template set, but degrade to a
    # still-grounded hint rather than crash the benchmark).
    meaning = ground_truth_semantic.get("meaning") or ""
    return f"Not quite. {meaning}".strip()


def entropy(dist: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


async def candidate_distribution(db, llm: LLMProvider, query: str, positive_cues: list[str]) -> dict[str, float]:
    """Best-effort probability distribution over candidate origin-capture
    groups, for the entropy/information-gain metric (not a calibrated
    model -- normalized fused scores, computed from the same retrieval
    signal the session itself uses, documented as an approximation in the
    report)."""
    result = await automatic_search(db, llm, query, extra_queries=positive_cues)
    ranked = apply_positive_cue_boost(result.ranked, positive_cues)
    groups: dict[str, float] = {}
    for s in ranked:
        cid = str(s.unit.capture_id)
        groups[cid] = max(groups.get(cid, 0.0), s.fused_score)
    total = sum(groups.values())
    if total <= 0:
        return {}
    return {cid: v / total for cid, v in groups.items()}


# ---------------------------------------------------------------------------
# Three retrieval modes
# ---------------------------------------------------------------------------

async def run_mode_a(db, llm: LLMProvider, query: str) -> dict:
    """Naive semantic baseline: one retrieval pass, no reformulation, no
    sufficiency reasoning -- "retrieve top-k -> LLM answer"."""
    ranked = await hybrid_search(db, llm, query, limit=TOP_K_FOR_ANSWER)
    if not ranked:
        return {"answer": ABSTENTION_MESSAGE, "grounded": False, "sources": ranked, "questions_asked": 0}
    context = sibling_context(db, ranked)
    answer_text, grounded = await generate_answer(llm, query, ranked, context, [])
    return {
        "answer": answer_text if grounded else ABSTENTION_MESSAGE,
        "grounded": grounded,
        "sources": ranked,
        "questions_asked": 0,
    }


async def run_mode_b(db, llm: LLMProvider, query: str) -> dict:
    """LLM-assisted retrieval: the existing bounded one-shot agent
    (search -> evaluate -> reformulate -> answer), no user interaction."""
    result = await run_agent(db, llm, query)
    return {
        "answer": result.answer,
        "grounded": not result.abstained,
        "sources": result.sources,
        "questions_asked": 0,
    }


async def run_mode_c(db, llm: LLMProvider, query: str, ground_truth_semantic: dict) -> dict:
    """Interactive retrieval: the real RecallSession, with a simulated
    user answering whatever clarifying question it actually asks."""
    session = await start_session(db, llm, query)
    questions: list[str] = []
    info_gains: list[float] = []
    positive_cues: list[str] = []

    while session.status == "needs_user_cue" and len(questions) < MAX_SIMULATED_TURNS:
        question = session.current_question
        questions.append(question)

        dist_before = await candidate_distribution(db, llm, query, positive_cues)
        h_before = entropy(dist_before)

        reply = simulate_reply(question, ground_truth_semantic)
        session = await continue_session(db, llm, session, reply)

        positive_cues = [c["text"] for c in session.cues if c["polarity"] == "positive"]
        dist_after = await candidate_distribution(db, llm, query, positive_cues)
        h_after = entropy(dist_after)
        info_gains.append(h_before - h_after)

    return {
        "answer": session.final_answer or ABSTENTION_MESSAGE,
        "grounded": session.status == "success",
        "sources": session.final_sources,  # list[dict] snapshot, not ScoredUnit
        "questions_asked": len(questions),
        "questions": questions,
        "info_gains": info_gains,
        "status": session.status,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def source_memory_ids(sources, mode: str, capture_to_memory: dict[str, str]) -> set[str]:
    ids = []
    for s in sources:
        cid = s["capture_id"] if mode == "c" else str(s.unit.capture_id)
        ids.append(cid)
    return {capture_to_memory[cid] for cid in ids if cid in capture_to_memory}


def score(result: dict, case: dict, mode: str, capture_to_memory: dict[str, str]) -> bool:
    if not result["grounded"]:
        return False
    if "expect_contains" in case:
        answer = (result["answer"] or "").lower()
        return any(p.lower() in answer for p in case["expect_contains"])
    if "expect_include_any" in case:
        mem_ids = source_memory_ids(result["sources"], mode, capture_to_memory)
        included = any(mid in mem_ids for mid in case["expect_include_any"])
        excluded = any(mid in mem_ids for mid in case.get("expect_exclude", []))
        return included and not excluded
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def ingest_corpus(db, llm: LLMProvider, memories: list[dict]) -> tuple[dict[str, str], dict[str, dict]]:
    """Returns (capture_id -> memory_id, memory_id -> semantic) after
    ingesting each memory as its own independent capture."""
    capture_to_memory: dict[str, str] = {}
    semantics: dict[str, dict] = {}
    for mem in memories:
        capture = await create_capture(db, llm, text=mem["text"], uploads=[])
        capture_to_memory[str(capture.id)] = mem["id"]
        # A short atomic sentence should segment into exactly one unit;
        # fall back to merging all units' semantic fields just in case.
        merged: dict = {}
        for unit in capture.units:
            for k, v in (unit.semantic or {}).items():
                if isinstance(v, list):
                    merged.setdefault(k, [])
                    merged[k].extend(v)
                elif k not in merged or not merged.get(k):
                    merged[k] = v
        semantics[mem["id"]] = merged
    return capture_to_memory, semantics


def print_report(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'ID':<5}{'Lvl':<4}{'Category':<24}{'A':<4}{'B':<4}{'C':<4}{'C-Qs':<6}{'C-IG':<8}")
    print("-" * 100)
    for r in rows:
        ig = f"{sum(r['c_info_gains']) / len(r['c_info_gains']):.2f}" if r["c_info_gains"] else "-"
        print(
            f"{r['id']:<5}{r['level']:<4}{r['category']:<24}"
            f"{'OK' if r['a'] else '--':<4}{'OK' if r['b'] else '--':<4}{'OK' if r['c'] else '--':<4}"
            f"{r['c_questions']:<6}{ig:<8}"
        )
    print("=" * 100)

    total = len(rows)
    for mode in ("a", "b", "c"):
        acc = sum(r[mode] for r in rows) / total
        print(f"Overall accuracy, mode {mode.upper()}: {acc:.0%} ({sum(r[mode] for r in rows)}/{total})")

    print("\nAccuracy by level (A / B / C):")
    levels = sorted({r["level"] for r in rows})
    for lvl in levels:
        subset = [r for r in rows if r["level"] == lvl]
        n = len(subset)
        a = sum(r["a"] for r in subset) / n
        b = sum(r["b"] for r in subset) / n
        c = sum(r["c"] for r in subset) / n
        print(f"  Level {lvl} ({subset[0]['category']}, n={n}): A={a:.0%}  B={b:.0%}  C={c:.0%}")

    level1 = [r for r in rows if r["level"] == 1]
    l1_questions = sum(r["c_questions"] for r in level1)
    print(
        f"\nEfficiency check (should be 0): mode C asked {l1_questions} clarifying "
        f"question(s) across {len(level1)} Level-1 (direct) queries."
    )

    improved = [r for r in rows if r["c"] and not r["b"] and r["c_questions"] > 0]
    print(
        f"\nQueries where C succeeded, B failed, and C actually asked a question: "
        f"{len(improved)}/{total} ({', '.join(r['id'] for r in improved) or 'none'})"
    )

    all_ig = [ig for r in rows for ig in r["c_info_gains"]]
    if all_ig:
        print(
            f"\nAverage information gain per clarifying question (approximate, "
            f"normalized-fused-score entropy, bits): {sum(all_ig) / len(all_ig):.3f}"
        )


async def main() -> int:
    data = yaml.safe_load((Path(__file__).parent / "interactive_benchmark.yaml").read_text())
    db = get_sessionmaker()()
    llm = get_llm_provider()

    db.execute(sql_text(TRUNCATE))
    db.commit()

    print(f"Ingesting {len(data['memories'])} memories as independent captures...")
    capture_to_memory, semantics = await ingest_corpus(db, llm, data["memories"])

    rows = []
    for case in data["queries"]:
        query = case["question"]
        print(f"\n[{case['id']}] {query}")

        res_a = await run_mode_a(db, llm, query)
        res_b = await run_mode_b(db, llm, query)
        ground_truth = semantics[case["ground_truth_memory"]]
        res_c = await run_mode_c(db, llm, query, ground_truth)

        ok_a = score(res_a, case, "a", capture_to_memory)
        ok_b = score(res_b, case, "b", capture_to_memory)
        ok_c = score(res_c, case, "c", capture_to_memory)

        print(f"    A ({'OK' if ok_a else 'MISS'}): {res_a['answer']}")
        print(f"    B ({'OK' if ok_b else 'MISS'}): {res_b['answer']}")
        print(f"    C ({'OK' if ok_c else 'MISS'}, {res_c['questions_asked']} q): {res_c['answer']}")
        for q in res_c.get("questions", []):
            print(f"        C asked: {q}")

        rows.append(
            {
                "id": case["id"],
                "level": case["level"],
                "category": case["category"],
                "a": ok_a,
                "b": ok_b,
                "c": ok_c,
                "c_questions": res_c["questions_asked"],
                "c_info_gains": res_c.get("info_gains", []),
            }
        )

    print_report(rows)

    db.execute(sql_text(TRUNCATE))
    db.commit()
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
