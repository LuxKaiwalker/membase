"""Latency benchmark for one-shot recall (run_agent). Not a quality
check (see eval/run_eval.py for that) -- this measures wall-clock time
per recall round against the real local model, and breaks it down by
which underlying LLM call (embed / evaluate_sufficiency / generate_answer)
the time actually went to, so a latency discussion has real numbers
behind it instead of call-counting alone.

Reuses eval/cases.yaml's corpus (ingested once, shared across all
queries -- same convention as eval/threshold_sweep.py) so retrieval is
searching a realistically-sized memory store, not a single isolated
case. Run via scripts/benchmark-recall-latency.sh (isolated
`membase_eval` database, same convention as the other eval scripts).
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from _helpers import assert_eval_database, create_capture_and_finish, finish_processing  # noqa: E402
from app.db import get_sessionmaker  # noqa: E402
from app.ingestion.pipeline import resolve_clarification  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.llm.base import LLMProvider  # noqa: E402
from app.retrieval.agent import run_agent  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"
TRUNCATE = "TRUNCATE captures, memory_units, attachments, attachment_chunks"


class TimingLLM(LLMProvider):
    """Wraps a real LLMProvider, recording (call_kind, seconds) for every
    call made through it -- the only way to see where a recall round's
    wall-clock time actually goes, since the agent's own call stack has
    no instrumentation and shouldn't be permanently modified just to
    answer this one question.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.calls: list[tuple[str, float]] = []

    def _kind_for_schema(self, schema: dict) -> str:
        props = schema.get("properties", {})
        if "sufficient" in props:
            return "evaluate_sufficiency"
        if "grounded" in props:
            return "generate_answer"
        return "generate_structured(other)"

    async def generate(self, prompt, *, system=None, temperature=0.2, images=None):
        start = time.perf_counter()
        try:
            return await self._inner.generate(prompt, system=system, temperature=temperature, images=images)
        finally:
            self.calls.append(("generate", time.perf_counter() - start))

    async def generate_structured(self, prompt, *, schema, system=None, images=None):
        kind = self._kind_for_schema(schema)
        start = time.perf_counter()
        try:
            return await self._inner.generate_structured(prompt, schema=schema, system=system, images=images)
        finally:
            self.calls.append((kind, time.perf_counter() - start))

    async def embed(self, texts):
        start = time.perf_counter()
        try:
            return await self._inner.embed(texts)
        finally:
            self.calls.append(("embed", time.perf_counter() - start))

    @property
    def embedding_model_name(self) -> str:
        return self._inner.embedding_model_name

    @property
    def embedding_dim(self) -> int:
        return self._inner.embedding_dim


async def ingest_corpus(db, llm, cases: list[dict]) -> None:
    for case in cases:
        captures = []
        own_units = []
        for text in case.get("memories", []):
            capture = await create_capture_and_finish(db, llm, text=text)
            captures.append(capture)
            own_units.extend(capture.units)
        if case.get("clarification_answer"):
            for unit in own_units:
                if unit.pending_clarification:
                    await resolve_clarification(db, llm, unit, case["clarification_answer"])
                    break
            # Answering a question can put a unit back into
            # status="processing" until enrichment finishes.
            await finish_processing(db, llm, *captures)
        distractor_captures = [
            await create_capture_and_finish(db, llm, text=text) for text in case.get("distractors", [])
        ]
        print(f"  ingested: {case['name']}", flush=True)


async def main() -> int:
    cases = yaml.safe_load(CASES_PATH.read_text())
    # Per-round LLM-call latency doesn't meaningfully depend on corpus
    # size (hybrid_search's own DB query is fast regardless; the slow
    # part is always the LLM calls) -- capping how many cases get
    # ingested keeps this benchmark's setup phase fast without making
    # the actual timing numbers any less representative. Defaults to 8
    # cases; set BENCHMARK_CASE_LIMIT=0 to use the full set.
    import os

    limit = int(os.environ.get("BENCHMARK_CASE_LIMIT", "8") or 0)
    if limit:
        cases = cases[:limit]
    db = get_sessionmaker()()
    assert_eval_database(db)
    db.execute(sql_text(TRUNCATE))
    db.commit()

    real_llm = get_llm_provider()
    print(f"Ingesting {len(cases)} cases into a shared corpus...", flush=True)
    await ingest_corpus(db, real_llm, cases)
    db.expire_all()
    print("Ingestion done.", flush=True)

    per_query: list[dict] = []
    for case in cases:
        llm = TimingLLM(real_llm)
        start = time.perf_counter()
        result = await run_agent(db, llm, case["query"])
        total = time.perf_counter() - start
        by_kind: dict[str, float] = {}
        by_kind_count: dict[str, int] = {}
        for kind, seconds in llm.calls:
            by_kind[kind] = by_kind.get(kind, 0.0) + seconds
            by_kind_count[kind] = by_kind_count.get(kind, 0) + 1
        per_query.append(
            {
                "name": case["name"],
                "total_s": total,
                "abstained": result.abstained,
                "call_count": len(llm.calls),
                "by_kind": by_kind,
                "by_kind_count": by_kind_count,
            }
        )
        print(
            f"  {case['name']:40s} total={total:6.2f}s  calls={len(llm.calls):2d}  "
            f"abstained={result.abstained}",
            flush=True,
        )

    totals = [q["total_s"] for q in per_query]
    print("\n=== Summary ===")
    print(f"n = {len(totals)}")
    print(f"mean   = {statistics.mean(totals):.2f}s")
    print(f"median = {statistics.median(totals):.2f}s")
    print(f"p90    = {sorted(totals)[int(len(totals) * 0.9)]:.2f}s")
    print(f"max    = {max(totals):.2f}s")
    print(f"min    = {min(totals):.2f}s")
    over_10s = sum(1 for t in totals if t > 10)
    print(f"rounds over 10s = {over_10s}/{len(totals)}")

    kind_totals: dict[str, float] = {}
    kind_counts: dict[str, int] = {}
    for q in per_query:
        for kind, seconds in q["by_kind"].items():
            kind_totals[kind] = kind_totals.get(kind, 0.0) + seconds
            kind_counts[kind] = kind_counts.get(kind, 0) + q["by_kind_count"][kind]
    print("\n=== Time by call kind (summed across all queries) ===")
    grand_total = sum(kind_totals.values())
    for kind, seconds in sorted(kind_totals.items(), key=lambda kv: -kv[1]):
        pct = 100 * seconds / grand_total if grand_total else 0
        print(f"  {kind:28s} total={seconds:7.2f}s  calls={kind_counts[kind]:3d}  avg={seconds / kind_counts[kind]:.2f}s  ({pct:.0f}% of LLM time)")

    import json

    out_path = Path(__file__).resolve().parent / "recall_latency_results.json"
    out_path.write_text(json.dumps(per_query, indent=2))
    print(f"\nRaw per-query results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
