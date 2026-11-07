"""Multi-strategy retrieval benchmark: do associative and iterative recall
actually beat conventional retrieval on the *same* stored memories?

Four strategies run read-only against one shared corpus
(eval/memory_corpus.yaml), ingested once through the real pipeline, with one
query set (eval/strategy_queries.yaml):

  1. vector          -- pure pgvector cosine top-k, no full-text, no LLM
                        reasoning, one generate_answer call. The conventional
                        baseline, assembled here (see run_vector below for why
                        it is a fair "vector similarity only" baseline).
  2. semantic_rerank -- the existing one-shot agent run_agent(): hybrid
                        vector+FTS fusion plus evaluate_sufficiency-driven
                        reformulation. No graph, no clarification.
  3. associative     -- automatic_search() seeds, then the level-6 engine
                        (app/retrieval/associative.py): expand_with_associations
                        + collect_evidence + rerank, answered from
                        format_evidence_block. Multi-hop, but single-shot: it
                        never asks the user anything.
  4. adaptive        -- the full RecallSession (start_session /
                        continue_session) driven by a simulated user, so
                        clarifying questions are in play.

Design borrowed wholesale from eval/run_interactive_benchmark.py (uniform
per-strategy coroutine, one shared corpus, one scorer, one report, a simulated
user that reverse-maps the generated question against _ATTRIBUTE_TEMPLATES and
answers from the ground-truth memory's real post-ingestion semantic fields),
with one deliberate change: each strategy normalises its own result into a
uniform dict -- including `retrieved_memory_ids` as a plain ranked list of
corpus ids -- so the scorer never branches on which strategy produced it. In
the precedent that branch (`source_memory_ids(..., mode, ...)`) was a wart.

Corpus caching
--------------
Seeding is ~6 generation calls + 1 embed per memory, roughly 20-25s of model
time each (docs/DECISIONS.md "Benchmark: fast pipeline vs. structurer
wall-clock time"), so a 44-memory corpus is 15-25 minutes. Nothing else in
eval/ caches that; this does. See `load_or_seed_corpus`.

Run via scripts/run-strategy-benchmark.sh (isolated `membase_eval` database,
same convention as the other eval scripts). `--self-check` validates the two
YAML files and exits without touching the database or the model.
"""

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from sqlalchemy import text as sql_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from _helpers import assert_eval_database, create_capture_and_finish  # noqa: E402
from app.db import get_sessionmaker  # noqa: E402
from app.llm import get_llm_provider  # noqa: E402
from app.llm.base import LLMProvider  # noqa: E402
from app.models import MemoryUnit  # noqa: E402
from app.retrieval.agent import automatic_search, run_agent  # noqa: E402
from app.retrieval.associative import (  # noqa: E402
    RetrievalBudget,
    apply_budget,
    build_direct_candidates,
    collect_evidence,
    expand_with_associations,
    rerank,
    should_expand,
)
from app.retrieval.discriminate import _ATTRIBUTE_TEMPLATES, values_for_field  # noqa: E402
from app.retrieval.evidence import format_evidence_block  # noqa: E402
from app.retrieval.pipeline import (  # noqa: E402
    ABSTENTION_MESSAGE,
    TOP_K_FOR_ANSWER,
    VECTOR_SIM_THRESHOLD,
    ScoredUnit,
    embed_query,
    generate_answer,
    sibling_context,
)
from app.retrieval.session import continue_session, start_session  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS_PATH = HERE / "memory_corpus.yaml"
QUERIES_PATH = HERE / "strategy_queries.yaml"
STATE_PATH = HERE / ".corpus_state.json"
RESULTS_PATH = HERE / "strategy_benchmark_results.json"

# Every table that holds corpus-derived state. CASCADE because the
# associative-graph projection tables (memory_entities / unit_entity_links /
# entity_relations / unit_associations) and episodes reference memory_units,
# and a bare TRUNCATE of a referenced table is rejected by Postgres.
TRUNCATE = (
    "TRUNCATE captures, memory_units, attachments, attachment_chunks, recall_sessions, "
    "memory_entities, unit_entity_links, entity_relations, unit_associations, "
    "episodes, episode_memberships CASCADE"
)

MAX_SIMULATED_TURNS = 5

STRATEGIES = ("vector", "semantic_rerank", "associative", "adaptive")

CATEGORIES = (
    "exact",
    "semantic",
    "indirect_association",
    "multi_hop",
    "temporal",
    "ambiguous",
    "vague_episodic",
    "unanswerable",
)

MIN_QUERIES = 32
MIN_UNANSWERABLE = 3
CORPUS_SIZE_RANGE = (36, 44)


# ---------------------------------------------------------------------------
# Self-check: a dangling memory id must fail loudly, not silently score 0
# ---------------------------------------------------------------------------


def self_check(corpus: dict, query_data: dict) -> list[str]:
    """Structural validation of the two YAML files. Returns a list of
    problems; an empty list means the fixtures are internally consistent.

    This exists because the failure mode it catches is invisible otherwise: a
    query pointing at a memory id that does not exist in the corpus simply
    scores zero recall for every strategy, which looks like a retrieval
    finding rather than a typo.
    """
    problems: list[str] = []

    memories = corpus.get("memories") or []
    ids: list[str] = [m.get("id") for m in memories]
    id_set = set(ids)
    if len(ids) != len(id_set):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"corpus: duplicate memory ids {dupes}")
    if not CORPUS_SIZE_RANGE[0] <= len(memories) <= CORPUS_SIZE_RANGE[1]:
        problems.append(
            f"corpus: {len(memories)} memories, expected "
            f"{CORPUS_SIZE_RANGE[0]}-{CORPUS_SIZE_RANGE[1]}"
        )
    for m in memories:
        if not m.get("theme"):
            problems.append(f"corpus: {m.get('id')} has no theme tag")
        if not (m.get("text") or "").strip():
            problems.append(f"corpus: {m.get('id')} has no text")

    queries = query_data.get("queries") or []
    qids = [q.get("id") for q in queries]
    if len(qids) != len(set(qids)):
        problems.append("queries: duplicate query ids")
    if len(queries) < MIN_QUERIES:
        problems.append(f"queries: {len(queries)} queries, expected at least {MIN_QUERIES}")

    seen_categories: dict[str, int] = {}
    unanswerable = 0
    for q in queries:
        qid = q.get("id")
        category = q.get("category")
        if category not in CATEGORIES:
            problems.append(f"{qid}: category {category!r} not in {CATEGORIES}")
        seen_categories[category] = seen_categories.get(category, 0) + 1
        if not (q.get("question") or "").strip():
            problems.append(f"{qid}: empty question")

        if q.get("expect") == "abstain":
            unanswerable += 1
            if q.get("relevant_memories") or q.get("ground_truth_memory"):
                problems.append(f"{qid}: unanswerable queries must have no ground truth")
            if category != "unanswerable":
                problems.append(f"{qid}: expect=abstain but category is {category!r}")
            continue

        if category == "unanswerable":
            problems.append(f"{qid}: category unanswerable but no 'expect: abstain'")

        relevant = q.get("relevant_memories") or []
        if not relevant:
            problems.append(f"{qid}: answerable query has no relevant_memories")
        for mid in relevant:
            if mid not in id_set:
                problems.append(f"{qid}: relevant_memories references unknown memory {mid!r}")
        gt = q.get("ground_truth_memory")
        if not gt:
            problems.append(f"{qid}: answerable query has no ground_truth_memory")
        elif gt not in id_set:
            problems.append(f"{qid}: ground_truth_memory references unknown memory {gt!r}")
        elif relevant and gt not in relevant:
            problems.append(f"{qid}: ground_truth_memory {gt} is not in its own relevant_memories")
        if not q.get("expect_contains"):
            problems.append(f"{qid}: answerable query has no expect_contains")

    if unanswerable < MIN_UNANSWERABLE:
        problems.append(
            f"queries: {unanswerable} unanswerable queries, expected at least {MIN_UNANSWERABLE} "
            "(they are the only thing that makes false retrieval measurable)"
        )
    for category in CATEGORIES:
        if not seen_categories.get(category):
            problems.append(f"queries: category {category!r} has no queries")

    return problems


# ---------------------------------------------------------------------------
# Per-strategy LLM cost accounting
# ---------------------------------------------------------------------------


class TimingLLM(LLMProvider):
    """Wraps the real provider, recording (call_kind, seconds) per call.

    Same decorator pattern as eval/benchmark_recall_latency.py's TimingLLM --
    the only way to get a per-strategy cost axis without instrumenting the
    retrieval code itself. Kind is inferred from the response schema's
    properties, which is what distinguishes the calls the strategies actually
    differ in (sufficiency reasoning vs. cue extraction vs. answering).
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
        if "cues" in props:
            return "extract_cues"
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

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.calls if k == kind)


# ---------------------------------------------------------------------------
# Simulated user (adaptive strategy only)
# ---------------------------------------------------------------------------


def _compile_template(template: str) -> re.Pattern:
    escaped = re.escape(template).replace(re.escape("{value}"), "(?P<value>.+)")
    return re.compile(f"^{escaped}$")


_TEMPLATE_PATTERNS = {field_: _compile_template(t) for field_, t in _ATTRIBUTE_TEMPLATES.items()}


def simulate_reply(question: str, ground_truth_semantic: dict | None) -> str:
    """Answer the session's own generated clarifying question from the *real*
    post-ingestion semantic metadata of the query's designated ground-truth
    memory -- never a hand-authored Q&A script, so it keeps working whatever
    phrasing the discrimination heuristic picks.

    Identical to eval/run_interactive_benchmark.py's oracle except for the
    `None` case, which that benchmark has no equivalent of: an unanswerable
    query has no ground truth, so the honest simulated user says it does not
    know rather than inventing a cue that would help narrow a search for a
    memory that is not there.
    """
    if ground_truth_semantic is None:
        return "I'm not sure, I don't remember any of that."
    for field_, pattern in _TEMPLATE_PATTERNS.items():
        m = pattern.match(question)
        if not m:
            continue
        value = m.group("value")
        truth_values = [v.lower() for v in values_for_field(ground_truth_semantic, field_)]
        if value.strip().lower() in truth_values:
            return f"Yes, that's right -- it was about {value}."
        return f"No, that's not it. It had nothing to do with {value}."
    meaning = (ground_truth_semantic or {}).get("meaning") or ""
    return f"Not quite. {meaning}".strip()


# ---------------------------------------------------------------------------
# Corpus seeding + caching
# ---------------------------------------------------------------------------


def corpus_fingerprint(raw_yaml: str, limit: int | None) -> str:
    """SHA256 over the corpus YAML *and* the limit.

    The limit is part of the fingerprint deliberately: without it, an 8-memory
    `--limit 8` smoke corpus would satisfy a cache check written by a full
    44-memory run (and vice versa), and a limited run would silently
    masquerade as a full one in the report.
    """
    h = hashlib.sha256()
    h.update(raw_yaml.encode("utf-8"))
    h.update(f"\nlimit={limit if limit else 'all'}\n".encode())
    return h.hexdigest()


def ready_unit_count(db) -> int:
    return int(db.execute(sql_text("SELECT count(*) FROM memory_units WHERE status = 'ready'")).scalar() or 0)


def graph_stats(db) -> dict:
    """Association-graph size, plus how much of the corpus has degraded
    semantics.

    Printed because both numbers can silently invalidate a run. An empty graph
    makes the `associative` strategy look like it adds nothing, when in fact
    there was nothing to walk. And a unit whose enrichment never completed
    (`enrichment_status == "unavailable"`, which happens when the model was
    unavailable or timed out during ingestion -- realistic on this box, where
    docs/RECALL_LATENCY.md already documents Ollama under contention) was
    matched against incomplete information, is penalised by `rerank`, and
    changes `should_expand`'s verdict. Either one is grounds to `--reseed`
    rather than to believe the numbers.
    """
    degraded = int(
        db.execute(
            sql_text(
                "SELECT count(*) FROM memory_units "
                "WHERE status = 'ready' AND semantic->>'enrichment_status' = 'unavailable'"
            )
        ).scalar()
        or 0
    )
    return {
        "entities": int(db.execute(sql_text("SELECT count(*) FROM memory_entities")).scalar() or 0),
        "unit_entity_links": int(db.execute(sql_text("SELECT count(*) FROM unit_entity_links")).scalar() or 0),
        "unit_associations": int(db.execute(sql_text("SELECT count(*) FROM unit_associations")).scalar() or 0),
        "degraded_units": degraded,
    }


def load_semantics(db, memory_captures: dict[str, str]) -> dict[str, dict]:
    """memory_id -> merged semantic fields of that memory's ready units.

    Read from the database rather than cached in the state file, so the cached
    and freshly-seeded paths produce this the same way (and so the simulated
    user always reflects what is actually stored).
    """
    capture_to_memory = {cid: mid for mid, cid in memory_captures.items()}
    units = (
        db.query(MemoryUnit)
        .filter(MemoryUnit.capture_id.in_(list(memory_captures.values())))
        .order_by(MemoryUnit.created_at)
        .all()
    )
    merged: dict[str, dict] = {mid: {} for mid in memory_captures}
    for unit in units:
        mid = capture_to_memory.get(str(unit.capture_id))
        if mid is None:
            continue
        target = merged[mid]
        for k, v in (unit.semantic or {}).items():
            if isinstance(v, list):
                target.setdefault(k, [])
                target[k].extend(v)
            elif k not in target or not target.get(k):
                target[k] = v
    return merged


async def seed_corpus(db, llm: LLMProvider, memories: list[dict]) -> tuple[dict[str, str], float]:
    """Ingest every memory as its own independent capture. Returns
    (memory_id -> capture_id, seconds spent).

    `allow_clarification=False`: the corpus is deliberately authored, and a
    unit parked in `awaiting_clarification` is excluded from hybrid_search by
    the status gate, which would silently remove part of the corpus from every
    strategy's reach. Suppressing the ingestion-time question keeps this a
    *retrieval* benchmark rather than a joint retrieval+clarification one. This
    is a real deviation from production ingestion and is listed in
    docs/EVALUATION.md "Caveats".
    """
    db.execute(sql_text(TRUNCATE))
    db.commit()
    started = time.perf_counter()
    memory_captures: dict[str, str] = {}
    for i, mem in enumerate(memories, start=1):
        capture = await create_capture_and_finish(db, llm, text=mem["text"], allow_clarification=False)
        memory_captures[mem["id"]] = str(capture.id)
        # Release the connection before the next memory's minutes of model
        # time. create_capture_and_finish() ends with a refresh, which
        # autobegins a read transaction on `db`; leaving it open across the
        # next iteration's LLM calls means a connection sitting idle in a
        # transaction for however long ingestion takes, and
        # app/db.py sets idle_in_transaction_session_timeout=180000. Observed
        # for real: with Ollama saturated by another process on this machine,
        # one memory exceeded 180s and Postgres terminated the connection
        # mid-seed. Committing here costs nothing (expire_on_commit=False) and
        # removes the failure mode entirely.
        db.commit()
        print(
            f"  seeded {i}/{len(memories)}  {mem['id']:<4} ({mem['theme']}) "
            f"-> {len(capture.units)} unit(s)  [{time.perf_counter() - started:.0f}s elapsed]",
            flush=True,
        )
    db.expire_all()
    return memory_captures, time.perf_counter() - started


def load_cached_state(db, fingerprint: str, expected_memories: int) -> dict | None:
    """Return the cached state iff it matches this corpus AND the database
    still holds what it says it does. Any mismatch returns None (re-seed)."""
    if not STATE_PATH.exists():
        return None
    try:
        state = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        print(f"  cache: {STATE_PATH.name} unreadable, ignoring it")
        return None

    database = db.execute(sql_text("SELECT current_database()")).scalar()
    if state.get("fingerprint") != fingerprint:
        print("  cache: corpus fingerprint differs from the cached one")
        return None
    if state.get("database") != database:
        print(f"  cache: cached for database {state.get('database')!r}, connected to {database!r}")
        return None
    memory_captures = state.get("memory_captures") or {}
    if len(memory_captures) != expected_memories:
        print(f"  cache: records {len(memory_captures)} memories, corpus has {expected_memories}")
        return None

    actual = ready_unit_count(db)
    if actual != state.get("ready_units"):
        print(f"  cache: database has {actual} ready units, cache recorded {state.get('ready_units')}")
        return None

    present = int(
        db.execute(
            sql_text("SELECT count(DISTINCT capture_id) FROM memory_units WHERE capture_id = ANY(:ids)"),
            {"ids": [str(c) for c in memory_captures.values()]},
        ).scalar()
        or 0
    )
    if present != len(memory_captures):
        print(f"  cache: only {present}/{len(memory_captures)} cached captures still exist")
        return None
    return state


async def load_or_seed_corpus(
    db, llm: LLMProvider, memories: list[dict], raw_yaml: str, limit: int | None, reseed: bool
) -> tuple[dict[str, str], dict]:
    """Returns (memory_id -> capture_id, info-about-what-happened)."""
    fingerprint = corpus_fingerprint(raw_yaml, limit)
    print(f"Corpus: {len(memories)} memories, fingerprint {fingerprint[:16]}" + (f" (--limit {limit})" if limit else ""))

    if reseed:
        print("  --reseed given: ignoring any cached corpus and re-ingesting from scratch")
        cached = None
    else:
        cached = load_cached_state(db, fingerprint, len(memories))

    if cached is not None:
        print(
            f"  CACHE HIT -- seeding SKIPPED. Reusing {cached['ready_units']} ready units "
            f"ingested at {cached.get('seeded_at', 'an unknown time')} "
            f"(that ingest took {cached.get('seed_seconds', 0):.0f}s)."
        )
        return cached["memory_captures"], {
            "seeded": False,
            "seed_seconds": 0.0,
            "original_seed_seconds": cached.get("seed_seconds"),
            "ready_units": cached["ready_units"],
            "fingerprint": fingerprint,
        }

    print(f"  CACHE MISS -- ingesting {len(memories)} memories through the real pipeline. This is the slow part.")
    memory_captures, seconds = await seed_corpus(db, llm, memories)
    ready = ready_unit_count(db)
    state = {
        "fingerprint": fingerprint,
        "database": db.execute(sql_text("SELECT current_database()")).scalar(),
        "corpus_file": CORPUS_PATH.name,
        "limit": limit,
        "memories": len(memories),
        "ready_units": ready,
        "seed_seconds": seconds,
        "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "memory_captures": memory_captures,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"  seeding took {seconds:.0f}s for {ready} ready units; cached to {STATE_PATH.name}")
    return memory_captures, {
        "seeded": True,
        "seed_seconds": seconds,
        "original_seed_seconds": seconds,
        "ready_units": ready,
        "fingerprint": fingerprint,
    }


# ---------------------------------------------------------------------------
# The four strategies
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    db: object
    real_llm: LLMProvider
    capture_to_memory: dict[str, str]
    semantics: dict[str, dict]
    budget: RetrievalBudget = field(default_factory=RetrievalBudget)

    def memory_ids(self, capture_ids) -> list[str]:
        """Normalise any strategy's sources into a ranked list of corpus ids,
        deduped, order preserved. This is the whole reason the scorer needs no
        per-strategy branch."""
        out: list[str] = []
        for cid in capture_ids:
            mid = self.capture_to_memory.get(str(cid))
            if mid is not None and mid not in out:
                out.append(mid)
        return out


# Pure vector retrieval. Written as SQL here rather than calling hybrid_search
# with the FTS half suppressed, for two reasons: (a) hybrid_search has no
# "vector only" mode, and adding one would mean editing production code purely
# to be benchmarked, and (b) hybrid_search's fused score (0.65*vector +
# 0.35*normalised-FTS) is not what a vector-only baseline should rank on --
# with FTS zeroed, every score would be scaled by 0.65 and the relevance bar
# would mean something different. Here the cosine similarity IS the score. The
# relevance bar (VECTOR_SIM_THRESHOLD) and the top-k (TOP_K_FOR_ANSWER) are
# imported from the production pipeline rather than re-picked, so this baseline
# is the real system's vector half at the real system's settings -- it is not
# handicapped by a different threshold.
_VECTOR_ONLY_SQL = """
    SELECT id, 1 - (embedding <=> (:qv)::vector) AS score
    FROM memory_units
    WHERE embedding IS NOT NULL AND status = 'ready'
    ORDER BY embedding <=> (:qv)::vector
    LIMIT :limit
"""


async def run_vector(ctx: Ctx, llm: LLMProvider, case: dict) -> dict:
    query = case["question"]
    vector = await embed_query(llm, query)
    if vector is None:
        return {"answer": ABSTENTION_MESSAGE, "grounded": False, "retrieved": [], "iterations": 1}

    literal = "[" + ",".join(repr(float(x)) for x in vector) + "]"
    rows = ctx.db.execute(sql_text(_VECTOR_ONLY_SQL), {"qv": literal, "limit": 20}).all()
    hits = [(r.id, float(r.score)) for r in rows if float(r.score) >= VECTOR_SIM_THRESHOLD][:TOP_K_FOR_ANSWER]
    if not hits:
        return {"answer": ABSTENTION_MESSAGE, "grounded": False, "retrieved": [], "iterations": 1}

    units = {u.id: u for u in ctx.db.query(MemoryUnit).filter(MemoryUnit.id.in_([h[0] for h in hits])).all()}
    ranked = [
        ScoredUnit(unit=units[uid], vector_score=score, fts_score=None, fused_score=score)
        for uid, score in hits
        if uid in units
    ]
    context = sibling_context(ctx.db, ranked)
    answer, grounded = await generate_answer(llm, query, ranked, context, [])
    return {
        "answer": answer if grounded else ABSTENTION_MESSAGE,
        "grounded": grounded,
        "retrieved": ctx.memory_ids(s.unit.capture_id for s in ranked),
        "iterations": 1,
    }


async def run_semantic_rerank(ctx: Ctx, llm: LLMProvider, case: dict) -> dict:
    result = await run_agent(ctx.db, llm, case["question"])
    return {
        "answer": result.answer,
        "grounded": not result.abstained,
        "retrieved": ctx.memory_ids(s.unit.capture_id for s in result.sources),
        "iterations": None,  # derived from the evaluate_sufficiency call count
    }


async def run_associative(ctx: Ctx, llm: LLMProvider, case: dict) -> dict:
    """The level-6 engine, single-shot: graph expansion and structured
    evidence, but no clarifying question. Mirrors the candidate-building half
    of app/retrieval/session.py::_run_round, minus everything user-mediated
    (negative cues, positive-cue boost, ambiguity narrowing), so the measured
    difference against `adaptive` is specifically clarification."""
    query = case["question"]
    started = time.monotonic()
    result = await automatic_search(ctx.db, llm, query)
    if not result.ranked and not result.ranked_chunks:
        return {"answer": ABSTENTION_MESSAGE, "grounded": False, "retrieved": [], "iterations": None}

    candidates = build_direct_candidates(result.ranked)
    expand, skip_reason = should_expand(candidates)
    expanded_count = 0
    if expand:
        expanded = expand_with_associations(ctx.db, candidates, ctx.budget)
        candidates.extend(expanded)
        expanded_count = len(expanded)
    collect_evidence(ctx.db, candidates, query=query, chunks=result.ranked_chunks)
    candidates = apply_budget(rerank(candidates), ctx.budget)

    direct_ranked = [c.direct for c in candidates if c.direct is not None]
    context = sibling_context(ctx.db, direct_ranked)
    evidence = format_evidence_block(candidates, context, result.ranked_chunks)
    answer, grounded = await generate_answer(
        llm, query, direct_ranked, context, result.ranked_chunks, evidence=evidence
    )
    return {
        "answer": answer if grounded else ABSTENTION_MESSAGE,
        "grounded": grounded,
        # Association-route candidates count as retrieved: reaching a memory by
        # inference is the whole claim this strategy makes.
        "retrieved": ctx.memory_ids(c.unit.capture_id for c in candidates),
        "iterations": None,
        "expanded_candidates": expanded_count,
        "expansion_skipped_reason": skip_reason,
    }


async def run_adaptive(ctx: Ctx, llm: LLMProvider, case: dict) -> dict:
    """The real RecallSession, with the simulated user answering whatever
    clarifying question it actually asks."""
    query = case["question"]
    truth = ctx.semantics.get(case.get("ground_truth_memory") or "") if case.get("ground_truth_memory") else None
    session = await start_session(ctx.db, llm, query)
    questions: list[str] = []
    while session.status == "needs_user_cue" and len(questions) < MAX_SIMULATED_TURNS:
        question = session.current_question
        questions.append(question)
        session = await continue_session(ctx.db, llm, session, simulate_reply(question, truth))

    # final_evidence is the reranked candidate list (direct + association) the
    # answer was actually generated from; final_sources is the directly-matched
    # subset. Prefer the former so this strategy's retrieved set is comparable
    # to `associative`'s, and fall back for the paths that set only sources.
    #
    # A session still in `needs_user_cue` after MAX_SIMULATED_TURNS has
    # snapshotted neither, so it reports an empty retrieved set: RecallSession
    # only records candidates when it reaches a decision. That makes recall@k a
    # LOWER bound for this strategy specifically on queries where it ran out of
    # simulated turns -- noted in docs/EVALUATION.md rather than papered over
    # by re-running retrieval here, which would measure something other than
    # what the session actually delivered to its user (nothing).
    evidence = session.final_evidence or []
    capture_ids = [e["capture_id"] for e in evidence] or [s["capture_id"] for s in (session.final_sources or [])]
    return {
        "answer": session.final_answer or ABSTENTION_MESSAGE,
        "grounded": session.status == "success",
        "retrieved": ctx.memory_ids(capture_ids),
        "questions_asked": len(questions),
        "questions": questions,
        "iterations": session.round_count,
        "status": session.status,
    }


_STRATEGY_FNS = {
    "vector": run_vector,
    "semantic_rerank": run_semantic_rerank,
    "associative": run_associative,
    "adaptive": run_adaptive,
}


async def run_strategy(name: str, ctx: Ctx, case: dict) -> dict:
    """Run one strategy and normalise its result into the uniform dict every
    downstream consumer (scorer, report, JSON) sees."""
    llm = TimingLLM(ctx.real_llm)
    started = time.perf_counter()
    raw = await _STRATEGY_FNS[name](ctx, llm, case)
    wall = time.perf_counter() - started

    iterations = raw.get("iterations")
    if iterations is None:
        # One search round plus one more for every sufficiency decision that
        # did not stop the loop. This is inferred from the call log rather than
        # reported by the retrieval code, which has no iteration counter to
        # read -- documented as an approximation in docs/EVALUATION.md.
        iterations = 1 + llm.count("evaluate_sufficiency")

    return {
        "strategy": name,
        "answer": raw.get("answer") or "",
        "grounded": bool(raw.get("grounded")),
        "retrieved_memory_ids": list(raw.get("retrieved") or []),
        "questions_asked": int(raw.get("questions_asked") or 0),
        "questions": raw.get("questions") or [],
        "iterations": int(iterations),
        "llm_calls": len(llm.calls),
        "llm_seconds": round(sum(s for _, s in llm.calls), 3),
        "wall_seconds": round(wall, 3),
        "llm_calls_by_kind": {kind: llm.count(kind) for kind in sorted({k for k, _ in llm.calls})},
        "status": raw.get("status"),
        "expanded_candidates": raw.get("expanded_candidates"),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def is_unanswerable(case: dict) -> bool:
    return case.get("expect") == "abstain"


def score(case: dict, result: dict) -> dict:
    """One strategy's result on one query, as booleans. No branch on which
    strategy produced it -- that normalisation already happened."""
    ids = result["retrieved_memory_ids"]
    if is_unanswerable(case):
        return {
            "answerable": False,
            "recall_1": None,
            "recall_5": None,
            "correct": None,
            "false_retrieval": result["grounded"],
            "abstained": not result["grounded"],
        }
    relevant = set(case["relevant_memories"])
    answer = (result["answer"] or "").lower()
    return {
        "answerable": True,
        "recall_1": bool(relevant & set(ids[:1])),
        "recall_5": bool(relevant & set(ids[:5])),
        "correct": result["grounded"] and any(p.lower() in answer for p in case["expect_contains"]),
        "false_retrieval": None,
        "abstained": not result["grounded"],
    }


def _rate(values: list[bool]) -> float | None:
    return (sum(1 for v in values if v) / len(values)) if values else None


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """rows: one dict per (query, strategy) with 'strategy', 'score', 'result'."""
    out: dict[str, dict] = {}
    for strategy in STRATEGIES:
        mine = [r for r in rows if r["strategy"] == strategy]
        answerable = [r for r in mine if r["score"]["answerable"]]
        unanswerable = [r for r in mine if not r["score"]["answerable"]]
        out[strategy] = {
            "n_queries": len(mine),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "recall_at_1": _rate([r["score"]["recall_1"] for r in answerable]),
            "recall_at_5": _rate([r["score"]["recall_5"] for r in answerable]),
            "correctness": _rate([r["score"]["correct"] for r in answerable]),
            "false_retrieval_rate": _rate([r["score"]["false_retrieval"] for r in unanswerable]),
            "abstention_rate_answerable": _rate([r["score"]["abstained"] for r in answerable]),
            "mean_questions": statistics.fmean([r["result"]["questions_asked"] for r in mine]) if mine else 0.0,
            "mean_iterations": statistics.fmean([r["result"]["iterations"] for r in mine]) if mine else 0.0,
            "mean_llm_calls": statistics.fmean([r["result"]["llm_calls"] for r in mine]) if mine else 0.0,
            "mean_llm_seconds": statistics.fmean([r["result"]["llm_seconds"] for r in mine]) if mine else 0.0,
            "mean_wall_seconds": statistics.fmean([r["result"]["wall_seconds"] for r in mine]) if mine else 0.0,
        }
    return out


def by_category(rows: list[dict]) -> dict[str, dict[str, dict]]:
    categories = sorted({r["category"] for r in rows}, key=lambda c: CATEGORIES.index(c))
    out: dict[str, dict[str, dict]] = {}
    for category in categories:
        subset = [r for r in rows if r["category"] == category]
        out[category] = {}
        for strategy in STRATEGIES:
            mine = [r for r in subset if r["strategy"] == strategy]
            answerable = [r for r in mine if r["score"]["answerable"]]
            out[category][strategy] = {
                "n": len(mine),
                "correctness": _rate([r["score"]["correct"] for r in answerable]),
                "recall_at_5": _rate([r["score"]["recall_5"] for r in answerable]),
                "abstain_ok": _rate([r["score"]["abstained"] for r in mine if not r["score"]["answerable"]]),
            }
    return out


def succeeded(row: dict) -> bool:
    """One definition of "this strategy got this query right", usable for both
    kinds of query: an answerable query needs a correct grounded answer, an
    unanswerable one needs an abstention."""
    s = row["score"]
    return bool(s["correct"]) if s["answerable"] else bool(s["abstained"])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_run_matrix(rows: list[dict], run_label: str) -> None:
    by_query: dict[str, dict[str, dict]] = {}
    order: list[tuple[str, str, str]] = []
    for r in rows:
        key = r["query_id"]
        if key not in by_query:
            by_query[key] = {}
            order.append((key, r["category"], r["question"]))
        by_query[key][r["strategy"]] = r

    print(f"\n{'=' * 118}")
    print(f"Per-query results ({run_label}).  OK = correct answer, or a correct abstention on an unanswerable query.")
    header = f"{'ID':<5}{'Category':<22}"
    for s in STRATEGIES:
        header += f"{s[:14]:<16}"
    header += f"{'adapt.Qs':<9}"
    print(header)
    print("-" * 118)
    for qid, category, _question in order:
        line = f"{qid:<5}{category:<22}"
        for s in STRATEGIES:
            r = by_query[qid].get(s)
            if r is None:
                line += f"{'?':<16}"
                continue
            mark = "OK" if succeeded(r) else "--"
            detail = "abst" if r["score"]["answerable"] and r["score"]["abstained"] else ""
            if not r["score"]["answerable"] and not r["score"]["abstained"]:
                detail = "FALSE"
            line += f"{mark + (' ' + detail if detail else ''):<16}"
        line += f"{by_query[qid]['adaptive']['result']['questions_asked']:<9}"
        print(line)
    print("=" * 118)


def print_summary(summaries: list[dict[str, dict]], n_runs: int) -> None:
    """Headline metrics. With --repeat N, mean across runs plus the observed
    min-max range, because a single run cannot distinguish a real difference
    from sampling noise (docs/TODO.md)."""

    def agg(strategy: str, key: str) -> tuple[float | None, float | None, float | None]:
        vals = [s[strategy][key] for s in summaries if s[strategy][key] is not None]
        if not vals:
            return None, None, None
        return statistics.fmean(vals), min(vals), max(vals)

    metrics = [
        ("recall@1", "recall_at_1", "pct"),
        ("recall@5", "recall_at_5", "pct"),
        ("answer correctness", "correctness", "pct"),
        ("abstention (answerable)", "abstention_rate_answerable", "pct"),
        ("mean clarifying Qs", "mean_questions", "num"),
        ("mean iterations", "mean_iterations", "num"),
        ("mean LLM calls", "mean_llm_calls", "num"),
        ("mean LLM seconds", "mean_llm_seconds", "num"),
        ("mean wall seconds", "mean_wall_seconds", "num"),
    ]

    print(f"\n{'=' * 118}")
    print(f"HEADLINE METRICS  (mean of {n_runs} run(s)" + (", [min-max] across runs)" if n_runs > 1 else ")"))
    print("-" * 118)
    print(f"{'metric':<26}" + "".join(f"{s[:22]:<23}" for s in STRATEGIES))
    print("-" * 118)
    for label, key, kind in metrics:
        line = f"{label:<26}"
        for strategy in STRATEGIES:
            mean, lo, hi = agg(strategy, key)
            if mean is None:
                cell = "n/a"
            elif kind == "pct":
                cell = f"{mean:.1%}" + (f" [{lo:.1%}-{hi:.1%}]" if n_runs > 1 else "")
            else:
                cell = f"{mean:.2f}" + (f" [{lo:.2f}-{hi:.2f}]" if n_runs > 1 else "")
            line += f"{cell:<23}"
        print(line)
    print("=" * 118)

    print("\n*** FALSE RETRIEVAL RATE (fraction of UNANSWERABLE queries answered instead of abstained) ***")
    print("    Lower is better. A strategy that answers everything buys its recall and correctness here.")
    n_unanswerable = summaries[0][STRATEGIES[0]]["n_unanswerable"]
    print(f"    Measured over {n_unanswerable} unanswerable query/queries per run.")
    for strategy in STRATEGIES:
        mean, lo, hi = agg(strategy, "false_retrieval_rate")
        if mean is None:
            print(f"      {strategy:<18} n/a (no unanswerable queries in this run)")
        else:
            rng = f"  [{lo:.1%}-{hi:.1%}]" if n_runs > 1 else ""
            print(f"      {strategy:<18} {mean:.1%}{rng}")


def print_category_breakdown(per_category_runs: list[dict], n_runs: int) -> None:
    categories = list(per_category_runs[0].keys())
    print(f"\n{'=' * 118}")
    print("BY CATEGORY -- answer correctness (abstention correctness for 'unanswerable'), recall@5 in brackets")
    print("-" * 118)
    print(f"{'category':<22}{'n':<4}" + "".join(f"{s[:20]:<21}" for s in STRATEGIES))
    print("-" * 118)
    for category in categories:
        n = per_category_runs[0][category][STRATEGIES[0]]["n"]
        line = f"{category:<22}{n:<4}"
        for strategy in STRATEGIES:
            if category == "unanswerable":
                vals = [r[category][strategy]["abstain_ok"] for r in per_category_runs]
                vals = [v for v in vals if v is not None]
                cell = f"{statistics.fmean(vals):.0%} abst" if vals else "n/a"
            else:
                corr = [r[category][strategy]["correctness"] for r in per_category_runs]
                rec = [r[category][strategy]["recall_at_5"] for r in per_category_runs]
                corr = [v for v in corr if v is not None]
                rec = [v for v in rec if v is not None]
                cell = (
                    f"{statistics.fmean(corr):.0%} [{statistics.fmean(rec):.0%}]"
                    if corr
                    else "n/a"
                )
            line += f"{cell:<21}"
        print(line)
    print("=" * 118)


def print_win_sets(runs: list[list[dict]], n_runs: int) -> None:
    """For every ordered pair, the queries one strategy got right and the other
    got wrong. This is the most informative construct in the existing report
    (docs/REPORT.md): an aggregate delta of a few points says nothing about
    whether the two strategies are failing on the same queries or different
    ones."""
    print(f"\n{'=' * 118}")
    print("PAIRWISE WIN SETS -- 'A succeeded, B failed', by query id")
    if n_runs > 1:
        print(f"(k/{n_runs}) is how many of the {n_runs} runs that query flipped in. k < {n_runs} is noise, not a finding.")
    print("-" * 118)
    for a in STRATEGIES:
        for b in STRATEGIES:
            if a == b:
                continue
            counts: dict[str, int] = {}
            for rows in runs:
                lookup = {(r["query_id"], r["strategy"]): r for r in rows}
                for qid in sorted({r["query_id"] for r in rows}):
                    ra, rb = lookup.get((qid, a)), lookup.get((qid, b))
                    if ra is None or rb is None:
                        continue
                    if succeeded(ra) and not succeeded(rb):
                        counts[qid] = counts.get(qid, 0) + 1
            if not counts:
                print(f"  {a} > {b}: none")
                continue
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            rendered = ", ".join(f"{qid}({k}/{n_runs})" if n_runs > 1 else qid for qid, k in ordered)
            print(f"  {a} > {b}: {len(counts)} query/queries -- {rendered}")
    print("=" * 118)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare four retrieval strategies on one shared corpus.")
    p.add_argument("--reseed", action="store_true", help="ignore the corpus cache and re-ingest from scratch")
    p.add_argument("--limit", type=int, default=None, help="seed only the first N memories (fast smoke run)")
    p.add_argument("--repeat", type=int, default=1, help="run the query set N times and report mean +/- spread")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="validate the corpus/query YAML and exit (no database, no model)",
    )
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # A full run is 20+ minutes and is normally watched through a redirected
    # log, where block buffering would hide all progress until the process
    # exits (or, worse, until it crashes).
    sys.stdout.reconfigure(line_buffering=True)

    raw_corpus = CORPUS_PATH.read_text()
    corpus = yaml.safe_load(raw_corpus)
    query_data = yaml.safe_load(QUERIES_PATH.read_text())

    print(f"Self-check: {CORPUS_PATH.name} + {QUERIES_PATH.name}")
    problems = self_check(corpus, query_data)
    if problems:
        print(f"  FAILED -- {len(problems)} problem(s):")
        for problem in problems:
            print(f"    - {problem}")
        print("\nRefusing to run: a dangling memory reference scores as a retrieval failure, not as a typo.")
        return 1
    print(
        f"  OK -- {len(corpus['memories'])} memories, {len(query_data['queries'])} queries, "
        f"every relevant_memories/ground_truth_memory id resolves, all "
        f"{len(CATEGORIES)} categories covered."
    )
    if args.self_check:
        return 0

    if args.repeat < 1:
        print("--repeat must be >= 1")
        return 1

    memories = corpus["memories"]
    if args.limit:
        memories = memories[: args.limit]
    seeded_ids = {m["id"] for m in memories}

    # A limited corpus can only answer the queries whose whole ground truth is
    # present. Running the rest would score them as retrieval failures for
    # every strategy, which is a lie about the strategies. Unanswerable queries
    # always run -- they have no ground truth to be missing.
    all_queries = query_data["queries"]
    queries = [
        q
        for q in all_queries
        if is_unanswerable(q) or (set(q["relevant_memories"]) | {q["ground_truth_memory"]}) <= seeded_ids
    ]
    kept_ids = {q["id"] for q in queries}
    skipped = [q["id"] for q in all_queries if q["id"] not in kept_ids]
    if skipped:
        print(
            f"\n--limit {args.limit}: running {len(queries)}/{len(all_queries)} queries. "
            f"Skipped (ground truth not seeded): {', '.join(skipped)}"
        )
        print("  A limited run is a plumbing smoke test, NOT a measurement of the full query set.")

    db = get_sessionmaker()()
    assert_eval_database(db)
    real_llm = get_llm_provider()

    memory_captures, seed_info = await load_or_seed_corpus(
        db, real_llm, memories, raw_corpus, args.limit, args.reseed
    )
    capture_to_memory = {str(cid): mid for mid, cid in memory_captures.items()}
    semantics = load_semantics(db, memory_captures)
    stats = graph_stats(db)
    print(
        f"  Association graph: {stats['entities']} entities, {stats['unit_entity_links']} unit-entity "
        f"links, {stats['unit_associations']} unit associations."
    )
    if stats["unit_associations"] == 0:
        print("  WARNING: the association graph is empty -- the `associative` strategy has nothing to walk.")
    if stats["degraded_units"]:
        print(
            f"  WARNING: {stats['degraded_units']} of {seed_info['ready_units']} ready units have "
            "enrichment_status='unavailable' (the model failed or timed out during ingestion). Their "
            "semantics are incomplete, which penalises them in rerank and changes should_expand's "
            "verdict -- re-run with --reseed on an idle machine before trusting these numbers."
        )

    ctx = Ctx(db=db, real_llm=real_llm, capture_to_memory=capture_to_memory, semantics=semantics)

    runs: list[list[dict]] = []
    for repeat in range(1, args.repeat + 1):
        print(f"\n{'#' * 118}")
        print(f"# RUN {repeat}/{args.repeat} -- {len(queries)} queries x {len(STRATEGIES)} strategies")
        print(f"{'#' * 118}")
        rows: list[dict] = []
        run_started = time.perf_counter()
        for case in queries:
            print(f"\n[{case['id']} / {case['category']}] {case['question']}", flush=True)
            for strategy in STRATEGIES:
                result = await run_strategy(strategy, ctx, case)
                # Same reason as in seed_corpus(): a strategy's reads autobegin
                # a transaction on `db` that would otherwise stay open through
                # the NEXT strategy's LLM calls, and
                # idle_in_transaction_session_timeout is 180s. All four
                # strategies are read-only with respect to stored memories, so
                # this commits nothing but the connection's idleness.
                db.commit()
                scored = score(case, result)
                rows.append(
                    {
                        "repeat": repeat,
                        "query_id": case["id"],
                        "category": case["category"],
                        "question": case["question"],
                        "strategy": strategy,
                        "result": result,
                        "score": scored,
                    }
                )
                mark = "OK  " if succeeded(rows[-1]) else "MISS"
                extra = f", {result['questions_asked']}q" if result["questions_asked"] else ""
                print(
                    f"    {strategy:<16} {mark} r@1={scored['recall_1']} r@5={scored['recall_5']} "
                    f"ids={result['retrieved_memory_ids'][:5]} "
                    f"({result['llm_calls']} calls, {result['llm_seconds']:.1f}s llm, "
                    f"{result['wall_seconds']:.1f}s wall{extra})",
                    flush=True,
                )
                print(f"        -> {result['answer'][:220]}", flush=True)
                for q in result["questions"]:
                    print(f"        asked: {q}", flush=True)
        print(f"\nRun {repeat} took {time.perf_counter() - run_started:.0f}s.")
        runs.append(rows)

    summaries = [aggregate(rows) for rows in runs]
    per_category = [by_category(rows) for rows in runs]

    for i, rows in enumerate(runs, start=1):
        print_run_matrix(rows, f"run {i}/{args.repeat}")
    print_summary(summaries, args.repeat)
    print_category_breakdown(per_category, args.repeat)
    print_win_sets(runs, args.repeat)

    print(f"\n{'=' * 118}")
    print("HOW TO READ THIS")
    print("-" * 118)
    if args.repeat == 1:
        print(
            "This is a SINGLE run. gemma4:12b is not deterministic: on a query set this size a one- or\n"
            "two-case flip between strategies is expected with no code change at all, so a small gap here\n"
            "is NOT evidence of a real difference. Re-run with --repeat 3 (or more) to see the noise floor\n"
            "before believing any ordering. docs/TODO.md calls this out explicitly."
        )
    else:
        print(
            f"Ran {args.repeat} times over the same stored corpus. The [min-max] ranges above are the\n"
            "sampling-noise floor for this corpus and model: a between-strategy gap smaller than the\n"
            "within-strategy range is not a finding. The (k/N) counts in the win sets show which\n"
            "individual queries are stable and which just flip."
        )
    print(
        "Correctness is an `expect_contains` substring test, not a judge -- it can mark a correct answer\n"
        "wrong for using a synonym, and a wrong answer right for naming the expected token in passing.\n"
        "See docs/EVALUATION.md for every metric's exact definition and the caveats."
    )
    print("=" * 118)

    payload = {
        "meta": {
            "corpus_file": CORPUS_PATH.name,
            "queries_file": QUERIES_PATH.name,
            "fingerprint": seed_info["fingerprint"],
            "limit": args.limit,
            "repeat": args.repeat,
            "memories_seeded": len(memories),
            "queries_run": [q["id"] for q in queries],
            "queries_skipped": skipped,
            "ready_units": seed_info["ready_units"],
            "seeding_skipped_cache_hit": not seed_info["seeded"],
            "seed_seconds_this_run": seed_info["seed_seconds"],
            "seed_seconds_original_ingest": seed_info["original_seed_seconds"],
            "graph": stats,
            "strategies": list(STRATEGIES),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary_per_run": summaries,
        "per_category_per_run": per_category,
        "runs": [
            [
                {
                    "repeat": r["repeat"],
                    "query_id": r["query_id"],
                    "category": r["category"],
                    "question": r["question"],
                    "strategy": r["strategy"],
                    "succeeded": succeeded(r),
                    **r["score"],
                    **{k: v for k, v in r["result"].items() if k != "strategy"},
                }
                for r in rows
            ]
            for rows in runs
        ],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nRaw per-query per-strategy results written to {RESULTS_PATH}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
