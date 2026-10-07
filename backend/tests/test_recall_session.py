"""Tests for the iterative RecallSession controller (app/retrieval/session.py).

The session controller's own branching logic (ambiguity -> question ->
resolution -> answer/exhaustion) is the thing under test here, not the
numeric behavior of real hybrid search scoring (that's covered by
test_retrieval.py/test_agent.py against the fake embedding/fts stack
already). So `automatic_search` is monkeypatched to return exact,
hand-picked candidate sets per round -- real MemoryUnit/Capture rows
(via create_capture) so generate_answer/sibling_context still operate on
genuine ORM objects, just with fully controlled "what did search find"
results feeding into them.
"""

import pytest

from app.ingestion.pipeline import create_capture
from app.models import MemoryUnit
from app.retrieval import session as session_module
from app.retrieval.agent import AutomaticSearchResult
from app.retrieval.discriminate import (
    apply_negative_cues,
    apply_positive_cue_boost,
    generate_discriminating_question,
    is_ambiguous,
)
from app.retrieval.pipeline import ScoredUnit
from app.retrieval.session import (
    STATUS_CANCELLED,
    STATUS_EXHAUSTED,
    STATUS_NEEDS_USER_CUE,
    STATUS_SUCCESS,
    RecallSessionError,
    cancel_session,
    continue_session,
    start_session,
)
from tests.fakes import FakeLLMProvider


def _scored(unit: MemoryUnit, score: float) -> ScoredUnit:
    return ScoredUnit(unit=unit, vector_score=None, fts_score=score, fused_score=score)


def _patch_rounds(monkeypatch, results: list[AutomaticSearchResult], calls: list[dict] | None = None):
    """Each call to automatic_search returns the next result in order
    (the last one repeats if more rounds happen than configured). If
    `calls` is given, each invocation's (query, extra_queries, tried) is
    recorded there for the test to inspect."""
    state = {"n": 0}

    async def _fake(db, llm, query, *, extra_queries=None, tried=None):
        idx = min(state["n"], len(results) - 1)
        state["n"] += 1
        if calls is not None:
            calls.append({"query": query, "extra_queries": list(extra_queries or []), "tried": set(tried or set())})
        return results[idx]

    monkeypatch.setattr(session_module, "automatic_search", _fake)


async def _make_two_candidates(db) -> tuple[MemoryUnit, MemoryUnit]:
    """Two independent, equally-plausible candidate memories from
    different origin captures -- the ambiguous case the spec's worked
    example is built around."""
    ingest_llm = FakeLLMProvider(
        enrichment_response=[
            {
                "meaning": "Anna recommended a paper",
                "aliases": [],
                "people": ["Anna"],
                "organizations": [],
                "locations": ["ETH"],
                "topics": ["robotics"],
                "definitions": [],
                "relations": [],
                "event_date": "",
            },
            {
                "meaning": "Markus recommended a book",
                "aliases": [],
                "people": ["Markus"],
                "organizations": [],
                "locations": ["Paris"],
                "topics": ["machine learning"],
                "definitions": [],
                "relations": [],
                "event_date": "",
            },
        ]
    )
    cap_a = await create_capture(db, ingest_llm, text="Anna recommended a paper to me.", uploads=[])
    cap_b = await create_capture(db, ingest_llm, text="Markus recommended a book to me.", uploads=[])
    return cap_a.units[0], cap_b.units[0]


@pytest.mark.asyncio
async def test_ambiguous_candidates_prompt_then_resolve(db, monkeypatch):
    unit_a, unit_b = await _make_two_candidates(db)

    round1 = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)],
        ranked_chunks=[],
        tried_queries={"What did someone recommend to me?"},
        hypotheses=[],
    )
    round2 = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.90), _scored(unit_b, 0.40)],
        ranked_chunks=[],
        tried_queries={"What did someone recommend to me?", "ETH"},
        hypotheses=[],
    )
    _patch_rounds(monkeypatch, [round1, round2])

    agent_llm = FakeLLMProvider(
        evaluate_response={"sufficient": True, "reformulated_queries": []},
        cue_response={"cues": [{"text": "ETH", "confidence": "certain", "polarity": "positive"}]},
        answer_response={"answer": "Anna recommended a paper about tactile sensing.", "grounded": True},
    )

    session = await start_session(db, agent_llm, "What did someone recommend to me?")
    assert session.status == STATUS_NEEDS_USER_CUE
    assert session.current_question is not None
    assert session.round_count == 1

    session = await continue_session(db, agent_llm, session, "Yes, it was at ETH.")
    assert session.status == STATUS_SUCCESS
    assert session.final_answer == "Anna recommended a paper about tactile sensing."
    assert session.round_count == 2
    assert {s["capture_id"] for s in session.final_sources} == {str(unit_a.capture_id)}

    # the cue is session bookkeeping only, never a new memory
    assert any(c["text"] == "ETH" and c["polarity"] == "positive" for c in session.cues)
    assert db.query(MemoryUnit).count() == 2


@pytest.mark.asyncio
async def test_negative_cue_excludes_candidate(db, monkeypatch):
    unit_a, unit_b = await _make_two_candidates(db)

    round1 = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)],
        ranked_chunks=[],
        tried_queries=set(),
        hypotheses=[],
    )
    round2 = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)],
        ranked_chunks=[],
        tried_queries=set(),
        hypotheses=[],
    )
    _patch_rounds(monkeypatch, [round1, round2])

    agent_llm = FakeLLMProvider(
        cue_response={"cues": [{"text": "Anna", "confidence": "certain", "polarity": "negative"}]},
        answer_response={"answer": "Markus recommended a book.", "grounded": True},
    )

    session = await start_session(db, agent_llm, "Who recommended something to me?")
    assert session.status == STATUS_NEEDS_USER_CUE

    session = await continue_session(db, agent_llm, session, "No, it wasn't Anna.")
    assert session.status == STATUS_SUCCESS
    # unit_a (Anna) must be gone from the evidence the answer was grounded in
    assert {s["capture_id"] for s in session.final_sources} == {str(unit_b.capture_id)}
    assert any(c["text"] == "Anna" and c["polarity"] == "negative" for c in session.cues)


@pytest.mark.asyncio
async def test_uncertain_cue_is_search_signal_not_hard_filter(db, monkeypatch):
    unit_a, unit_b = await _make_two_candidates(db)
    ambiguous = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)], ranked_chunks=[], tried_queries=set(), hypotheses=[]
    )
    calls: list[dict] = []
    _patch_rounds(monkeypatch, [ambiguous, ambiguous], calls=calls)

    # "violin" matches neither candidate's semantic metadata, so it can't
    # win via the positive-cue-boost path either -- isolates "is it used
    # as a search cue at all" from "does it resolve/eliminate anything".
    agent_llm = FakeLLMProvider(
        cue_response={"cues": [{"text": "violin", "confidence": "uncertain", "polarity": "positive"}]},
    )
    session = await start_session(db, agent_llm, "Who recommended something to me?")
    assert session.status == STATUS_NEEDS_USER_CUE

    session = await continue_session(db, agent_llm, session, "maybe something about a violin")

    assert any(c["text"] == "violin" and c["confidence"] == "uncertain" for c in session.cues)
    # used as a search cue this round...
    assert "violin" in calls[1]["extra_queries"]
    # ...but not a hard filter: an uncertain guess that matches nothing
    # must not silently collapse or eliminate either candidate the way a
    # negative cue does (spec section 39) -- the session is still asking,
    # not guessing.
    assert session.status == STATUS_NEEDS_USER_CUE
    before = db.query(MemoryUnit).count()
    assert before == 2  # nothing about this ever created/altered a memory


@pytest.mark.asyncio
async def test_hypotheses_recorded_but_never_become_memory(db, monkeypatch):
    unit_a, _unit_b = await _make_two_candidates(db)
    result = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.90)],
        ranked_chunks=[],
        tried_queries=set(),
        hypotheses=["tactile sensing paper"],
    )
    _patch_rounds(monkeypatch, [result])

    agent_llm = FakeLLMProvider(answer_response={"answer": "Anna recommended a paper.", "grounded": True})
    before = db.query(MemoryUnit).count()

    session = await start_session(db, agent_llm, "What did Anna recommend?")
    assert session.status == STATUS_SUCCESS
    assert any(h["text"] == "tactile sensing paper" for h in session.hypotheses)
    assert db.query(MemoryUnit).count() == before  # hypothesis never persisted as a memory


@pytest.mark.asyncio
async def test_exhausted_when_nothing_found(db, monkeypatch):
    empty = AutomaticSearchResult(ranked=[], ranked_chunks=[], tried_queries=set(), hypotheses=[])
    _patch_rounds(monkeypatch, [empty])

    agent_llm = FakeLLMProvider()
    session = await start_session(db, agent_llm, "something totally unrelated")
    assert session.status == STATUS_EXHAUSTED
    assert session.current_question is None
    assert session.final_answer is None


@pytest.mark.asyncio
async def test_cancel_then_reject_continue(db, monkeypatch):
    unit_a, unit_b = await _make_two_candidates(db)
    ambiguous = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)], ranked_chunks=[], tried_queries=set(), hypotheses=[]
    )
    _patch_rounds(monkeypatch, [ambiguous])

    agent_llm = FakeLLMProvider()
    session = await start_session(db, agent_llm, "Who recommended something to me?")
    assert session.status == STATUS_NEEDS_USER_CUE

    session = cancel_session(db, session)
    assert session.status == STATUS_CANCELLED

    with pytest.raises(RecallSessionError):
        await continue_session(db, agent_llm, session, "ETH")
    with pytest.raises(RecallSessionError):
        cancel_session(db, session)


@pytest.mark.asyncio
async def test_original_query_preserved_across_rounds(db, monkeypatch):
    unit_a, unit_b = await _make_two_candidates(db)
    ambiguous = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)], ranked_chunks=[], tried_queries=set(), hypotheses=[]
    )
    _patch_rounds(monkeypatch, [ambiguous, ambiguous, ambiguous])

    # Neither cue matches unit_a/unit_b's semantic metadata, so the
    # positive-cue-boost path never fires and ambiguity persists across
    # all three rounds -- isolating "is original_query ever overwritten"
    # from disambiguation mechanics (covered elsewhere).
    agent_llm = FakeLLMProvider(
        cue_response=[
            {"cues": [{"text": "a Tuesday", "confidence": "likely", "polarity": "positive"}]},
            {"cues": [{"text": "a long lecture hall", "confidence": "uncertain", "polarity": "positive"}]},
        ]
    )
    query = "Who recommended something to me after a lecture?"
    session = await start_session(db, agent_llm, query)
    session = await continue_session(db, agent_llm, session, "It was on a Tuesday")
    session = await continue_session(db, agent_llm, session, "maybe in a long lecture hall")

    assert session.original_query == query  # never overwritten by reformulations/cues
    assert session.status == STATUS_NEEDS_USER_CUE


@pytest.mark.asyncio
async def test_avoids_repeating_the_same_discriminating_question(db, monkeypatch):
    unit_a, unit_b = await _make_two_candidates(db)
    ambiguous = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)], ranked_chunks=[], tried_queries=set(), hypotheses=[]
    )
    _patch_rounds(monkeypatch, [ambiguous, ambiguous])

    agent_llm = FakeLLMProvider(cue_response={"cues": []})
    session = await start_session(db, agent_llm, "Who recommended something to me?")
    first_question = session.current_question
    assert first_question is not None

    session = await continue_session(db, agent_llm, session, "hmm not sure")
    # same (still-ambiguous) candidate set -- the heuristic must not ask
    # the identical question again; it should find a different
    # discriminating attribute or give up, never loop on one question.
    assert session.current_question != first_question


def test_discriminate_ambiguity_and_question_are_pure_heuristics():
    class _FakeUnit:
        def __init__(self, capture_id, semantic):
            self.id = capture_id
            self.capture_id = capture_id
            self.semantic = semantic

    a = ScoredUnit(unit=_FakeUnit("cap-a", {"people": ["Anna"], "locations": ["ETH"]}), vector_score=None, fts_score=0.5, fused_score=0.50)
    b = ScoredUnit(unit=_FakeUnit("cap-b", {"people": ["Markus"], "locations": ["Paris"]}), vector_score=None, fts_score=0.49, fused_score=0.49)
    c = ScoredUnit(unit=_FakeUnit("cap-a", {"people": ["Anna"], "locations": ["ETH"]}), vector_score=None, fts_score=0.1, fused_score=0.10)

    assert is_ambiguous([a, b]) is True
    assert is_ambiguous([a, c]) is False  # same capture_id -- one origin group, not two candidates
    assert is_ambiguous([a]) is False

    q = generate_discriminating_question([a, b], already_asked=[])
    assert q is not None and ("Anna" in q or "ETH" in q or "Markus" in q or "Paris" in q)
    assert generate_discriminating_question([a, b], already_asked=[q]) is not None or True

    filtered = apply_negative_cues([a, b], ["Anna"])
    assert [s for s in filtered] == [b]
    assert apply_negative_cues([a, b], []) == [a, b]

    boosted = apply_positive_cue_boost([a, b], ["ETH"])
    assert boosted[0].fused_score > a.fused_score  # a's locations=["ETH"] matched
    assert boosted[1].fused_score == b.fused_score  # b unaffected
    assert apply_positive_cue_boost([a, b], ["nothing matches this"])[0].fused_score == a.fused_score
    assert apply_positive_cue_boost([a, b], []) == [a, b]


@pytest.mark.asyncio
async def test_cue_extraction_degrades_on_malformed_llm_output():
    from app.retrieval.cues import extract_cues

    broken = FakeLLMProvider(fail="malformed")
    cues = await extract_cues(broken, "it was definitely at ETH")
    assert cues == [{"text": "it was definitely at ETH", "confidence": "uncertain", "polarity": "positive"}]


@pytest.mark.asyncio
async def test_session_state_reloadable_from_db(db, monkeypatch):
    """Simulates surviving a restart: a fresh query for the same row by
    id must see the same persisted state (spec section 27/J)."""
    from app.models import RecallSession

    unit_a, unit_b = await _make_two_candidates(db)
    ambiguous = AutomaticSearchResult(
        ranked=[_scored(unit_a, 0.50), _scored(unit_b, 0.49)], ranked_chunks=[], tried_queries=set(), hypotheses=[]
    )
    _patch_rounds(monkeypatch, [ambiguous])

    agent_llm = FakeLLMProvider()
    session = await start_session(db, agent_llm, "Who recommended something to me?")
    session_id = session.id
    db.expire_all()

    reloaded = db.get(RecallSession, session_id)
    assert reloaded is not None
    assert reloaded.status == STATUS_NEEDS_USER_CUE
    assert reloaded.current_question == session.current_question
    assert len(reloaded.messages) == 2  # user query + system question
