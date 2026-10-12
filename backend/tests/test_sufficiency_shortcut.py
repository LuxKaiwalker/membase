import datetime as dt

from app.models import MemoryUnit
from app.retrieval import agent as agent_module
from app.retrieval.agent import HIGH_CONFIDENCE_FUSED_SCORE, HIGH_CONFIDENCE_MARGIN, _is_high_confidence, automatic_search
from app.retrieval.pipeline import ScoredUnit
from tests.fakes import FakeLLMProvider

# Streamlining phase (see docs/DECISIONS.md "Deterministic sufficiency
# short-circuit"): a conservative, deterministic bypass around the
# expensive evaluate_sufficiency LLM call for the unambiguous case --
# never fires unless the top candidate is both overwhelmingly dominant
# and backed by real (non-degraded) semantic analysis.


def _unit(enrichment_status: str = "ok") -> MemoryUnit:
    return MemoryUnit(
        authoritative_content="x",
        semantic={"enrichment_status": enrichment_status, "meaning": "x"},
        created_at=dt.datetime.now(dt.UTC),
    )


def _scored(fused_score: float, enrichment_status: str = "ok") -> ScoredUnit:
    return ScoredUnit(unit=_unit(enrichment_status), vector_score=fused_score, fts_score=fused_score, fused_score=fused_score)


def test_empty_ranked_is_never_high_confidence():
    assert _is_high_confidence([]) is False


def test_single_dominant_candidate_above_threshold_is_high_confidence():
    assert _is_high_confidence([_scored(HIGH_CONFIDENCE_FUSED_SCORE)]) is True


def test_single_candidate_below_threshold_is_not_high_confidence():
    assert _is_high_confidence([_scored(HIGH_CONFIDENCE_FUSED_SCORE - 0.01)]) is False


def test_two_candidates_too_close_together_is_not_high_confidence():
    top = HIGH_CONFIDENCE_FUSED_SCORE + 0.1
    runner_up = top - (HIGH_CONFIDENCE_MARGIN - 0.01)
    assert _is_high_confidence([_scored(top), _scored(runner_up)]) is False


def test_two_candidates_with_wide_enough_margin_is_high_confidence():
    top = HIGH_CONFIDENCE_FUSED_SCORE + 0.1
    runner_up = top - (HIGH_CONFIDENCE_MARGIN + 0.01)
    assert _is_high_confidence([_scored(top), _scored(runner_up)]) is True


def test_degraded_evidence_is_never_high_confidence_even_at_a_top_score():
    assert _is_high_confidence([_scored(0.99, enrichment_status="unavailable")]) is False


async def test_automatic_search_skips_evaluate_call_when_high_confidence(db, monkeypatch):
    winner = _scored(0.95)

    async def fake_hybrid_search(db, llm, query, limit=5, query_vector=None):
        return [winner]

    async def fake_hybrid_search_chunks(db, llm, query, limit=5, query_vector=None):
        return []

    monkeypatch.setattr(agent_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(agent_module, "hybrid_search_chunks", fake_hybrid_search_chunks)

    llm = FakeLLMProvider()
    result = await automatic_search(db, llm, "anything")

    assert "evaluate" not in llm.calls
    assert result.ranked == [winner]


async def test_automatic_search_still_evaluates_when_not_high_confidence(db, monkeypatch):
    weak = _scored(0.3)

    async def fake_hybrid_search(db, llm, query, limit=5, query_vector=None):
        return [weak]

    async def fake_hybrid_search_chunks(db, llm, query, limit=5, query_vector=None):
        return []

    monkeypatch.setattr(agent_module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(agent_module, "hybrid_search_chunks", fake_hybrid_search_chunks)

    llm = FakeLLMProvider()
    await automatic_search(db, llm, "anything")

    assert "evaluate" in llm.calls
