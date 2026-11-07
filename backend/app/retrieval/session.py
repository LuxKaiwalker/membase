"""Iterative recall as a first-class, persisted process (spec sections
1-11): a RecallSession tracks a user's recall attempt across possibly
several rounds, accumulating cues and search hypotheses, rather than
treating every query as one isolated search-and-answer.

State machine (spec section 21):

    start_session()  --(round)-->  needs_user_cue | success | exhausted | error
                                         |
                                    continue_session() --(round)--> (same set)

    cancel_session() -> cancelled, from any non-terminal state.

Each round runs the SAME bounded automatic search loop used by the
one-shot agent (app/retrieval/agent.py:automatic_search) -- that loop
itself stays bounded to MAX_AGENT_ITERATIONS; what's different here is
that *rounds* (user-mediated) are not bounded the same way (spec section
10), and each round's search is seeded with every positive cue
accumulated so far, not just the original query.

Nothing in this module ever writes to Capture/MemoryUnit (spec section
5/29): cues and hypotheses live only on the RecallSession row.
"""

import logging
import time
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.config import get_settings
from app.llm.base import LLMError, LLMProvider
from app.models import RecallSession
from app.retrieval.agent import automatic_search
from app.retrieval.associative import (
    RetrievalBudget,
    RetrievalTrace,
    apply_budget,
    build_direct_candidates,
    collect_evidence,
    expand_with_associations,
    rerank,
    should_expand,
)
from app.retrieval.cues import extract_cues
from app.retrieval.discriminate import (
    apply_negative_cues,
    apply_positive_cue_boost,
    is_ambiguous,
    narrow_to_dominant_group,
    select_question,
)
from app.retrieval.evidence import (
    ROUTE_ASSOCIATION,
    format_evidence_block,
    overall_confidence,
)
from app.retrieval.pipeline import ABSTENTION_MESSAGE, generate_answer, sibling_context

logger = logging.getLogger(__name__)

# Safeguard against a pathological runaway conversation (spec section
# 10/36) -- deliberately generous, NOT a UX-facing "you get N questions"
# limit. For comparison, MAX_AGENT_ITERATIONS (agent.py) bounds the fast
# automatic sub-loop within a single round to 3.
MAX_SESSION_ROUNDS = 12

STATUS_ACTIVE = "active"
STATUS_NEEDS_USER_CUE = "needs_user_cue"
STATUS_SUCCESS = "success"
STATUS_EXHAUSTED = "exhausted"
STATUS_CANCELLED = "cancelled"
STATUS_ERROR = "error"

_TERMINAL_STATUSES = (STATUS_SUCCESS, STATUS_EXHAUSTED, STATUS_CANCELLED, STATUS_ERROR)


class RecallSessionError(Exception):
    """Raised for an invalid operation on a session (e.g. continuing one
    that already reached a terminal state)."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _append_message(session: RecallSession, role: str, text: str) -> None:
    session.messages = [*session.messages, {"role": role, "text": text, "created_at": _now()}]


def _positive_cue_texts(session: RecallSession) -> list[str]:
    seen: dict[str, None] = {}
    for c in session.cues:
        if c["polarity"] == "positive":
            seen.setdefault(c["text"], None)
    return list(seen)


def _negative_cue_texts(session: RecallSession) -> list[str]:
    return [c["text"] for c in session.cues if c["polarity"] == "negative"]


def _all_cue_texts(session: RecallSession) -> list[str]:
    """Every value the user has spoken about this session, either polarity.

    Used to stop the question selector asking about something the user just
    told it -- a confirmation and a denial are equally "already answered".
    """
    return [c["text"] for c in session.cues if c.get("text")]


def _snapshot_attachment(a) -> dict:
    return {
        "id": str(a.id),
        "original_filename": a.original_filename,
        "content_type": a.content_type,
        "size_bytes": a.size_bytes,
        "extraction_status": a.extraction_status,
    }


def _snapshot_sources(ranked) -> list[dict]:
    return [
        {
            "id": str(s.unit.id),
            "capture_id": str(s.unit.capture_id),
            "content": s.unit.authoritative_content,
            "semantic": s.unit.semantic,
            "created_at": s.unit.created_at.isoformat(),
            "vector_score": s.vector_score,
            "fts_score": s.fts_score,
            "fused_score": s.fused_score,
            "attachments": [_snapshot_attachment(a) for a in s.unit.capture.attachments],
        }
        for s in ranked
    ]


def _snapshot_chunk_sources(ranked_chunks) -> list[dict]:
    return [
        {
            "id": str(c.chunk.id),
            "capture_id": str(c.chunk.capture_id),
            "attachment_id": str(c.chunk.attachment_id),
            "attachment_filename": c.chunk.attachment.original_filename,
            "page_number": c.chunk.page_number,
            "excerpt": c.chunk.text[:500],
            "vector_score": c.vector_score,
            "fts_score": c.fts_score,
            "fused_score": c.fused_score,
        }
        for c in ranked_chunks
    ]


def _finish(session: RecallSession, status: str, message: str, *, reason: str | None = None) -> None:
    session.status = status
    session.current_question = None
    if reason is not None:
        session.abstention_reason = reason
    _append_message(session, "system", message)


def _record_round(session: RecallSession, trace: RetrievalTrace, extra: dict | None = None) -> None:
    """Append this round's retrieval trace to the session's inspectable trail.

    Appended rather than replaced: the point of the trail is that a user can
    see how the search narrowed across rounds, which a single
    latest-round-only field could not show.
    """
    entry = {"round": session.round_count, **trace.as_dict(), **(extra or {})}
    session.strategy_trail = [*session.strategy_trail, entry]


def _ask(
    session: RecallSession, choice, trace: RetrievalTrace, candidates: list | None = None
) -> None:
    """Put the session into needs_user_cue with one concrete question.

    `candidates` is snapshotted into `final_evidence` even though no answer has
    been produced yet. That is deliberate: while a question is pending, the
    candidate set *is* the result so far, and a UI that cannot show "here are
    the five memories I am choosing between" forces the user to answer a
    question about candidates they cannot see. The snapshot is replaced when
    the round that actually answers runs.
    """
    session.status = STATUS_NEEDS_USER_CUE
    session.current_question = choice.question
    session.asked_questions = [*session.asked_questions, choice.question]
    if candidates is not None:
        session.final_evidence = [c.as_dict() for c in candidates]
        session.final_confidence = overall_confidence(candidates)
    _append_message(session, "system", choice.question)
    _record_round(session, trace, {"question": choice.as_dict()})


async def _run_round(db: Session, llm: LLMProvider, session: RecallSession) -> None:
    """Execute one user-mediated retrieval round: bounded automatic
    search seeded with every cue accumulated so far, then decide whether
    to answer, ask a discriminating question, or give up (spec section
    21).
    """
    round_started = time.monotonic()
    session.round_count += 1
    if session.round_count > MAX_SESSION_ROUNDS:
        _finish(
            session,
            STATUS_EXHAUSTED,
            ABSTENTION_MESSAGE,
            reason=f"Reached the {MAX_SESSION_ROUNDS}-round safeguard without resolving the memory.",
        )
        db.commit()
        return

    extra_queries = _positive_cue_texts(session)
    try:
        result = await automatic_search(
            db, llm, session.original_query, extra_queries=extra_queries, tried=set(session.tried_queries)
        )
    except LLMError:
        logger.warning("RecallSession %s: search failed", session.id)
        _finish(session, STATUS_ERROR, "Something went wrong while searching. Please try again.")
        db.commit()
        return

    session.tried_queries = sorted(result.tried_queries)
    for h in result.hypotheses:
        if h not in [x["text"] for x in session.hypotheses]:
            session.hypotheses = [*session.hypotheses, {"text": h, "round": session.round_count, "created_at": _now()}]

    # Everything below is pure-Python candidate reasoning plus one more
    # LLM call (generate_answer, which already degrades LLMError to
    # grounded=False internally) -- wrapped broadly because an unexpected
    # bug here must not propagate uncaught: that would leave the session
    # in a non-terminal status with nothing committed, the same failure
    # mode the ingestion pipeline's _run_full_enrichment already guards
    # against (see docs/DECISIONS.md). Degrading to STATUS_ERROR keeps
    # the session inspectable/retryable instead of a raw 500.
    try:
        budget = RetrievalBudget()
        trace = RetrievalTrace(
            strategies=["hybrid_search", "evaluate_reformulate"],
            hypotheses=list(result.hypotheses),
            budget={
                "max_association_hops": budget.max_association_hops,
                "max_expanded_candidates": budget.max_expanded_candidates,
                "max_total_candidates": budget.max_total_candidates,
                "max_session_rounds": MAX_SESSION_ROUNDS,
            },
        )

        negatives = _negative_cue_texts(session)
        before_negatives = {str(s.unit.id) for s in result.ranked}
        ranked = apply_negative_cues(result.ranked, negatives)
        # Record what the user's own "no, it wasn't that" removed. Excluding
        # silently would make a narrowing session look like it simply stopped
        # finding things, when in fact the user ruled those candidates out.
        removed = before_negatives - {str(s.unit.id) for s in ranked}
        trace.ruled_out = [
            {
                "unit_id": str(s.unit.id),
                "content": s.unit.authoritative_content[:200],
                "reason": "you ruled this out",
            }
            for s in result.ranked
            if str(s.unit.id) in removed
        ]

        ranked = apply_positive_cue_boost(ranked, extra_queries)
        ranked = sorted(ranked, key=lambda s: s.fused_score, reverse=True)

        if not ranked and not result.ranked_chunks:
            trace.elapsed_ms = int((time.monotonic() - round_started) * 1000)
            session.last_candidate_count = 0
            # Say why association exploration did not run, rather than leaving
            # the field null and making the reader guess whether it was skipped
            # deliberately or forgotten. This branch returns before
            # should_expand() is consulted, so record the same reason it would
            # have given.
            trace.expansion_skipped_reason = (
                "nothing matched directly, so there was nothing to explore from"
            )
            _record_round(session, trace)
            _finish(
                session,
                STATUS_EXHAUSTED,
                ABSTENTION_MESSAGE,
                reason=(
                    "No stored memory cleared the relevance bar for this query or for any "
                    "reformulation of it."
                ),
            )
            db.commit()
            return

        # --- Retrieval level 6: associative exploration. Adaptive -- a query
        # already answered decisively by direct matching skips this entirely
        # (see app/retrieval/associative.py::should_expand), so a clear query
        # costs exactly what it did before this feature existed.
        candidates = build_direct_candidates(ranked)
        trace.direct_candidates = len(candidates)
        expand, skip_reason = should_expand(candidates)
        if expand:
            trace.strategies.append("association_expansion")
            expanded = expand_with_associations(db, candidates, budget)
            candidates.extend(expanded)
            trace.expanded_candidates = len(expanded)
        else:
            trace.expansion_skipped_reason = skip_reason

        trace.strategies.append("evidence_collection")
        collect_evidence(
            db,
            candidates,
            query=session.original_query,
            positive_cues=extra_queries,
            chunks=result.ranked_chunks,
        )
        candidates = apply_budget(rerank(candidates), budget)
        session.last_candidate_count = len(candidates) + len(result.ranked_chunks)

        # Ambiguity detection and dominant-group narrowing still operate on
        # the ScoredUnit view of *directly matched* candidates. Both are
        # defined over things that actually matched the query: an associated
        # lead is not a competing answer to disambiguate between, it is
        # supporting context for whichever direct candidate wins.
        direct_ranked = [c.direct for c in candidates if c.direct is not None]

        if is_ambiguous(direct_ranked):
            trace.strategies.append("ambiguity_narrowing")
            choice = select_question(
                direct_ranked,
                already_asked=session.asked_questions,
                known_values=_all_cue_texts(session),
            )
            if choice is not None:
                trace.elapsed_ms = int((time.monotonic() - round_started) * 1000)
                _ask(session, choice, trace, candidates)
                db.commit()
                return
            # No remaining attribute separates the candidates informatively --
            # fall through and answer with the best evidence available rather
            # than loop, or ask something that would gain nothing.
        elif direct_ranked:
            # A clear winner exists: don't let an unrelated, lower-scoring
            # group that merely cleared the relevance bar keep cluttering
            # the answer-generation evidence (spec section 12 governs the
            # *ambiguous* case; once resolved, there's no decision left
            # for the answering step to make between groups). Associated
            # candidates survive this: they were reached FROM the winner, so
            # they are its context rather than a rival group.
            winners = {str(s.unit.id) for s in narrow_to_dominant_group(direct_ranked)}
            candidates = [
                c for c in candidates if c.route == ROUTE_ASSOCIATION or c.unit_id in winners
            ]
            direct_ranked = [c.direct for c in candidates if c.direct is not None]

        context = sibling_context(db, direct_ranked)
        evidence_text = format_evidence_block(candidates, context, result.ranked_chunks)
        answer_text, grounded = await generate_answer(
            llm,
            session.original_query,
            direct_ranked,
            context,
            result.ranked_chunks,
            evidence=evidence_text,
        )
        trace.elapsed_ms = int((time.monotonic() - round_started) * 1000)

        if grounded:
            session.final_answer = answer_text
            session.final_sources = _snapshot_sources(direct_ranked)
            session.final_chunk_sources = _snapshot_chunk_sources(result.ranked_chunks)
            session.final_evidence = [c.as_dict() for c in candidates]
            session.final_confidence = overall_confidence(candidates)
            session.abstention_reason = None
            _record_round(session, trace)
            _finish(session, STATUS_SUCCESS, answer_text)
            db.commit()
            return

        # Candidates exist but don't yet clearly answer the question (and
        # either aren't ambiguous, or were ambiguous with no fresh angle
        # left). Ask one more discriminating question if a new angle
        # exists; otherwise this line of inquiry is genuinely exhausted
        # (spec section 11: a weak match is not proof retrieval is
        # impossible, but it isn't grounds to keep asking forever either).
        choice = select_question(
                direct_ranked,
                already_asked=session.asked_questions,
                known_values=_all_cue_texts(session),
            )
        if choice is not None:
            _ask(session, choice, trace, candidates)
        else:
            session.final_evidence = [c.as_dict() for c in candidates]
            session.final_confidence = overall_confidence(candidates)
            _record_round(session, trace)
            _finish(
                session,
                STATUS_EXHAUSTED,
                ABSTENTION_MESSAGE,
                reason=(
                    "Memories were found, but none of them actually answers the question, and no "
                    "remaining detail would separate them."
                ),
            )
        db.commit()
    except Exception:
        logger.exception("RecallSession %s: unexpected failure deciding round outcome", session.id)
        _finish(session, STATUS_ERROR, "Something went wrong while searching. Please try again.")
        db.commit()


async def start_session(db: Session, llm: LLMProvider, query: str) -> RecallSession:
    query = query.strip()
    max_chars = get_settings().membase_max_text_chars
    if len(query) > max_chars:
        raise RecallSessionError(f"query exceeds max length of {max_chars} characters")
    session = RecallSession(original_query=query, status=STATUS_ACTIVE, messages=[])
    _append_message(session, "user", query)
    db.add(session)
    db.flush()

    if not query:
        _finish(session, STATUS_EXHAUSTED, ABSTENTION_MESSAGE)
        db.commit()
        return session

    await _run_round(db, llm, session)
    return session


async def continue_session(db: Session, llm: LLMProvider, session: RecallSession, message: str) -> RecallSession:
    """Add a user-mediated turn to an in-progress session: a direct
    answer to `current_question`, an unprompted additional recollection,
    or a correction (spec section 15/37) -- all are just more cues.
    """
    if session.status in _TERMINAL_STATUSES:
        raise RecallSessionError(f"Recall session is '{session.status}' and cannot continue")

    message = message.strip()
    if not message:
        raise RecallSessionError("Message must not be empty")
    max_chars = get_settings().membase_max_text_chars
    if len(message) > max_chars:
        raise RecallSessionError(f"message exceeds max length of {max_chars} characters")

    _append_message(session, "user", message)
    cues = await extract_cues(llm, message, pending_question=session.current_question)
    round_number = session.round_count + 1
    session.cues = [
        *session.cues,
        *({**c, "round": round_number, "created_at": _now()} for c in cues),
    ]

    await _run_round(db, llm, session)
    return session


def user_supplied_texts(session: RecallSession) -> set[str]:
    """Every piece of text this session can attribute directly to the
    user -- their own message turns and the cues extracted from them --
    as distinct from agent-generated hypotheses or system questions.
    Used to guard "save as memory" (see docs/DECISIONS.md): recall-
    session state is never promoted into a MemoryUnit automatically, and
    the one explicit action that lets a user promote *something* from a
    session must still only ever act on text the user actually said,
    never a system-generated hypothesis or question.
    """
    texts = {m["text"] for m in session.messages if m.get("role") == "user"}
    texts |= {c["text"] for c in session.cues}
    return texts


def cancel_session(db: Session, session: RecallSession) -> RecallSession:
    if session.status in _TERMINAL_STATUSES:
        raise RecallSessionError(f"Recall session is already '{session.status}'")
    session.status = STATUS_CANCELLED
    session.current_question = None
    db.commit()
    return session
