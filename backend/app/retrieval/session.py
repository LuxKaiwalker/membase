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
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.llm.base import LLMError, LLMProvider
from app.models import RecallSession
from app.retrieval.agent import automatic_search
from app.retrieval.cues import extract_cues
from app.retrieval.discriminate import (
    apply_negative_cues,
    apply_positive_cue_boost,
    generate_discriminating_question,
    is_ambiguous,
    narrow_to_dominant_group,
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
        }
        for s in ranked
    ]


def _snapshot_chunk_sources(ranked_chunks) -> list[dict]:
    return [
        {
            "id": str(c.chunk.id),
            "capture_id": str(c.chunk.capture_id),
            "attachment_filename": c.chunk.attachment.original_filename,
            "page_number": c.chunk.page_number,
            "excerpt": c.chunk.text[:500],
            "vector_score": c.vector_score,
            "fts_score": c.fts_score,
            "fused_score": c.fused_score,
        }
        for c in ranked_chunks
    ]


def _finish(session: RecallSession, status: str, message: str) -> None:
    session.status = status
    session.current_question = None
    _append_message(session, "system", message)


async def _run_round(db: Session, llm: LLMProvider, session: RecallSession) -> None:
    """Execute one user-mediated retrieval round: bounded automatic
    search seeded with every cue accumulated so far, then decide whether
    to answer, ask a discriminating question, or give up (spec section
    21).
    """
    session.round_count += 1
    if session.round_count > MAX_SESSION_ROUNDS:
        _finish(session, STATUS_EXHAUSTED, ABSTENTION_MESSAGE)
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

    negatives = _negative_cue_texts(session)
    ranked = apply_negative_cues(result.ranked, negatives)
    ranked = apply_positive_cue_boost(ranked, extra_queries)
    ranked = sorted(ranked, key=lambda s: s.fused_score, reverse=True)
    session.last_candidate_count = len(ranked) + len(result.ranked_chunks)

    if not ranked and not result.ranked_chunks:
        _finish(session, STATUS_EXHAUSTED, ABSTENTION_MESSAGE)
        db.commit()
        return

    if is_ambiguous(ranked):
        question = generate_discriminating_question(ranked, already_asked=session.asked_questions)
        if question is not None:
            session.status = STATUS_NEEDS_USER_CUE
            session.current_question = question
            session.asked_questions = [*session.asked_questions, question]
            _append_message(session, "system", question)
            db.commit()
            return
        # No remaining angle distinguishes the candidates -- fall through
        # and answer with the best evidence available rather than loop.
    else:
        # A clear winner exists: don't let an unrelated, lower-scoring
        # group that merely cleared the relevance bar keep cluttering the
        # answer-generation evidence (spec section 12 governs the
        # *ambiguous* case; once resolved, there's no decision left for
        # the answering step to make between groups).
        ranked = narrow_to_dominant_group(ranked)

    context = sibling_context(db, ranked)
    answer_text, grounded = await generate_answer(llm, session.original_query, ranked, context, result.ranked_chunks)
    if grounded:
        session.final_answer = answer_text
        session.final_sources = _snapshot_sources(ranked)
        session.final_chunk_sources = _snapshot_chunk_sources(result.ranked_chunks)
        _finish(session, STATUS_SUCCESS, answer_text)
        db.commit()
        return

    # Candidates exist but don't yet clearly answer the question (and
    # either aren't ambiguous, or were ambiguous with no fresh angle
    # left). Ask one more discriminating question if a new angle exists;
    # otherwise this line of inquiry is genuinely exhausted (spec section
    # 11: a weak match is not proof retrieval is impossible, but it isn't
    # grounds to keep asking forever either).
    question = generate_discriminating_question(ranked, already_asked=session.asked_questions)
    if question is not None:
        session.status = STATUS_NEEDS_USER_CUE
        session.current_question = question
        session.asked_questions = [*session.asked_questions, question]
        _append_message(session, "system", question)
    else:
        _finish(session, STATUS_EXHAUSTED, ABSTENTION_MESSAGE)
    db.commit()


async def start_session(db: Session, llm: LLMProvider, query: str) -> RecallSession:
    query = query.strip()
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

    _append_message(session, "user", message)
    cues = await extract_cues(llm, message, pending_question=session.current_question)
    round_number = session.round_count + 1
    session.cues = [
        *session.cues,
        *({**c, "round": round_number, "created_at": _now()} for c in cues),
    ]

    await _run_round(db, llm, session)
    return session


def cancel_session(db: Session, session: RecallSession) -> RecallSession:
    if session.status in _TERMINAL_STATUSES:
        raise RecallSessionError(f"Recall session is already '{session.status}'")
    session.status = STATUS_CANCELLED
    session.current_question = None
    db.commit()
    return session
