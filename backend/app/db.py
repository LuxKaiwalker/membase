from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_settings().database_url,
            pool_pre_ping=True,
            # Safety net against a request that holds a transaction open
            # indefinitely (e.g. stuck/abandoned mid-request) silently
            # blocking every later request that needs the same rows --
            # observed for real: a stuck connection blocked a later
            # UPDATE for 5+ minutes with no error or timeout anywhere.
            # The streamlined pipeline (see docs/DECISIONS.md "Collapse
            # ingestion LLM calls") makes at most one LLM call
            # (analyze_capture/reanalyze_unit) per open transaction --
            # embedding runs separately, outside any transaction, via the
            # processing-lease pattern (see _claim_processing). 180s is
            # comfortably above that one call's llm_timeout_seconds (90s
            # by default, see config.py) with margin, so this should only
            # ever fire for a genuinely stuck transaction, not a
            # legitimately slow one.
            connect_args={"options": "-c idle_in_transaction_session_timeout=180000"},
        )
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
