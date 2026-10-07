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
            # 450s is comfortably above the worst legitimate case: up to
            # five sequential LLM calls can now happen in one ingestion
            # transaction -- generate_questions, enrich()'s own three
            # sub-calls (narrative/entities/retrieval-aids, see
            # app/ingestion/enrichment.py), and embed -- each up to the
            # configured llm_timeout_seconds (90s by default, see
            # config.py) -- 5*90s=450s, so this should only ever fire for
            # a genuinely stuck transaction, not a legitimately slow one.
            connect_args={"options": "-c idle_in_transaction_session_timeout=450000"},
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
