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
            # 180s is comfortably above the worst legitimate case (two
            # sequential LLM calls in one ingestion transaction, each up
            # to the 60s OllamaProvider httpx timeout), so this should
            # only ever fire for a genuinely stuck transaction.
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
