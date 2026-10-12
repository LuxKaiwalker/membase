import pytest
from sqlalchemy import text as sql_text

from app.db import get_sessionmaker
from app.rate_limit import reset_all as reset_rate_limits
from tests.fakes import FakeLLMProvider


@pytest.fixture()
def db():
    session = get_sessionmaker()()
    session.execute(
        sql_text(
            "TRUNCATE captures, memory_units, attachments, attachment_chunks, recall_sessions, auth_tokens"
        )
    )
    session.commit()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    # /api/auth/login's rate limiter (app/rate_limit.py) is process-
    # global state -- without this, a test that deliberately fails
    # login a few times would leak a lockout into whichever test runs
    # next.
    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.fixture()
def fake_llm():
    return FakeLLMProvider()
