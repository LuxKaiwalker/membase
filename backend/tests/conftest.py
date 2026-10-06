import pytest
from sqlalchemy import text as sql_text

from app.db import get_sessionmaker
from tests.fakes import FakeLLMProvider


@pytest.fixture()
def db():
    session = get_sessionmaker()()
    session.execute(sql_text("TRUNCATE memories, attachments"))
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def fake_llm():
    return FakeLLMProvider()
