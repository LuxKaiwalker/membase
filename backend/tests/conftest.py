import pytest
from sqlalchemy import text as sql_text

from app.db import get_sessionmaker
from tests.fakes import FakeLLMProvider


# Every model table, truncated before each test. A table missing from this
# list leaks state across tests -- which for the graph tables would be
# particularly confusing, since a stale `memory_entities` row silently changes
# the document-frequency figures every association weight is computed from.
# `memory_entities` is listed explicitly even though CASCADE from
# `memory_units` would clear its dependent link rows: entity rows themselves
# have no FK to anything and would otherwise survive forever.
_TRUNCATE_TABLES = (
    "captures",
    "memory_units",
    "attachments",
    "attachment_chunks",
    "recall_sessions",
    "memory_entities",
    "unit_entity_links",
    "entity_relations",
    "unit_associations",
)


@pytest.fixture()
def db():
    session = get_sessionmaker()()
    session.execute(sql_text(f"TRUNCATE {', '.join(_TRUNCATE_TABLES)}"))
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def fake_llm():
    return FakeLLMProvider()
