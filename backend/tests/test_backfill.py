"""scripts/reembed.py is a standalone script (not an importable package),
so it's exercised here as a subprocess against the real test DB -- this
also verifies it behaves correctly as actually invoked, env vars and all,
rather than only its internal functions in isolation.
"""

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text as sql_text

from app.config import get_settings
from tests.helpers import create_capture_and_finish

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# scripts/reembed.py always uses the real configured LLMProvider (that's
# the point of an operational backfill tool) -- there's no fake to inject
# into a subprocess, so this module is a real-Ollama integration test.
pytestmark = pytest.mark.integration


def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{get_settings().ollama_host}/api/tags", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_ollama():
    if not _ollama_reachable():
        pytest.skip("Ollama is not reachable; skipping real-LLM integration tests")


def _run_reembed(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "reembed.py"), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=os.environ,
    )


async def test_reembed_backfills_a_null_embedding(db, fake_llm):
    capture = await create_capture_and_finish(db, fake_llm, text="A memory to test backfill.")
    unit = capture.units[0]
    assert unit.status == "ready", "test assumes no clarification fired for this fake"

    db.execute(
        sql_text("UPDATE memory_units SET embedding = NULL, embedding_model = NULL, embedding_dim = NULL WHERE id = :id"),
        {"id": unit.id},
    )
    db.commit()

    result = _run_reembed("--dry-run")
    assert result.returncode == 0, result.stderr
    # Exact match, not a loose "stale / " substring check -- the `db`
    # fixture truncates memory_units before this test, so exactly one
    # stale row out of one total is the only correct outcome; a looser
    # check would also pass if the tool wrongly found zero stale rows.
    assert "memory_units: 1 stale / 1 total" in result.stdout

    result = _run_reembed()
    assert result.returncode == 0, result.stderr
    assert "re-embedded" in result.stdout

    db.refresh(unit)
    assert unit.embedding is not None
    assert unit.embedding_model is not None


async def test_reembed_is_idempotent_on_second_run(db, fake_llm):
    await create_capture_and_finish(db, fake_llm, text="Another memory.")

    first = _run_reembed()
    assert first.returncode == 0, first.stderr
    second = _run_reembed()
    assert second.returncode == 0, second.stderr
    # Exact match on the memory_units line specifically -- the previous
    # "0/0" fallback could be satisfied by an unrelated table's stat
    # line elsewhere in the output, masking a real idempotency
    # regression in memory_units specifically.
    assert "memory_units: 0 stale / 1 total" in second.stdout
