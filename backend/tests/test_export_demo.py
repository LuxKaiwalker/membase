"""Tests for the public demo snapshot exporter (scripts/export_demo.py).

Three things are worth pinning here, and they are the three ways this
mechanism can break without anyone noticing:

  1. It must refuse to run against any database but `membase_demo`. The
     output directory gets published to a public web server; pointing it
     at the owner's real store is the one unrecoverable mistake
     available.
  2. The URL -> filename rule must match app/static/core.js's
     `snapshotKey()` exactly. Two copies of one rule in two languages is
     a drift risk taken deliberately (the alternative is a build step
     and a bundler, which this repo does not have), so a test compares
     them character-for-character instead of trusting a comment.
  3. The constants the exported request space is built from must match
     the values the frontend builds its request URLs from. A snapshot
     that answers `max_nodes=400` while the page asks for `max_nodes=500`
     is a 404 on every graph view, in public mode only -- exactly the
     kind of thing nobody runs into until it is deployed.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_demo  # noqa: E402

STATIC_DIR = REPO_ROOT / "backend" / "app" / "static"
CORE_JS = STATIC_DIR / "core.js"


# --- 1. the database guard ---------------------------------------------


def test_refuses_a_database_that_is_not_the_demo_store(db, capsys):
    # The `db` fixture is the dev database ("membase"), which is exactly
    # the kind of near-miss this guard exists for.
    assert export_demo._assert_demo_database(db) is False
    err = capsys.readouterr().err
    assert "refusing to export" in err
    assert "membase_demo" in err


# --- 2. the URL -> filename rule ---------------------------------------

URL_CASES = [
    "/api/captures",
    "/api/graph/stats",
    "/api/graph?max_nodes=400&hops=1",
    # Deliberately out of alphabetical order: both sides sort, so these
    # two must land on the same file.
    "/api/graph?hops=2&max_nodes=400",
    "/api/graph?max_nodes=400&hops=2&center_unit_id=2b0f6a1c-0000-4000-8000-000000000001",
    "/api/graph/entities/2b0f6a1c-0000-4000-8000-000000000002",
    "/api/graph/units/2b0f6a1c-0000-4000-8000-000000000003/associations?hops=1&limit=12",
    "/api/graph/units/2b0f6a1c-0000-4000-8000-000000000003/associations?hops=1",
    "/api/episodes?limit=5",
    "/api/episodes?limit=25&offset=0&include_undated=true",
    "/api/episodes/for-unit/2b0f6a1c-0000-4000-8000-000000000004",
    "/api/recall/sessions/2b0f6a1c-0000-4000-8000-000000000005",
    "/api/memory-units/2b0f6a1c-0000-4000-8000-000000000006",
]


def test_query_parameter_order_does_not_change_the_file():
    assert export_demo.snapshot_path("/api/graph?max_nodes=400&hops=1") == export_demo.snapshot_path(
        "/api/graph?hops=1&max_nodes=400"
    )


def test_paths_are_relative_and_contain_no_query_punctuation():
    for url in URL_CASES:
        path = export_demo.snapshot_path(url)
        assert path.startswith("api/")
        assert not path.startswith("/")
        assert ".." not in path
        for char in "?&=":
            assert char not in path, (url, path)


def _run_core_js_snapshot_key(urls: list[str]) -> list[str]:
    """Execute core.js's own `snapshotKey` over `urls`.

    The function is lifted out of the shipped file by brace matching
    rather than reimplemented here -- a paraphrase would pass this test
    while the real page still asked for the wrong filename. node is not
    a dependency of this project (no npm, no bundler, by repo rule); it
    is used only as a way to run the real function, so these tests skip
    where it is absent.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; cannot execute core.js's snapshotKey")

    source = CORE_JS.read_text()
    start = source.index("function snapshotKey(path)")
    depth = 0
    end = None
    for i in range(source.index("{", start), len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, "could not find the end of snapshotKey() in core.js"

    script = (
        source[start:end]
        + "\nconsole.log(JSON.stringify("
        + json.dumps(urls)
        + ".map(snapshotKey)));\n"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_snapshot_path_matches_core_js_snapshot_key():
    assert _run_core_js_snapshot_key(URL_CASES) == [
        export_demo.snapshot_path(url) for url in URL_CASES
    ]


def test_attachment_bytes_keep_their_own_path_with_no_json_suffix():
    """An attachment is not JSON and must not get a .json suffix.

    The exporter writes these in its own loop rather than through
    snapshot_path(), so the agreement that matters is between that loop
    and the JS -- hence asserting on the literal path the exporter
    builds, not on snapshot_path().
    """
    attachment_id = "2b0f6a1c-0000-4000-8000-00000000000a"
    url = f"/api/attachments/{attachment_id}/file"

    assert _run_core_js_snapshot_key([url]) == [f"api/attachments/{attachment_id}/file"]


# --- 3. constants shared with the frontend -----------------------------


def _js_number(path: Path, pattern: str) -> int:
    match = re.search(pattern, path.read_text())
    assert match, f"{pattern!r} not found in {path.name}"
    return int(match.group(1))


def test_explorer_constants_match_the_exported_request_space():
    explorer = STATIC_DIR / "pages" / "explorer.js"
    assert export_demo.EXPLORER_MAX_NODES == _js_number(explorer, r"const MAX_NODES = (\d+)")
    # The associations fetch the explorer's inspector makes.
    assert re.search(
        rf"associations\?hops=\$\{{state\.hops\}}&limit={export_demo.EXPLORER_ASSOC_LIMIT}\b",
        explorer.read_text(),
    ), "explorer.js's associations limit no longer matches EXPLORER_ASSOC_LIMIT"


def test_overview_episode_limit_matches():
    overview = STATIC_DIR / "pages" / "overview.js"
    assert export_demo.OVERVIEW_EPISODE_LIMIT == _js_number(
        overview, r"RECENT_EPISODE_LIMIT = (\d+)"
    )


def test_timeline_page_sizes_match_the_template():
    options = re.findall(
        r'<option value="(\d+)"',
        (REPO_ROOT / "backend" / "app" / "templates" / "timeline.html").read_text(),
    )
    assert tuple(int(v) for v in options) == export_demo.TIMELINE_PAGE_SIZES


def test_hop_choices_cover_the_backend_ceiling():
    from app.memory import graph as memory_graph

    assert max(export_demo.HOP_CHOICES) == memory_graph.MAX_HOPS
    assert min(export_demo.HOP_CHOICES) == 1


def test_snapshot_base_is_where_the_build_publishes_the_snapshot():
    """core.js reads /demo/; build_frontend.py must write dist/demo."""
    assert 'SNAPSHOT_BASE_DEFAULT = "/demo"' in CORE_JS.read_text()
    build = (REPO_ROOT / "scripts" / "build_frontend.py").read_text()
    assert 'DIST_DIR / "demo"' in build
    assert "PUBLIC_DEMO_DIR" in build
