#!/usr/bin/env python3
"""Builds the static frontend published to GitHub Pages (see
docs/DEPLOYMENT.md "Publishing the frontend to GitHub Pages").

The page templates under backend/app/templates/ are pure structure with
zero request-dependent data (rendered with an empty context dict, same
as the backend's own app/web/routes.py used to before it was removed --
see docs/DECISIONS.md "API-only backend: stop serving the frontend's own
HTML/static files"). The `{{ }}` interpolation they do contain reads
only template-local `{% set %}` data -- base.html's `sections` registry,
which drives the nav, the active-item marker and each page's script tag
-- never anything from a request. That makes them safe to render exactly
once, at build time, with plain Jinja2 (not FastAPI's Jinja2Templates
wrapper, which needs a live Request object this script has no reason to
construct).

Output layout (in `dist/`, the GitHub Pages publish artifact). Every app
page is an `app/<name>/index.html` so its clean URL works on GitHub
Pages with no server-side config (and `/app/` itself is the Overview):
    index.html                <- landing.html         ("/")
    app/index.html            <- overview.html        ("/app/")
    app/recall/index.html     <- recall.html          ("/app/recall/")
    app/explorer/index.html   <- explorer.html        ("/app/explorer/")
    app/timeline/index.html   <- timeline.html        ("/app/timeline/")
    app/library/index.html    <- library.html         ("/app/library/")
    app/evaluation/index.html <- evaluation.html      ("/app/evaluation/")
    app/store/index.html      <- store.html           ("/app/store/")
    static/app.css, core.js, landing.css, pages/*.js
                              <- copied verbatim from backend/app/static/
    static/evaluation-results.json
                              <- eval/strategy_benchmark_results.json, if
                                 it exists. The Evaluation page is the one
                                 section with no API behind it: the backend
                                 exposes no evaluation endpoint, and those
                                 numbers are produced offline by
                                 eval/strategy_benchmark.py (20+ minutes,
                                 an isolated database and a local model --
                                 see docs/EVALUATION.md). Publishing the
                                 benchmark artefact alongside the frontend
                                 is what lets the page render it as a
                                 same-origin static file instead of
                                 needing a service to ask. Absent on any
                                 machine that has never run the benchmark,
                                 which is not a build failure -- the page
                                 has its own empty state for exactly that.
    .nojekyll                 <- disables GitHub's Jekyll processing;
                                 this is a plain static site, not a Jekyll
                                 one, and Jekyll would otherwise try to
                                 interpret stray "{{ }}"-looking content.
    CNAME                     <- copied from the repo root, if present
"""

import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "backend" / "app" / "templates"
STATIC_DIR = REPO_ROOT / "backend" / "app" / "static"
DIST_DIR = REPO_ROOT / "dist"
BENCHMARK_RESULTS = REPO_ROOT / "eval" / "strategy_benchmark_results.json"
# The published snapshot of the synthetic demo store, which is what the
# app reads when nobody is signed in (see app/static/core.js's
# "public (snapshot) mode" and scripts/export_demo.py). Committed to the
# repo rather than generated during the build: the Pages workflow has no
# database, no model and no GPU, so it cannot regenerate this -- it can
# only copy it. Re-export and commit it when the demo corpus changes.
PUBLIC_DEMO_DIR = REPO_ROOT / "public-demo"

# template name -> output path (relative to DIST_DIR). The route registry:
# a new app section needs an entry here, an entry in base.html's own
# `sections` list (which drives the nav + its script tag), and its
# static/pages/<name>.js file.
PAGES = {
    "landing.html": "index.html",
    "overview.html": "app/index.html",
    "recall.html": "app/recall/index.html",
    "explorer.html": "app/explorer/index.html",
    "timeline.html": "app/timeline/index.html",
    "library.html": "app/library/index.html",
    "evaluation.html": "app/evaluation/index.html",
    "store.html": "app/store/index.html",
}


def main() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    for template_name, out_rel_path in PAGES.items():
        html = env.get_template(template_name).render()
        out_path = DIST_DIR / out_rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html)
        print(f"rendered {template_name} -> dist/{out_rel_path}")

    shutil.copytree(STATIC_DIR, DIST_DIR / "static")
    print(f"copied {STATIC_DIR} -> dist/static")

    # The Evaluation page's only data source (see the module docstring).
    # Optional on purpose: the benchmark is an offline 20+ minute run, so a
    # build on a machine that has never run it must still succeed -- the
    # page renders an empty state naming the command instead.
    if BENCHMARK_RESULTS.exists():
        shutil.copy(BENCHMARK_RESULTS, DIST_DIR / "static" / "evaluation-results.json")
        print(f"copied {BENCHMARK_RESULTS} -> dist/static/evaluation-results.json")
    else:
        print(f"no {BENCHMARK_RESULTS} -- skipping (Evaluation page shows its empty state)")

    # Published at /demo/, which is the one path app/static/core.js's
    # SNAPSHOT_BASE points at. Optional in the same way and for the same
    # reason as the benchmark results above: a build on a machine that
    # has never seeded the demo store must still produce a working site
    # (a signed-in user never reads these files at all), and a logged-out
    # visitor gets core.js's explicit "not part of the public snapshot"
    # message rather than a blank page.
    if PUBLIC_DEMO_DIR.is_dir():
        shutil.copytree(PUBLIC_DEMO_DIR, DIST_DIR / "demo")
        files = sum(1 for _ in (DIST_DIR / "demo").rglob("*") if _.is_file())
        print(f"copied {PUBLIC_DEMO_DIR} -> dist/demo ({files} files)")
    else:
        print(f"no {PUBLIC_DEMO_DIR} -- skipping (the public demo will be unavailable)")

    (DIST_DIR / ".nojekyll").touch()

    cname = REPO_ROOT / "CNAME"
    if cname.exists():
        shutil.copy(cname, DIST_DIR / "CNAME")
        print(f"copied {cname} -> dist/CNAME")


if __name__ == "__main__":
    main()
