#!/usr/bin/env python3
"""Builds the static frontend published to GitHub Pages (see
docs/DEPLOYMENT.md "Publishing the frontend to GitHub Pages").

The three page templates under backend/app/templates/ are pure structure
with zero request-dependent data (app/web/routes.py renders each with an
empty context dict) -- confirmed, not assumed, by grep: no `{{ }}`
variable interpolation anywhere outside `{% block %}`/`{% extends %}`.
That makes them safe to render exactly once, at build time, with plain
Jinja2 (not FastAPI's Jinja2Templates wrapper, which needs a live
Request object this script has no reason to construct).

Output layout (in `dist/`, the GitHub Pages publish artifact):
    index.html              <- create.html (served at "/")
    recall/index.html       <- recall.html (served at "/recall/")
    captures/index.html     <- captures.html (served at "/captures/")
    static/app.js, app.css  <- copied verbatim from backend/app/static/
    .nojekyll               <- disables GitHub's Jekyll processing;
                               this is a plain static site, not a Jekyll
                               one, and Jekyll would otherwise try to
                               interpret stray "{{ }}"-looking content.
    CNAME                   <- copied from the repo root, if present
"""

import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "backend" / "app" / "templates"
STATIC_DIR = REPO_ROOT / "backend" / "app" / "static"
DIST_DIR = REPO_ROOT / "dist"

# template name -> output path (relative to DIST_DIR)
PAGES = {
    "create.html": "index.html",
    "recall.html": "recall/index.html",
    "captures.html": "captures/index.html",
    "docs.html": "docs/index.html",
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

    (DIST_DIR / ".nojekyll").touch()

    cname = REPO_ROOT / "CNAME"
    if cname.exists():
        shutil.copy(cname, DIST_DIR / "CNAME")
        print(f"copied {cname} -> dist/CNAME")


if __name__ == "__main__":
    main()
