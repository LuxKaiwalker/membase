# Deployment readiness audit

Scope: can this app be safely run somewhere other than a developer's
own `nix develop` shell on localhost? Specifically the scenario the
audit was commissioned for: **single user by design, but the HTTP
endpoint itself reachable from the public internet.**

Short answer: **everything is fixed except one thing, and that one
thing is a hard blocker.** There is currently no authentication
anywhere in this app. Do not put this on the public internet without
adding one of the options in "Auth system ideas" below first.

## What's already safe (verified, not assumed)

- **Postgres never listens on any network interface, even if the app
  were misconfigured.** `scripts/dev-db.sh` starts it with `-h ''`
  (TCP listener disabled) and a Unix-socket-only connection string
  (`DATABASE_URL` in `flake.nix`'s `shellHook`) -- there is no password
  to leak because there is no network path to the database at all,
  independent of anything the web app does.
- **Ollama is bound to `127.0.0.1:11434`** (`app/config.py`'s
  `ollama_host` default) and is never reconfigured anywhere in this
  codebase -- same isolation.
- **No secrets are committed.** `.gitignore` covers `.env` and `.data/`
  (the project-local Postgres data dir and uploads); `.env.example`
  contains no real credentials (now also fixed to show the current
  `gemma4:12b` default instead of a stale pre-migration model name).
- **No SQL injection surface.** Every raw `text()` query in
  `app/retrieval/pipeline.py` binds user-supplied values as parameters
  (`:q`, `:qv`, `:limit`); the only f-string-interpolated parts are
  fixed module-level constants, never request data.
- **Frontend XSS handling is consistent.** Every place `app/static/
  app.js` inserts server- or model-derived text via `innerHTML` runs it
  through `escapeHtml()` first; everywhere else uses `textContent`
  (which never parses HTML regardless). Checked across all three pages
  during this audit, including the newest additions (structure badges,
  attachment lists).
- **No path traversal in file uploads.** `process_upload()` (`app/
  ingestion/files.py`) builds the stored filename from
  `Path(filename).name` (strips any directory components) plus a fresh
  UUID prefix -- a filename like `../../etc/passwd` can't escape the
  per-capture upload directory.
- **No unauthenticated file-serving route exists at all.** `/static`
  only serves `app/static/` (the JS/CSS themselves); uploaded
  attachments are never served back over HTTP by any route -- they're
  write-only from the API's perspective. (This is also a minor product
  gap -- see "Known non-blocking gaps" below.)
- **Image decompression-bomb protection is intact.** Nothing in this
  codebase touches `PIL.Image.MAX_IMAGE_PIXELS`; Pillow's own default
  guard is untouched.
- **Unhandled exceptions don't leak stack traces.** No `debug=True`
  anywhere; FastAPI/Starlette's default `ServerErrorMiddleware` returns
  a generic 500 with no traceback to the client.

## Found and fixed this pass

- **No server-side length limit on any free-text input.** Unlike file
  uploads (`membase_max_upload_mb`), the `text` field on
  `POST /api/captures` had no size cap at all -- combined with this
  phase's chunking logic (`app/ingestion/analysis.py`), an
  unauthenticated caller posting a gigabytes-sized body wouldn't just
  waste storage, it would get chunked into many real LLM calls, turning
  one cheap HTTP request into a large, attacker-controlled compute bill.
  Fixed with a new `membase_max_text_chars` setting (50,000, generously
  above any real single memory) enforced in `create_capture()`,
  `correct_unit()`, `resolve_clarification()`,
  `RecallSession.start_session()`, and `.continue_session()` -- every
  free-text entry point, not just the main one. Covered by
  `tests/test_input_length_limits.py`.
- **`POST /api/recall`'s one-shot endpoint was missing the
  `RecallSessionError` handler** that `POST /api/recall/sessions` and
  `.../messages` already had -- an oversized query there would have
  produced an unhandled 500 instead of a clean 400. Fixed alongside the
  length-limit addition.
- **Stale `.env.example` values**: the generation model still named the
  pre-streamlining-phase `qwen2.5:14b-instruct-q4_K_M` instead of the
  current `gemma4:12b`; the upload-size example (25MB) didn't match
  `app/config.py`'s actual default (50MB). Both corrected.
- **No production run script existed** -- only `scripts/run-dev.sh`,
  which uses `--reload` (a file-watcher with no purpose once you're not
  editing code, and measurably slower per request). Added
  `scripts/run-prod.sh`: no `--reload`, explicitly one worker (this
  app's LLM provider and DB engine are process-local `@lru_cache`
  singletons, never designed for multiple workers sharing one process --
  and there's no throughput reason to want more than one anyway, since
  Ollama itself only serves one request at a time). Still binds
  `127.0.0.1` only -- see below for why that's deliberate, not a gap.

## The blocking issue: no authentication

There is no `Authorization` header check, no session, no password,
anywhere in this codebase (`grep` for `Authorization|HTTPBearer|
APIKeyHeader|OAuth|SessionMiddleware` across `app/` returns nothing).
Every route -- read your memories, delete a capture, attach a file,
trigger an LLM call -- is wide open to anyone who can reach the port.

This is fine, by design, for `127.0.0.1`-only local use (the only
thing that can reach it is something already running as you, on your
own machine). It stops being fine the moment anything -- a reverse
proxy, a port-forward, a cloud VM's public IP -- makes that port
reachable from the internet. At that point, with zero auth, you'd be
running an app that lets any anonymous visitor read your entire stored
memory, delete it, or run up your LLM compute, for free, with no sign-in
screen at all.

**This audit did not add an auth system.** Which mechanism fits depends
on how you want to actually use this day to day (type a password every
time? use a VPN client? both?) -- that's a real product decision, not
an "obviously correct" engineering default the way the fixes above
were. See the next section for concrete options and a recommendation.

## Auth system ideas

All of these assume the FastAPI app keeps binding `127.0.0.1` only
(via `scripts/run-prod.sh`) and something else -- a reverse proxy or a
VPN -- sits in front of it and is the thing actually exposed. **Never
bind this app itself to `0.0.0.0`** regardless of which option below
you pick; defense in depth is cheap and this app was never written or
tested with the assumption that arbitrary network peers can reach it
directly.

### Option A (recommended as the default): network-level isolation via Tailscale (or another WireGuard-based mesh VPN)

Put the machine running membase on a Tailscale network (or Headscale if
self-hosting the control plane) instead of exposing the port to the
raw internet at all. Access it from any of your own devices once
they're on the same tailnet, over Tailscale's own encrypted,
authenticated mesh. Tailscale's own "Funnel" feature can expose a
single app to the public internet *through* that same authenticated
network if you specifically want a plain URL reachable by anyone who
has it -- but for genuinely single-user use, you likely don't want
that; just keep it tailnet-only.

- **Pros:** zero application code changes -- nothing in this repo needs
  to know this exists. Strong security (WireGuard-grade encryption +
  device-level auth, not app-level). No password to phish, no session
  to hijack, no auth code to get wrong.
- **Cons:** every device you want to use this from needs the Tailscale
  client installed and logged into your account first -- you can't just
  hand someone a URL and have it work from an arbitrary browser with no
  setup (which, for a genuinely single-user app, is usually a feature,
  not a limitation).

### Option B: HTTP Basic Auth with one hardcoded credential pair

A single username/password pair from environment variables
(`MEMBASE_AUTH_USER` / `MEMBASE_AUTH_PASSWORD`), checked via a FastAPI
dependency applied to every route (`app.add_middleware` or a global
`Depends()`), using `secrets.compare_digest()` for the comparison (not
`==`, which leaks timing information byte-by-byte).

- **Pros:** genuinely simple -- maybe 20 lines of code, no new DB
  table, no session/cookie management, no CSRF surface (Basic Auth
  re-sends credentials on every request rather than relying on ambient
  cookie authority). Every browser already knows how to prompt for and
  cache Basic Auth credentials natively.
- **Cons:** credentials travel on every single request (base64-encoded,
  **not** encrypted -- this option is only acceptable over HTTPS,
  meaning you need a reverse proxy or equivalent TLS termination in
  front regardless). No real "log out" button (clearing it requires
  clearing browser-cached credentials, which varies by browser). No
  custom login page -- you get the browser's own native auth prompt.

### Option C: a login endpoint + signed cookie session

A `POST /login` that checks one password (same env-var-based
comparison as Option B) and, on success, sets an `HttpOnly`, `Secure`,
`SameSite=Strict` cookie containing a signed token (e.g. `itsdangerous`,
or FastAPI's own `SessionMiddleware` with a `SECRET_KEY` env var). Every
other route requires that cookie via a dependency.

- **Pros:** a real login page and logout button, nicer UX than a raw
  browser Basic Auth prompt, still only one password, no user table.
- **Cons:** meaningfully more code than B (a login route, a logout
  route, cookie/session configuration, a new `SECRET_KEY` to generate
  and keep secret). Needs CSRF protection for any state-changing
  request once you're relying on an ambient cookie rather than a header
  sent explicitly by the client code -- `POST`/`DELETE` routes called
  from `fetch()` with the cookie attached automatically by the browser
  are exactly the shape CSRF attacks target, so this isn't free.

### Option D: a single static bearer token

Like Option B, but the UI sends `Authorization: Bearer <token>`
(a long random value from an env var) on every `fetch()` call instead
of relying on the browser's native Basic Auth prompt or a cookie.

- **Pros:** no CSRF surface (same reasoning as Basic Auth -- the token
  is sent explicitly by your own JS, not ambient browser-attached
  cookie authority); works well if you ever want a non-browser client
  (a phone shortcut, a CLI) to talk to the same API.
- **Cons:** `app/static/app.js` needs to read the token from somewhere
  and attach it to every request (one `fetchWithTimeout()` call site to
  touch, but it's a real code change, not zero-touch like Option A);
  the token has to live *somewhere* client-side (localStorage is the
  obvious place, which means it's readable by any JS running on the
  page -- fine for a single trusted user, less fine if this UI ever
  embeds third-party content).

### What this audit recommends

**Option A (Tailscale) if you're the only person who'll ever use this
and don't mind installing a client.** It requires no code changes at
all and is strictly more secure than anything app-level, since there's
no password to guess, phish, or leak in a log line.

**Option B (HTTP Basic Auth) if you specifically want "visit a plain
HTTPS URL from any browser, type a password once."** It's the smallest
code change that closes the actual gap this audit found, and this
project's whole engineering philosophy this phase has been "the
smallest change that's actually correct" -- Basic Auth over HTTPS,
with a reverse proxy doing TLS termination in front of
`scripts/run-prod.sh`, is that for the auth gap specifically.

Not recommended for this project's stated scope: anything involving a
user table, OAuth/SSO, or multi-factor -- all solve problems this app
doesn't have (there is, by design, exactly one user), at a complexity
cost this phase's whole streamlining effort was specifically about
avoiding.

## Deployment checklist

1. Run `scripts/run-prod.sh` (not `run-dev.sh`) for anything other than
   active local development.
2. Put a reverse proxy in front for TLS -- Caddy is the least-effort
   option (automatic HTTPS via Let's Encrypt with a ~5-line Caddyfile);
   nginx + certbot works too if you already run nginx for other things.
3. Pick and implement one of the auth options above *before* the
   reverse proxy is reachable from the public internet, not after.
4. Confirm `app/main.py`'s `uvicorn.run(..., host="127.0.0.1", ...)`
   fallback and `scripts/run-prod.sh` both still say `127.0.0.1` --
   this should never change to `0.0.0.0` regardless of which auth
   option you pick; let the reverse proxy be the only thing with a
   public-facing socket.
5. Set `membase_max_upload_mb` / `membase_max_text_chars` to whatever
   you're actually comfortable with publicly-reachable-even-if-
   authenticated input costing in storage/LLM compute -- the defaults
   (50MB / 50,000 chars) are reasonable starting points, not
   load-tested ceilings.
6. Make sure the machine's firewall doesn't already expose port 5544
   (Postgres) or 11434 (Ollama) directly -- both are already bound in a
   way that shouldn't be reachable remotely (see "What's already safe"
   above), but a host-level firewall rule is free defense-in-depth on
   top of that, not instead of it.

## Known non-blocking gaps (not fixed this pass, not auth-related)

- **No rate limiting beyond the new length caps.** An authenticated-but-
  malicious (or just buggy) client could still call any endpoint in a
  tight loop. Once an auth option from above is in place, this matters
  much less (the blast radius is "someone with your one password"
  rather than "anyone on the internet"); a reverse-proxy-level rate
  limit (Caddy/nginx both support this natively) is the lowest-effort
  place to add it if it turns out to matter in practice.
- **No security response headers** (`X-Content-Type-Options`,
  `X-Frame-Options`, a `Content-Security-Policy`). Lower priority given
  this app has no cross-origin surface and consistent output-escaping
  already (see "What's already safe"), but cheap defense-in-depth a
  reverse proxy can also add without touching app code.
- **No health-check endpoint** (`GET /healthz` or similar) for a reverse
  proxy or process supervisor to poll. Trivial to add if/when one is
  actually wired up to supervise this process.
- **Dependency currency wasn't audited for known CVEs** -- this
  project's Python dependencies come from `nixpkgs` (see CLAUDE.md), so
  staying current means tracking a newer `nixpkgs` revision in
  `flake.lock`, not a separate `pip`-style audit. Not checked how far
  behind the pinned revision currently is.
- **Uploaded attachments can never be viewed/downloaded again** -- a
  product gap noted during this audit, not a security one (if anything,
  having zero file-serving routes is the safer state). Would need a new
  authenticated route if ever wanted.
