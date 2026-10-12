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

## Authentication: implemented

This used to be the blocking issue -- there was no `Authorization`
header check, no session, no password, anywhere in this codebase.
That's now fixed: every route under `/api/captures`, `/api/memory-units`,
and `/api/recall` requires a valid `Authorization: Bearer <token>`
header (see `app/auth.py`'s `require_auth` dependency, wired on at
`app.include_router(..., dependencies=[Depends(require_auth)])` in
`app/main.py`). See docs/DECISIONS.md "Password login with per-device
tokens" for the design and why it was chosen over the plain options
below.

In short: there's one shared password, read from a plaintext file
(`MEMBASE_AUTH_PASSWORD_FILE`, default `.data/auth_password`) fresh on
every login attempt rather than from an env var captured once at
startup -- so rotating it is just editing that file (e.g. over SSH)
with no restart. A missing/empty file fails every login closed. But
`POST /api/auth/login` issues a fresh, independently-revocable bearer
token *per device* rather than handing back that shared secret itself.
That's what makes this ready for a future mobile app or a second
browser/device: each one logs in once with the same password and gets
its own token, manageable (listed, individually revocable) from the
settings menu without having to change the password everywhere else.
Tokens are stored server-side only as a SHA-256 hash; the raw value is
shown to the client exactly once, at login.

`/login` is also rate-limited server-side (`app/rate_limit.py`): a
global cap of 5 requests/second across all callers (a flood guard),
plus a per-source-IP lockout after 10 failed attempts in 15 minutes (a
brute-force guard, cleared immediately by one successful login). See
docs/DECISIONS.md "File-based password, with server-side rate limiting
on /login".

HTML page routes (`/`, `/recall`, `/captures`) are intentionally NOT
gated server-side -- they still return a static shell with no real
data in it. The actual gate is client-side: `app/static/app.js`'s
`initAuth()` shows a full-page login overlay whenever no valid token is
stored, and every data-fetching call goes through the authenticated API
routes above, so an unauthenticated visitor never sees real content,
just an empty page and a password prompt.

This still assumes the app is bound `127.0.0.1`-only with a reverse
proxy doing TLS termination in front (see the checklist below) --
bearer tokens over plain HTTP are exactly as interceptable as a Basic
Auth header would be. **Password-login does not replace TLS; it
replaces having no authentication at all.**

## Auth system ideas (alternatives considered)

The options below were evaluated before picking the token-based login
above. Kept here for reference -- e.g. Option A (Tailscale) is still a
good complementary layer (network-level isolation *underneath*
app-level auth, not instead of it), and Option C's cookie/CSRF
trade-offs are why this project used a header-based bearer token
instead.

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

### What was actually implemented

Neither B nor D exactly: a hybrid of "one password, any browser" (B's
appeal) with D's bearer-token mechanics, extended to issue one token
*per device* instead of one static token for everyone. This was the
right fit once multi-device/mobile access became an explicit
requirement -- Option B's native browser prompt has no concept of
"log this one device out," and Option D as originally scoped assumed a
single static secret. See docs/DECISIONS.md "Password login with
per-device tokens" for the full reasoning.

Still not recommended for this project's stated scope: anything
involving a user table, OAuth/SSO, or multi-factor -- all solve
problems this app doesn't have (there is, by design, exactly one
user, just possibly several of their own devices), at a complexity
cost this project's whole streamlining effort was specifically about
avoiding.

## Deployment checklist

1. Run `scripts/run-prod.sh` (not `run-dev.sh`) for anything other than
   active local development.
2. Put a reverse proxy in front for TLS -- Caddy is the least-effort
   option (automatic HTTPS via Let's Encrypt with a ~5-line Caddyfile);
   nginx + certbot works too if you already run nginx for other things.
3. Put a real password in the file `MEMBASE_AUTH_PASSWORD_FILE` points
   at (`chmod 600`, owned by whichever user runs the app) *before* the
   reverse proxy is reachable from the public internet -- the dev-only
   default content `nix develop`'s shellHook seeds under `.data/` must
   not be what ends up running in production. Changing it later is
   just editing that file (e.g. over SSH) and takes effect on the next
   login attempt, no restart needed. Log in once from each device you
   want to use and keep its token; use the settings menu's session
   list to revoke a lost device later.
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
