# Deployment readiness audit

Scope: can this app be safely run somewhere other than a developer's
own `nix develop` shell on localhost? Specifically the scenario the
audit was commissioned for: **single user by design, but the HTTP
endpoint itself reachable from the public internet.** The deployment
shape itself is now split: backend local (behind nginx + Cloudflare
Tunnel, see "nginx + Cloudflare Tunnel" below), frontend published
separately to GitHub Pages.

Short answer: **everything needed is in place.** Authentication is
HTTP Basic Auth enforced by nginx in front of the backend -- see
"Authentication: nginx Basic Auth" below; this app has no auth code of
its own anymore. Don't skip the nginx setup; without it, nothing in
this app gates access to your stored memories.

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

## Authentication: nginx Basic Auth (app-level auth retired)

This app used to implement its own password-login + per-device
bearer-token system. That was **retired** in favor of HTTP Basic Auth
enforced by the nginx reverse proxy in front of the backend -- see
docs/DECISIONS.md "Retire app-level auth for nginx Basic Auth" for the
full reasoning and the trade-off accepted (one shared credential pair,
no per-device revocation anymore). The app itself no longer has any
`/api/auth/*` routes, any session/token table, or any rate limiting of
its own on login attempts -- nginx owns all of that now (see "nginx +
Cloudflare Tunnel" below).

The frontend's login overlay (`app/static/app.js`'s `initAuth()`)
still exists, but its job changed: it collects a backend URL plus a
username/password once, stores them in this browser's `localStorage`,
and attaches `Authorization: Basic <base64(user:pass)>` to every API
request itself. There's no `/login` endpoint to call first -- the
credentials are only ever actually checked by nginx, on the very next
real API request. A 401 clears the stored credentials and re-shows the
overlay.

## nginx + Cloudflare Tunnel

The backend stays local (bound `127.0.0.1` only, as always), reached
publicly through: **Cloudflare edge (TLS to the browser) → encrypted
tunnel → `cloudflared` on this machine → nginx (`127.0.0.1:8080`) →
FastAPI backend (`127.0.0.1:8000`)**. Cloudflare Tunnel specifically
(rather than a conventional reverse-proxy-with-port-forwarding setup)
because this machine is behind CGNAT -- there is no public IP under
local control to forward a port on at all. See docs/DECISIONS.md
"Cloudflare Tunnel for CGNAT-safe public reachability" for why
Tailscale Funnel doesn't substitute here (it doesn't support an
arbitrary custom domain).

**Setup:**

1. **nginx**: copy `deploy/nginx-membase.conf` in (as a traditional
   `/etc/nginx/sites-enabled/` file, or adapted into a NixOS
   `services.nginx.virtualHosts` block's `extraConfig` -- the file
   itself is portable either way). It listens on `127.0.0.1:8080` only
   -- TLS is already terminated by Cloudflare's edge before
   `cloudflared` ever forwards a request here, so nginx never needs
   its own certificate.
2. **Basic Auth credentials**: generate the htpasswd file nginx's
   config points at:
   ```
   printf 'yourusername:%s\n' "$(openssl passwd -apr1)" | sudo tee /etc/nginx/membase.htpasswd
   ```
   (prompts for the password interactively; `openssl passwd -apr1` is
   used instead of requiring the separate `apache2-utils`/`httpd`
   package just for its `htpasswd` binary).
3. **Cloudflare Tunnel**: requires `membase.vxliu.com`'s DNS to be
   hosted on Cloudflare first (point the domain's nameservers at
   Cloudflare, or add it as a site there). Then:
   ```
   cloudflared tunnel login
   cloudflared tunnel create membase
   cloudflared tunnel route dns membase membase.vxliu.com
   ```
   Copy `deploy/cloudflared-config.yml` to wherever `cloudflared`
   expects it, filling in the tunnel ID and credentials-file path
   `tunnel create` printed. Run it (`cloudflared tunnel run membase`,
   or via the NixOS `services.cloudflared` module / a systemd unit for
   anything longer-running than a manual test).
4. Confirm end to end: `curl -u yourusername https://membase.vxliu.com/api/captures`
   should prompt for the password and then return `[]` (or your real
   captures) -- not a connection error, not an unauthenticated 200.

This fully replaces the old "put *a* TLS-terminating reverse proxy in
front" advice from earlier revisions of this doc -- nginx here never
holds a TLS certificate at all; Cloudflare's edge is what the browser
actually negotiates TLS with.

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

None of the above, in the end -- see docs/DECISIONS.md "Retire
app-level auth for nginx Basic Auth". Option B's exact mechanics
(HTTP Basic Auth, one shared credential pair) ended up being what's
actually used, just enforced by nginx (see "nginx + Cloudflare
Tunnel" above) rather than written into this app. The per-device
token system described in earlier revisions of this doc was built,
then deliberately retired once a reverse proxy was being introduced
for other reasons anyway (TLS, headers, rate limiting) -- consolidating
"is this request allowed at all" into one place was judged simpler
than splitting it across two layers, at the cost of losing per-device
revocation (now back to Option B's own trade-off: one shared password,
no concept of "log this one device out" short of changing it for
everyone).

Still not recommended for this project's stated scope: anything
involving a user table, OAuth/SSO, or multi-factor -- all solve
problems this app doesn't have (there is, by design, exactly one
user), at a complexity cost this project's whole streamlining effort
was specifically about avoiding.

## Deployment checklist

1. Run `scripts/run-prod.sh` (not `run-dev.sh`) for anything other than
   active local development.
2. Set up nginx + Cloudflare Tunnel per "nginx + Cloudflare Tunnel"
   above -- generate the htpasswd file with a real password before
   the tunnel is live, not the placeholder you tested with.
3. Set `FRONTEND_ORIGIN` to the actual published GitHub Pages origin
   (e.g. `https://luxkaiwalker.github.io`) in the backend's production
   environment -- the dev default (`http://127.0.0.1:8000`) won't
   match once the frontend is served from a different origin, and a
   mismatch here means every cross-origin API call gets blocked by the
   browser's own CORS enforcement (the request never even reaches this
   app's logic -- check the browser console, not server logs, if this
   is wrong).
4. Confirm `app/main.py`'s `uvicorn.run(..., host="127.0.0.1", ...)`
   fallback and `scripts/run-prod.sh` both still say `127.0.0.1` --
   this should never change to `0.0.0.0`; nginx (itself only listening
   on `127.0.0.1:8080`, reached only via `cloudflared`'s local
   connection) is the only thing in this chain with any path from the
   public internet, and even it has no direct public-facing socket.
5. Set `membase_max_upload_mb` / `membase_max_text_chars` to whatever
   you're actually comfortable with publicly-reachable-even-if-
   authenticated input costing in storage/LLM compute -- the defaults
   (50MB / 50,000 chars) are reasonable starting points, not
   load-tested ceilings. `deploy/nginx-membase.conf`'s
   `client_max_body_size` should stay in sync with
   `membase_max_upload_mb` if you change the latter.
6. Make sure the machine's firewall doesn't already expose port 5544
   (Postgres), 11434 (Ollama), or 8000/8080 (the app/nginx) directly --
   none of them need to be, or should be, reachable from outside this
   machine; `cloudflared` is the only process that talks to the outside
   world, and it does so via an outbound connection, not a listening
   socket.

## Known non-blocking gaps (not fixed this pass, not auth-related)

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
