// membase front-end core: everything every page needs.
//
// Loaded by base.html before any page script, as a plain classic
// <script> (deliberately NOT type="module" -- modules break opening the
// built site straight off disk via file:// and add CORS sensitivity for
// zero benefit here). Page scripts under static/pages/ read what they
// need off the single `window.membase` global this file defines and then
// register their own `initXPage` onto it; base.html calls that init at
// the very bottom of the body.
//
// Zero dependencies, by repo rule: no npm, no bundler, no CDN script, no
// vendored library, no web font. The whole build is `pip install jinja2`.
(() => {
  "use strict";

  function showStatus(el, message, isError) {
    el.classList.remove("hidden");
    el.classList.toggle("error", !!isError);
    if (isError) {
      el.textContent = message;
    } else {
      el.innerHTML = "";
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      spinner.setAttribute("aria-hidden", "true");
      el.appendChild(spinner);
      el.appendChild(document.createTextNode(message));
    }
  }

  function hide(el) {
    el.classList.add("hidden");
  }

  function show(el) {
    el.classList.remove("hidden");
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    // The textContent/innerHTML round-trip escapes &, <, > but not
    // quote characters -- fine for every current call site (plain
    // text nodes, or a server-generated UUID in a data-* attribute),
    // but the function's name promises general HTML-escaping safety,
    // so cover quotes too rather than leave a latent attribute-
    // injection trap for the next caller that uses this inside a
    // "..."-delimited attribute with real user text. Every page script
    // that builds markup with innerHTML must run user-supplied text
    // through this -- see docs/DEPLOYMENT.md's XSS audit.
    return div.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // Authentication is handled by the nginx reverse proxy in front of
  // the backend (HTTP Basic Auth), not by the app itself -- see
  // docs/DECISIONS.md "Retire app-level auth for nginx Basic Auth".
  // This file's job is just: collect a username/password once, store
  // them in this browser, and attach them as a Basic Auth header on
  // every API request -- there is no server-side session, token, or
  // per-device identity to manage anymore (one shared credential pair,
  // like any other HTTP Basic Auth deployment). The backend's address
  // is fixed at build time (below), not asked for at login -- this
  // frontend is only ever published for this one backend, so prompting
  // for its URL every time was pure friction with nothing to gain.
  // The deployed backend. Baked in at build time rather than asked for at
  // login -- see docs/DECISIONS.md "Hardcode the backend URL instead of
  // asking for it at login".
  const DEFAULT_BACKEND_URL = "https://membase-api.vxliu.com";
  // ...with one escape hatch, for pointing the built site at a local backend:
  //
  //   localStorage.setItem("membase.backendUrl", "http://127.0.0.1:8010")
  //
  // Reason this exists at all: the frontend is a static site served from a
  // different origin than the API, so there is otherwise no way to run the
  // real built pages against a dev backend without editing this file and
  // rebuilding -- which means the thing you test is not the thing you ship.
  // It is read once at load, never written by the app, and absent for every
  // ordinary visitor, so the deployed default is what a real user always
  // gets.
  const BACKEND_URL_KEY = "membase.backendUrl";
  let BACKEND_URL = DEFAULT_BACKEND_URL;
  try {
    const override = window.localStorage.getItem(BACKEND_URL_KEY);
    if (override && /^https?:\/\//.test(override)) {
      BACKEND_URL = override.replace(/\/+$/, "");
      // Loud on purpose: a stale override silently pointing at a dead local
      // server looks exactly like the backend being down.
      console.info(`membase: using backend override ${BACKEND_URL}`);
    }
  } catch {
    // localStorage unavailable (private mode, disabled): keep the default.
  }
  const CREDENTIALS_KEY = "membase.basicAuth";

  function getCredentials() {
    try {
      return localStorage.getItem(CREDENTIALS_KEY);
    } catch {
      return null;
    }
  }

  function setCredentials(username, password) {
    try {
      localStorage.setItem(CREDENTIALS_KEY, btoa(`${username}:${password}`));
    } catch {
      // Not persisted across a reload -- not worth surfacing as an error.
    }
  }

  function clearCredentials() {
    try {
      localStorage.removeItem(CREDENTIALS_KEY);
    } catch {
      // Nothing to do -- if localStorage is unavailable, there was
      // nothing persisted to clear either.
    }
  }

  function showLoginOverlay() {
    const overlay = document.getElementById("login-overlay");
    if (overlay) show(overlay);
  }

  function hideLoginOverlay() {
    const overlay = document.getElementById("login-overlay");
    if (overlay) hide(overlay);
  }

  function initAuth() {
    const overlay = document.getElementById("login-overlay");
    const form = document.getElementById("login-form");
    const usernameInput = document.getElementById("login-username");
    const passwordInput = document.getElementById("login-password");
    const errorEl = document.getElementById("login-error");
    const logoutButton = document.getElementById("logout-button");
    if (!overlay || !form || !usernameInput || !passwordInput || !errorEl || !logoutButton) return;

    // A section that never calls the API must not be put behind a credential
    // wall. The Evaluation page is the case this exists for: it reads a
    // static results file published alongside the site, needs no backend and
    // no credentials, and is meant to be a readable report -- yet a
    // logged-out visitor got a full-screen login form over it.
    //
    // Declared in base.html's section registry rather than sniffed here, so
    // the answer lives next to the rest of the section's definition. The
    // overlay still exists in the DOM and still appears the moment the user
    // navigates to a section that does need the backend.
    const needsBackend = document.body.dataset.requiresBackend !== "false";

    if (getCredentials() || !needsBackend) {
      hideLoginOverlay();
    } else {
      showLoginOverlay();
    }

    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      hide(errorEl);
      if (!usernameInput.value || !passwordInput.value) {
        errorEl.textContent = "Username and password are both required.";
        show(errorEl);
        return;
      }
      setCredentials(usernameInput.value, passwordInput.value);
      passwordInput.value = "";
      // Reload rather than just hiding the overlay -- there's no
      // separate "verify these credentials" call to make (nginx
      // checks them on the very next real API request); a reload
      // re-runs the page's own init, which surfaces a 401 the normal
      // way (fetchWithTimeout's handler below) if they were wrong.
      window.location.reload();
    });

    logoutButton.addEventListener("click", () => {
      // Nothing server-side to revoke -- there's no per-device session
      // anymore, just a shared credential pair nginx checks on every
      // request. "Logging out" is purely this browser forgetting it.
      clearCredentials();
      showLoginOverlay();
    });
  }

  // Present in base.html on every page, so this runs unconditionally --
  // no per-page init call needed.
  function initSettingsMenu() {
    const toggle = document.getElementById("settings-toggle");
    const menu = document.getElementById("settings-menu");
    if (!toggle || !menu) return;

    toggle.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const isHidden = menu.classList.contains("hidden");
      if (isHidden) {
        show(menu);
      } else {
        hide(menu);
      }
      toggle.setAttribute("aria-expanded", String(isHidden));
    });

    document.addEventListener("click", (ev) => {
      if (!menu.contains(ev.target) && ev.target !== toggle) {
        hide(menu);
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  // A hung request (e.g. a stuck DB transaction server-side) must never
  // leave the page silently spinning forever with no feedback -- that's
  // indistinguishable from a frozen UI. POST /api/captures itself makes
  // no LLM call at all any more (see docs/DECISIONS.md "Restore the
  // multi-call ingestion pipeline") and returns almost immediately, but
  // other endpoints (recall, clarify, correct) still make one or more
  // synchronous LLM calls -- this bounds those to a visible error
  // instead of an infinite wait, rather than aborting a slow-but-working
  // request and misreporting it as stuck.
  const REQUEST_TIMEOUT_MS = 150_000;

  async function fetchWithTimeout(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const headers = { ...(options.headers || {}) };
    const credentials = getCredentials();
    // nginx (not the app) checks this -- see docs/DECISIONS.md
    // "Retire app-level auth for nginx Basic Auth". Attached
    // explicitly by this JS rather than relying on the browser's own
    // native Basic Auth prompt/cache, so the login UI stays this
    // app's custom overlay instead of a jarring browser dialog.
    // Authorization and Content-Type are the only two headers the
    // reverse proxy's CORS config allows -- never add a custom one.
    if (credentials) headers.Authorization = `Basic ${credentials}`;
    // Every API path is relative ("/api/...") in the markup/call
    // sites; prefixed here with the fixed backend origin (this
    // frontend is only ever published for this one backend) since
    // they're cross-origin -- GitHub Pages serves this page, not
    // the backend itself.
    const url = path.startsWith("/") ? `${BACKEND_URL}${path}` : path;
    try {
      const resp = await fetch(url, { ...options, headers, signal: controller.signal });
      if (resp.status === 401) {
        // Wrong or missing credentials -- drop them and ask again.
        // Whatever request just 401'd still surfaces its own error to
        // its caller below.
        clearCredentials();
        showLoginOverlay();
      }
      return resp;
    } catch (err) {
      if (err.name === "AbortError") {
        throw new Error("Request timed out. The server may be stuck -- please try again in a moment.");
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  async function postJSON(url, body) {
    const resp = await fetchWithTimeout(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      // A plain-string `detail` (the common FastAPI HTTPException
      // shape) displays as-is; only a non-string shape (e.g. the
      // list-of-dicts validation-error format) gets JSON.stringify'd --
      // stringifying a plain string would wrap it in literal quote
      // characters the user would see on screen.
      const message =
        typeof data.detail === "string"
          ? data.detail
          : data.detail
            ? JSON.stringify(data.detail)
            : `Request failed (${resp.status})`;
      const err = new Error(message);
      // Marks this as "the server responded, just with an error status" --
      // distinct from a timeout or a connection that never got a response
      // at all. See describeError().
      err.apiStatus = resp.status;
      throw err;
    }
    return data;
  }

  function formatElapsed(ms) {
    return (ms / 1000).toFixed(1) + "s";
  }

  // Three failure modes that look identical to a user staring at a spinner
  // but mean very different things: the client gave up waiting (backend
  // is likely still working -- large recall queries can genuinely take a
  // while), the request never reached the server at all (connectivity),
  // or the server answered with an error status (an actual API problem,
  // not a timing issue). Distinguishing them is the point -- see the
  // recall-page transparency work in docs/DECISIONS.md.
  function describeError(err, elapsedMs) {
    const elapsed = formatElapsed(elapsedMs);
    if (err.name === "AbortError" || /timed out/i.test(err.message)) {
      return `Timed out after ${elapsed} -- the backend is likely still processing. Try again shortly.`;
    }
    if (typeof err.apiStatus === "number") {
      return `The server returned an error after ${elapsed} (HTTP ${err.apiStatus}): ${err.message}`;
    }
    if (err instanceof TypeError || /Failed to fetch|NetworkError/i.test(err.message)) {
      return `Couldn't reach the server after ${elapsed} -- check your connection and that the backend is running.`;
    }
    return `${err.message} (after ${elapsed})`;
  }

  // Fetches an attachment's bytes through fetchWithTimeout (the only
  // thing that attaches the Authorization header this backend requires
  // -- a plain <a href> can't carry it) and opens/downloads them client-
  // side. Images open in a new tab; anything else triggers a download
  // via a throwaway <a download>. Shared by the Store page, the Library
  // page, and the Recall page's rendered sources -- the three places a
  // memory's original attachments can surface.
  async function openAttachment(attachment, triggerEl) {
    const originalLabel = triggerEl ? triggerEl.textContent : null;
    if (triggerEl) {
      triggerEl.disabled = true;
      triggerEl.textContent = "Loading...";
    }
    try {
      const resp = await fetchWithTimeout(`/api/attachments/${attachment.id}/file`);
      if (!resp.ok) throw new Error(`Failed to load attachment (${resp.status})`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      if (attachment.content_type.startsWith("image/")) {
        window.open(url, "_blank");
      } else {
        const link = document.createElement("a");
        link.href = url;
        link.download = attachment.original_filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
      }
      // Revoked after a delay rather than immediately -- the new tab/
      // download needs time to actually read the blob first.
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (err) {
      alert(err.message || "Failed to open attachment");
    } finally {
      if (triggerEl) {
        triggerEl.disabled = false;
        triggerEl.textContent = originalLabel;
      }
    }
  }

  // Builds "Attachments: name1, name2" as real DOM nodes (each name a
  // clickable button wired to openAttachment), not a joined string --
  // every attachment, of every supported type, gets a working view/
  // download link wherever this is used.
  function buildAttachmentList(attachments) {
    const wrap = document.createElement("span");
    if (!attachments || !attachments.length) return wrap;
    wrap.appendChild(document.createTextNode("Attachments: "));
    attachments.forEach((a, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "attachment-link";
      button.textContent = a.original_filename;
      button.addEventListener("click", () => openAttachment(a, button));
      wrap.appendChild(button);
      if (i < attachments.length - 1) wrap.appendChild(document.createTextNode(", "));
    });
    return wrap;
  }

  // The one global. Page scripts add their own initXPage() to it; nothing
  // else is ever attached to `window`.
  window.membase = {
    BACKEND_URL,
    showStatus,
    hide,
    show,
    escapeHtml,
    getCredentials,
    setCredentials,
    clearCredentials,
    showLoginOverlay,
    hideLoginOverlay,
    initAuth,
    initSettingsMenu,
    fetchWithTimeout,
    postJSON,
    formatElapsed,
    describeError,
    openAttachment,
    buildAttachmentList,
  };

  // Both of these drive chrome that base.html puts on every page, so they
  // run at load, exactly as they did in the single-file app.js.
  initAuth();
  initSettingsMenu();
})();
