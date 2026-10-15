const membase = (() => {
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
    // "..."-delimited attribute with real user text.
    return div.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // Authentication is handled by the nginx reverse proxy in front of
  // this app (HTTP Basic Auth), not by this app itself -- see
  // docs/DECISIONS.md "Retire app-level auth for nginx Basic Auth".
  // This file's job is just: collect a username/password once, store
  // them in this browser, and attach them as a Basic Auth header on
  // every API request -- there is no server-side session, token, or
  // per-device identity to manage anymore (one shared credential pair,
  // like any other HTTP Basic Auth deployment). The backend's address
  // is fixed at build time (below), not asked for at login -- this
  // frontend is only ever published for this one backend, so prompting
  // for its URL every time was pure friction with nothing to gain.
  const BACKEND_URL = "https://membase-api.vxliu.com";
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

    if (getCredentials()) {
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

  initAuth();
  initSettingsMenu();

  // A hung request (e.g. a stuck DB transaction server-side) must never
  // leave the page silently spinning forever with no feedback -- that's
  // indistinguishable from a frozen UI. Sized to the server's own worst
  // legitimate case, not guessed independently: the synchronous part of
  // saving a capture makes at most one LLM call (analyze_capture -- see
  // docs/DECISIONS.md "Collapse ingestion LLM calls"), up to the server's
  // configured 90s llm_timeout_seconds -- embedding/finalization happens
  // in a background task after this request already returned. A shorter
  // client timeout would abort a slow-but-working request and misreport
  // it as stuck; this still bounds the true-hang case to a visible error
  // instead of an infinite wait.
  const REQUEST_TIMEOUT_MS = 150_000;

  async function fetchWithTimeout(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const headers = { ...(options.headers || {}) };
    const credentials = getCredentials();
    // nginx (not this app) checks this -- see docs/DECISIONS.md
    // "Retire app-level auth for nginx Basic Auth". Attached
    // explicitly by this JS rather than relying on the browser's own
    // native Basic Auth prompt/cache, so the login UI stays this
    // app's custom overlay instead of a jarring browser dialog.
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

  function renderUnits(derivedEl, units) {
    const lines = units.map((u) => ({
      content: u.authoritative_content,
      meaning: u.semantic && u.semantic.meaning,
      aliases: u.semantic && u.semantic.aliases,
    }));
    derivedEl.textContent = JSON.stringify(lines, null, 2);
  }

  // Whether a unit's background processing (the embedding step, and --
  // see docs/DECISIONS.md "membase structurer" -- the deeper structurer
  // pass) has actually finished is deliberately NOT waited for or
  // polled on this page any more: the Store page shows "Saved" as soon
  // as the text itself is durably persisted (authoritative_content and
  // semantic fields are already set synchronously by the time this
  // page's request returns -- only retrieval_text/embedding/structuring
  // are backgrounded). Post-processing status/progress is visualized on
  // the Browse page instead, which is the one place that view belongs.

  function buildCaptureCard() {
    const card = document.createElement("div");
    card.className = "card capture-card";
    card.innerHTML = `
      <p class="capture-saving muted"><span class="spinner" aria-hidden="true"></span>Saving...</p>
      <div class="capture-result hidden">
        <h2 class="capture-result-heading">Saved.</h2>
        <p class="muted">Derived information (automatically extracted; your original text above is never changed by it):</p>
        <pre class="derived"></pre>
        <p class="muted capture-attachment-list"></p>
        <p class="muted capture-pending-note hidden">
          One of these needs a quick follow-up question answered before
          it's searchable -- see the Browse tab.
        </p>
        <p class="muted capture-safe-to-leave">
          Your words are already saved. Everything else -- indexing it
          for search, and the slower background enrichment pass -- keeps
          running on the server, so it's safe to close this page now;
          check the Browse page later if you want to see its progress.
        </p>
      </div>
      <p class="capture-error status error hidden"></p>
    `;
    return card;
  }

  // Drives one capture from submission to the result card, entirely
  // through this card's own elements -- nothing here is shared with any
  // other card, so several captures can be mid-flight at once. No
  // clarification handling here any more -- a unit that needs a
  // follow-up question answered just stays awaiting_clarification;
  // resolving it only ever happens on the Browse tab now (see
  // docs/DECISIONS.md "Clarification moved to Browse"), never blocking
  // this page.
  function runCapture(card, formData) {
    const savingEl = card.querySelector(".capture-saving");
    const resultSection = card.querySelector(".capture-result");
    const resultHeading = card.querySelector(".capture-result-heading");
    const derivedEl = card.querySelector(".derived");
    const errorEl = card.querySelector(".capture-error");
    const attachmentListEl = card.querySelector(".capture-attachment-list");
    const pendingNoteEl = card.querySelector(".capture-pending-note");

    function renderAttachmentList(attachments) {
      attachmentListEl.textContent = attachments.length
        ? `Attachments: ${attachments.map((a) => a.original_filename).join(", ")}`
        : "";
    }

    function showError(message) {
      hide(savingEl);
      showStatus(errorEl, message, true);
    }

    return async function start() {
      try {
        const resp = await fetchWithTimeout("/api/captures", { method: "POST", body: formData });
        const capture = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(capture.detail || "Failed to save");

        hide(savingEl);
        const units = capture.units;
        resultHeading.textContent = units.length > 1 ? `Saved ${units.length} memories.` : "Saved.";
        renderUnits(derivedEl, units);
        renderAttachmentList(capture.attachments);
        if (units.some((u) => u.pending_clarification)) show(pendingNoteEl);
        show(resultSection);
      } catch (err) {
        showError(err.message);
      }
    };
  }

  function initCreatePage() {
    const form = document.getElementById("create-form");
    const capturesEl = document.getElementById("captures");
    const fileInput = document.getElementById("files");
    const selectedFilesEl = document.getElementById("selected-files");

    // Accumulated across possibly several picker interactions -- a bare
    // <input type="file" multiple> only accumulates selections WITHIN
    // one picker dialog; reopening the dialog to add more replaces its
    // .files entirely, which otherwise silently drops whatever was
    // already chosen. This list (and the visible preview below it) is
    // what actually gets submitted; the input's own .files is read once
    // per "change" and then cleared.
    let selectedFiles = [];

    function renderSelectedFiles() {
      selectedFilesEl.innerHTML = "";
      for (const [index, file] of selectedFiles.entries()) {
        const li = document.createElement("li");
        li.className = "selected-file";
        const name = document.createElement("span");
        name.textContent = file.name;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => {
          selectedFiles.splice(index, 1);
          renderSelectedFiles();
        });
        li.appendChild(name);
        li.appendChild(remove);
        selectedFilesEl.appendChild(li);
      }
    }

    fileInput.addEventListener("change", () => {
      selectedFiles.push(...fileInput.files);
      fileInput.value = ""; // so the next picker interaction is purely additive
      renderSelectedFiles();
    });

    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const formData = new FormData();
      formData.set("text", document.getElementById("text").value);
      for (const file of selectedFiles) formData.append("files", file);
      form.reset();
      selectedFiles = [];
      renderSelectedFiles();

      const card = buildCaptureCard();
      capturesEl.prepend(card);
      const start = runCapture(card, formData);
      start();
    });
  }

  function renderSources(listEl, sources) {
    listEl.innerHTML = "";
    if (sources.length === 0) {
      const li = document.createElement("li");
      li.textContent = "(none)";
      listEl.appendChild(li);
      return;
    }
    for (const s of sources) {
      const li = document.createElement("li");
      const summary = s.semantic && s.semantic.meaning ? s.semantic.meaning : s.content;
      const when = new Date(s.created_at).toLocaleDateString();
      li.innerHTML = `<strong>${escapeHtml(summary)}</strong><br>
        <span class="muted">Memory ${escapeHtml(s.id)} &middot; stored ${when}</span><br>
        <span class="caption">&ldquo;${escapeHtml(s.content)}&rdquo;</span>`;
      listEl.appendChild(li);
    }
  }

  // One-shot recall (app/retrieval/agent.py:run_agent): a single query,
  // a single grounded answer or abstention. The multi-round interactive
  // RecallSession (app/retrieval/session.py, POST /api/recall/sessions)
  // remains fully implemented and reachable via the API -- it's just not
  // this page's default, after a benchmark measured it underperforming
  // this simpler path on a deliberately ambiguous corpus; see
  // docs/DECISIONS.md.
  function initRecallPage() {
    const form = document.getElementById("recall-form");
    const statusEl = document.getElementById("status");
    const resultCard = document.getElementById("result");
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");
    const sourcesWrap = document.getElementById("sources-wrap");

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      hide(resultCard);
      const query = document.getElementById("query").value;

      const startedAt = performance.now();
      showStatus(statusEl, "Remembering...");
      const ticker = setInterval(() => {
        showStatus(statusEl, `Remembering... (${formatElapsed(performance.now() - startedAt)} elapsed)`);
      }, 200);

      try {
        const data = await postJSON("/api/recall", { query });
        clearInterval(ticker);
        hide(statusEl);
        answerEl.textContent = data.answer;
        answerEl.classList.toggle("abstained", data.abstained);
        if (data.abstained) {
          hide(sourcesWrap);
        } else {
          show(sourcesWrap);
          renderSources(sourcesEl, data.sources);
        }
        show(resultCard);
      } catch (err) {
        clearInterval(ticker);
        showStatus(statusEl, describeError(err, performance.now() - startedAt), true);
      }
    });
  }

  function renderCaptureList(listEl, captures, onClarificationResolved) {
    listEl.innerHTML = "";
    if (captures.length === 0) {
      listEl.innerHTML = '<p class="muted">Nothing saved yet.</p>';
      return;
    }
    for (const capture of captures) {
      const card = document.createElement("div");
      card.className = "card capture-row";
      const when = new Date(capture.created_at).toLocaleString();
      const unitBadges = capture.units
        .map((u) => {
          // data-unit-id so the clarification handler below can update
          // this specific badge in place once a question is answered,
          // without a full list reload.
          const statusBadge = `<span class="status-badge status-badge-${escapeHtml(u.status)}" data-unit-id="${escapeHtml(u.id)}">${escapeHtml(u.status)}</span>`;
          // The structurer's own progress (see docs/DECISIONS.md "The
          // 'membase structurer'") -- deliberately only shown here, not
          // on the Store page, and only read at page-load time (no
          // polling), same as the status badge beside it.
          const structureBadge = `<span class="status-badge structure-badge-${escapeHtml(u.structure_status)}">structurer: ${escapeHtml(u.structure_status)}</span>`;
          return `${statusBadge} ${structureBadge}`;
        })
        .join(" ");
      const attachmentList = capture.attachments.length
        ? `<p class="muted">Attachments: ${capture.attachments.map((a) => escapeHtml(a.original_filename)).join(", ")}</p>`
        : "";
      const failedStructureUnits = capture.units.filter((u) => u.structure_status === "failed");
      const restructureButtons = failedStructureUnits
        .map((u) => `<button type="button" class="capture-restructure" data-unit-id="${escapeHtml(u.id)}">Retry structuring</button>`)
        .join(" ");
      // A unit awaiting clarification is durably saved but not yet
      // searchable (see docs/ARCHITECTURE.md "Status gate") -- it never
      // gets structured either, since structuring only runs on units
      // that have already reached "ready". Answering here is the same
      // POST /api/memory-units/{id}/clarify the Store page's own
      // in-flight card uses, just reachable later for anything left
      // unanswered from a previous visit.
      const pendingUnits = capture.units.filter((u) => u.pending_clarification);
      const clarificationBlocks = pendingUnits
        .map(
          (u) => `
        <div class="capture-clarification" data-unit-id="${escapeHtml(u.id)}">
          <p class="muted">This memory isn't searchable yet until this is answered.</p>
          <p class="capture-clarification-question">${escapeHtml(u.pending_clarification.question)}</p>
          <form class="capture-clarification-form">
            <input class="capture-clarification-answer" type="text" placeholder="Your answer" />
            <button type="submit" class="capture-clarification-submit">Submit answer</button>
            <button type="button" class="capture-clarification-discard">Discard this one</button>
          </form>
          <p class="capture-clarification-error status error hidden"></p>
        </div>`
        )
        .join("");
      card.innerHTML = `
        <p class="muted">${escapeHtml(when)}</p>
        <p>${escapeHtml(capture.raw_text)}</p>
        <p>${unitBadges}</p>
        ${clarificationBlocks}
        ${attachmentList}
        ${restructureButtons}
        <p class="capture-restructure-error status error hidden"></p>
        <button type="button" class="capture-delete">Delete</button>
      `;

      for (const block of card.querySelectorAll(".capture-clarification")) {
        const unitId = block.dataset.unitId;
        const questionEl = block.querySelector(".capture-clarification-question");
        const form = block.querySelector(".capture-clarification-form");
        const answerInput = block.querySelector(".capture-clarification-answer");
        const submitButton = block.querySelector(".capture-clarification-submit");
        const discardButton = block.querySelector(".capture-clarification-discard");
        const clarificationErrorEl = block.querySelector(".capture-clarification-error");

        function setBusy(busy, label) {
          submitButton.disabled = busy;
          discardButton.disabled = busy;
          answerInput.disabled = busy;
          submitButton.textContent = busy ? label : "Submit answer";
          if (busy) hide(clarificationErrorEl);
        }

        // A unit can ask another round of questions after this one is
        // answered (see docs/ARCHITECTURE.md) -- if so, this same block
        // just shows the next question instead of disappearing; it only
        // goes away once nothing is left pending for this unit.
        function applyUpdatedUnit(updatedUnit) {
          const unit = capture.units.find((u) => u.id === updatedUnit.id);
          if (unit) {
            unit.status = updatedUnit.status;
            unit.pending_clarification = updatedUnit.pending_clarification;
            unit.clarifications = updatedUnit.clarifications;
          }
          const badge = card.querySelector(`.status-badge[data-unit-id="${CSS.escape(updatedUnit.id)}"]`);
          if (badge) {
            badge.textContent = updatedUnit.status;
            badge.className = `status-badge status-badge-${updatedUnit.status}`;
          }
          if (updatedUnit.pending_clarification) {
            questionEl.textContent = updatedUnit.pending_clarification.question;
            answerInput.value = "";
            setBusy(false);
          } else {
            block.remove();
            if (onClarificationResolved) onClarificationResolved();
          }
        }

        form.addEventListener("submit", async (ev) => {
          ev.preventDefault();
          setBusy(true, "Submitting...");
          try {
            const updatedUnit = await postJSON(`/api/memory-units/${unitId}/clarify`, { answer: answerInput.value });
            applyUpdatedUnit(updatedUnit);
          } catch (err) {
            setBusy(false);
            showStatus(clarificationErrorEl, err.message, true);
          }
        });

        discardButton.addEventListener("click", async () => {
          setBusy(true, "Discarding...");
          try {
            const resp = await fetchWithTimeout(`/api/memory-units/${unitId}/discard`, { method: "POST" });
            const updatedUnit = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(updatedUnit.detail || "Failed to discard");
            applyUpdatedUnit(updatedUnit);
          } catch (err) {
            setBusy(false);
            showStatus(clarificationErrorEl, err.message, true);
          }
        });
      }

      const restructureError = card.querySelector(".capture-restructure-error");
      for (const button of card.querySelectorAll(".capture-restructure")) {
        button.addEventListener("click", async () => {
          button.disabled = true;
          hide(restructureError);
          showStatus(restructureError, "Retrying...", false);
          try {
            const resp = await fetchWithTimeout(`/api/memory-units/${button.dataset.unitId}/restructure`, { method: "POST" });
            const updated = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(updated.detail || `Failed to retry (${resp.status})`);
            const unit = capture.units.find((u) => u.id === updated.id);
            if (unit) unit.structure_status = updated.structure_status;
            hide(restructureError);
            if (updated.structure_status !== "failed") button.remove();
            else button.disabled = false;
          } catch (err) {
            showStatus(restructureError, err.message, true);
            button.disabled = false;
          }
        });
      }

      card.querySelector(".capture-delete").addEventListener("click", async () => {
        if (!confirm("Delete this memory for good? This can't be undone.")) return;
        try {
          const resp = await fetchWithTimeout(`/api/captures/${capture.id}`, { method: "DELETE" });
          if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || `Failed to delete (${resp.status})`);
          }
          card.remove();
        } catch (err) {
          alert(err.message);
        }
      });
      listEl.appendChild(card);
    }
  }

  function initCapturesPage() {
    const statusEl = document.getElementById("status");
    const listEl = document.getElementById("capture-list");
    const skipAllButton = document.getElementById("skip-all-followups");

    function updateSkipAllVisibility() {
      if (!skipAllButton) return;
      if (listEl.querySelector(".capture-clarification")) show(skipAllButton);
      else hide(skipAllButton);
    }

    if (skipAllButton) {
      skipAllButton.addEventListener("click", async () => {
        const blocks = [...listEl.querySelectorAll(".capture-clarification")];
        if (blocks.length === 0) return;
        if (!confirm(`Discard all ${blocks.length} memory unit(s) still awaiting a follow-up question? This can't be undone.`)) return;
        skipAllButton.disabled = true;
        // Fire sequentially, not in parallel -- these hit the same
        // rate-limited backend as everything else, and there's no need
        // to race several discard calls against each other.
        for (const block of blocks) {
          const unitId = block.dataset.unitId;
          try {
            const resp = await fetchWithTimeout(`/api/memory-units/${unitId}/discard`, { method: "POST" });
            const updatedUnit = await resp.json().catch(() => ({}));
            if (resp.ok) {
              const badge = listEl.querySelector(`.status-badge[data-unit-id="${CSS.escape(unitId)}"]`);
              if (badge) {
                badge.textContent = updatedUnit.status;
                badge.className = `status-badge status-badge-${updatedUnit.status}`;
              }
              block.remove();
            }
            // A single failure (network hiccup, already-resolved unit)
            // just leaves that one block in place, still answerable or
            // discardable individually -- doesn't abort the rest of the
            // batch.
          } catch {
            // Same as above: move on to the next one.
          }
        }
        skipAllButton.disabled = false;
        updateSkipAllVisibility();
      });
    }

    (async () => {
      const startedAt = performance.now();
      showStatus(statusEl, "Loading...");
      try {
        const resp = await fetchWithTimeout("/api/captures");
        const captures = await resp.json().catch(() => ([]));
        if (!resp.ok) throw new Error(captures.detail || "Failed to load captures");
        hide(statusEl);
        // Captures with a unit still awaiting a follow-up question sort
        // first -- a stable sort, so within each group (pending / not)
        // the backend's own newest-first order is preserved.
        const sorted = [...captures].sort((a, b) => {
          const aPending = a.units.some((u) => u.pending_clarification) ? 1 : 0;
          const bPending = b.units.some((u) => u.pending_clarification) ? 1 : 0;
          return bPending - aPending;
        });
        renderCaptureList(listEl, sorted, updateSkipAllVisibility);
        updateSkipAllVisibility();
      } catch (err) {
        showStatus(statusEl, describeError(err, performance.now() - startedAt), true);
      }
    })();
  }

  return { initCreatePage, initRecallPage, initCapturesPage };
})();
