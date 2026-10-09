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
    return div.innerHTML;
  }

  // Whether follow-up (clarification) questions must be answered is a
  // client-side-only preference -- the API always accepts a blank answer
  // as "skip this question" (see docs/DECISIONS.md), there is no
  // server-side enforcement to keep in sync with. Defaults to true so
  // existing behavior (answer required) is unchanged for anyone who
  // never opens the settings menu.
  const OBLIGATORY_SETTING_KEY = "membase.obligatoryQuestions";

  // localStorage can throw (not just be unavailable) in some browser
  // privacy modes/configurations -- this is called unconditionally at
  // page-load time via initSettingsMenu(), so an uncaught exception here
  // would break this entire script's top-level execution, silently
  // disabling every button on the page, not just the settings menu.
  function isObligatory() {
    try {
      return localStorage.getItem(OBLIGATORY_SETTING_KEY) !== "0";
    } catch {
      return true;
    }
  }

  function setObligatory(value) {
    try {
      localStorage.setItem(OBLIGATORY_SETTING_KEY, value ? "1" : "0");
    } catch {
      // Preference just won't persist across reloads -- not worth
      // surfacing an error for.
    }
  }

  // Unlike "obligatory" above, this one IS enforced server-side (see
  // docs/DECISIONS.md "Disable follow-up questions") -- when on, the
  // `disable_followup_questions` form field is sent with every
  // POST /api/captures, and the backend never surfaces a clarification
  // question for that capture at all, regardless of what the model (or
  // an explicit-metadata conflict) would otherwise have asked. Defaults
  // to false so existing behavior is unchanged for anyone who never
  // opens the settings menu.
  const DISABLE_FOLLOWUP_SETTING_KEY = "membase.disableFollowupQuestions";

  function isFollowupDisabled() {
    try {
      return localStorage.getItem(DISABLE_FOLLOWUP_SETTING_KEY) === "1";
    } catch {
      return false;
    }
  }

  function setFollowupDisabled(value) {
    try {
      localStorage.setItem(DISABLE_FOLLOWUP_SETTING_KEY, value ? "1" : "0");
    } catch {
      // Preference just won't persist across reloads -- not worth
      // surfacing an error for.
    }
  }

  // Present in base.html on every page, so this runs unconditionally --
  // no per-page init call needed.
  function initSettingsMenu() {
    const toggle = document.getElementById("settings-toggle");
    const menu = document.getElementById("settings-menu");
    const obligatoryCheckbox = document.getElementById("setting-obligatory-questions");
    const disableFollowupCheckbox = document.getElementById("setting-disable-followup-questions");
    if (!toggle || !menu || !obligatoryCheckbox || !disableFollowupCheckbox) return;

    obligatoryCheckbox.checked = isObligatory();
    disableFollowupCheckbox.checked = isFollowupDisabled();
    // "Must you answer a question that will never be asked" doesn't
    // make sense -- grey the obligatory setting out while follow-up
    // questions are disabled entirely, rather than leave two settings
    // visibly contradicting each other.
    obligatoryCheckbox.disabled = disableFollowupCheckbox.checked;

    toggle.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const isHidden = menu.classList.contains("hidden");
      if (isHidden) show(menu);
      else hide(menu);
      toggle.setAttribute("aria-expanded", String(isHidden));
    });

    document.addEventListener("click", (ev) => {
      if (!menu.contains(ev.target) && ev.target !== toggle) {
        hide(menu);
        toggle.setAttribute("aria-expanded", "false");
      }
    });

    obligatoryCheckbox.addEventListener("change", () => setObligatory(obligatoryCheckbox.checked));
    disableFollowupCheckbox.addEventListener("change", () => {
      setFollowupDisabled(disableFollowupCheckbox.checked);
      obligatoryCheckbox.disabled = disableFollowupCheckbox.checked;
    });
  }

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

  async function fetchWithTimeout(url, options) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
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
      const err = new Error(data.detail ? JSON.stringify(data.detail) : `Request failed (${resp.status})`);
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
      <div class="capture-clarification hidden">
        <h2>One question</h2>
        <p class="muted">This memory isn't saved as searchable yet until this is answered.</p>
        <p class="capture-clarification-question"></p>
        <form class="capture-clarification-form">
          <input class="capture-clarification-answer" type="text" placeholder="Your answer" />
          <button type="submit" class="capture-clarification-submit">Submit answer</button>
          <button type="button" class="capture-clarification-discard">Discard this one</button>
        </form>
        <p class="capture-clarification-stage muted hidden"></p>
        <p class="capture-clarification-error status error hidden"></p>
      </div>
      <div class="capture-result hidden">
        <h2 class="capture-result-heading">Saved.</h2>
        <p class="muted">Derived information (automatically extracted; your original text above is never changed by it):</p>
        <pre class="derived"></pre>
        <p class="muted capture-attachment-list"></p>
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

  // Drives one capture, from submission through any clarification
  // rounds to background processing, entirely through this card's own
  // elements and local state -- nothing here is shared with any other
  // card, so several captures can be mid-flight at once.
  function runCapture(card, formData) {
    const savingEl = card.querySelector(".capture-saving");
    const clarificationSection = card.querySelector(".capture-clarification");
    const clarificationQuestion = card.querySelector(".capture-clarification-question");
    const clarificationForm = card.querySelector(".capture-clarification-form");
    const clarificationAnswer = card.querySelector(".capture-clarification-answer");
    const clarificationSubmit = card.querySelector(".capture-clarification-submit");
    const clarificationDiscard = card.querySelector(".capture-clarification-discard");
    const clarificationStage = card.querySelector(".capture-clarification-stage");
    const clarificationError = card.querySelector(".capture-clarification-error");
    const resultSection = card.querySelector(".capture-result");
    const resultHeading = card.querySelector(".capture-result-heading");
    const derivedEl = card.querySelector(".derived");
    const errorEl = card.querySelector(".capture-error");
    const attachmentListEl = card.querySelector(".capture-attachment-list");

    let allUnits = [];
    // Capture may split into several units; each may independently have
    // a pending clarification. Ask them one at a time rather than all at
    // once (spec: don't turn this into an interrogation).
    let pendingQueue = [];
    let currentAttachments = [];

    // Read-only: just what's already attached to this memory. Attaching
    // files happens before saving (see initCreatePage()'s selected-files
    // list) -- this card never offers to add more after the fact (see
    // docs/DECISIONS.md "Multi-file posthoc attachment").
    function renderAttachmentList() {
      attachmentListEl.textContent = currentAttachments.length
        ? `Attachments: ${currentAttachments.map((a) => a.original_filename).join(", ")}`
        : "";
    }

    function showError(message) {
      hide(savingEl);
      hide(clarificationSection);
      showStatus(errorEl, message, true);
    }

    // Shown as soon as every pending clarification question (if any) is
    // settled -- does not wait for background embedding/structuring to
    // finish, since authoritative content and the derived fields shown
    // below are already durably saved and known synchronously by this
    // point. Whether background processing later succeeds, fails, or is
    // still running is a Browse-page concern now, not this page's (see
    // docs/DECISIONS.md "membase structurer").
    function showResult() {
      const saved = allUnits.filter((u) => u.status !== "discarded");
      resultHeading.textContent =
        saved.length > 1 ? `Saved ${saved.length} memories.` : saved.length === 1 ? "Saved." : "Nothing saved.";
      renderUnits(derivedEl, saved);
      renderAttachmentList();
      show(resultSection);
    }

    function askNextClarification() {
      setClarificationBusy(false);
      if (pendingQueue.length === 0) {
        showResult();
        return;
      }
      clarificationQuestion.textContent = pendingQueue[0].question;
      // Soft guard only: the API always accepts a blank answer as "skip"
      // (see docs/DECISIONS.md) -- this toggles the HTML `required`
      // attribute and the discard button, nothing server-enforced.
      // Re-read on every question, not just once, so a mid-flight
      // settings change takes effect on the very next question.
      const obligatory = isObligatory();
      clarificationAnswer.required = obligatory;
      // When questions are obligatory, "Discard this one" is the only
      // escape hatch from answering; when they're optional, skipping
      // (submitting blank) already covers that, so the button is
      // redundant and hidden to keep the choice simple.
      if (obligatory) hide(clarificationDiscard);
      else show(clarificationDiscard);
      show(clarificationSection);
    }

    function enqueueIfPending(unit) {
      if (unit.pending_clarification) {
        pendingQueue.push({ unitId: unit.id, question: unit.pending_clarification.question });
      }
    }

    // Disables both buttons and shows a progress label while a clarify/
    // discard request is in flight, so a click gets immediate feedback
    // instead of looking unresponsive during real network/LLM latency --
    // and can't be double-submitted while the first click is still
    // pending. Restored on error so a transient failure (e.g. a dropped
    // connection) can just be retried, not just reported as a dead end.
    function setClarificationBusy(busy, label) {
      clarificationSubmit.disabled = busy;
      clarificationDiscard.disabled = busy;
      clarificationAnswer.disabled = busy;
      if (busy) {
        clarificationSubmit.textContent = label;
        hide(clarificationError);
        showStatus(clarificationStage, label, false);
      } else {
        clarificationSubmit.textContent = "Submit answer";
        hide(clarificationStage);
      }
    }

    function showClarificationError(message) {
      showStatus(clarificationError, message, true);
    }

    clarificationForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const current = pendingQueue[0];
      setClarificationBusy(true, "Submitting answer...");
      try {
        const updatedUnit = await postJSON(`/api/memory-units/${current.unitId}/clarify`, {
          answer: clarificationAnswer.value,
        });
        clarificationAnswer.value = "";
        allUnits = allUnits.map((u) => (u.id === updatedUnit.id ? updatedUnit : u));
        pendingQueue = pendingQueue.slice(1);
        // Answering can surface a further question -- another round, or
        // the next one already queued server-side for this unit.
        enqueueIfPending(updatedUnit);
        askNextClarification();
      } catch (err) {
        setClarificationBusy(false);
        showClarificationError(err.message);
      }
    });

    clarificationDiscard.addEventListener("click", async () => {
      const current = pendingQueue[0];
      setClarificationBusy(true, "Discarding...");
      try {
        const resp = await fetchWithTimeout(`/api/memory-units/${current.unitId}/discard`, { method: "POST" });
        const updatedUnit = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(updatedUnit.detail || "Failed to discard");
        allUnits = allUnits.map((u) => (u.id === updatedUnit.id ? updatedUnit : u));
        pendingQueue = pendingQueue.slice(1);
        askNextClarification();
      } catch (err) {
        setClarificationBusy(false);
        showClarificationError(err.message);
      }
    });

    return async function start() {
      try {
        const resp = await fetchWithTimeout("/api/captures", { method: "POST", body: formData });
        const capture = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(capture.detail || "Failed to save");

        hide(savingEl);
        currentAttachments = capture.attachments;
        allUnits = capture.units;
        pendingQueue = [];
        for (const unit of capture.units) enqueueIfPending(unit);
        askNextClarification();
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
      formData.set("disable_followup_questions", isFollowupDisabled() ? "true" : "false");
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

  function renderCaptureList(listEl, captures) {
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
          const statusBadge = `<span class="status-badge status-badge-${escapeHtml(u.status)}">${escapeHtml(u.status)}</span>`;
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
      card.innerHTML = `
        <p class="muted">${escapeHtml(when)}</p>
        <p>${escapeHtml(capture.raw_text)}</p>
        <p>${unitBadges}</p>
        ${attachmentList}
        ${restructureButtons}
        <p class="capture-restructure-error status error hidden"></p>
        <button type="button" class="capture-delete">Delete</button>
      `;

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

    (async () => {
      const startedAt = performance.now();
      showStatus(statusEl, "Loading...");
      try {
        const resp = await fetchWithTimeout("/api/captures");
        const captures = await resp.json().catch(() => ([]));
        if (!resp.ok) throw new Error(captures.detail || "Failed to load captures");
        hide(statusEl);
        renderCaptureList(listEl, captures);
      } catch (err) {
        showStatus(statusEl, describeError(err, performance.now() - startedAt), true);
      }
    })();
  }

  return { initCreatePage, initRecallPage, initCapturesPage };
})();
