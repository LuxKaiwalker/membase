const membase = (() => {
  function showStatus(el, message, isError) {
    el.textContent = message;
    el.classList.remove("hidden");
    el.classList.toggle("error", !!isError);
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

  function isObligatory() {
    return localStorage.getItem(OBLIGATORY_SETTING_KEY) !== "0";
  }

  function setObligatory(value) {
    localStorage.setItem(OBLIGATORY_SETTING_KEY, value ? "1" : "0");
  }

  // Present in base.html on every page, so this runs unconditionally --
  // no per-page init call needed.
  function initSettingsMenu() {
    const toggle = document.getElementById("settings-toggle");
    const menu = document.getElementById("settings-menu");
    const checkbox = document.getElementById("setting-obligatory-questions");
    if (!toggle || !menu || !checkbox) return;

    checkbox.checked = isObligatory();

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

    checkbox.addEventListener("change", () => setObligatory(checkbox.checked));
  }

  initSettingsMenu();

  // A hung request (e.g. a stuck DB transaction server-side) must never
  // leave the page silently spinning forever with no feedback -- that's
  // indistinguishable from a frozen UI. 420s is sized to the server's own
  // worst legitimate case, not guessed independently: ingestion can run
  // up to 5 sequential LLM calls (generate_questions, enrich()'s three
  // narrative/entities/retrieval-aids sub-calls, embed), each up to the
  // server's configured 90s llm_timeout_seconds -- a shorter client
  // timeout would abort a slow-but-working request and misreport it as
  // stuck. Still bounds the true-hang case to a visible error instead of
  // an infinite wait.
  const REQUEST_TIMEOUT_MS = 420_000;

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

  // How often to re-check a unit stuck in "processing" (full enrichment
  // now runs as a background task -- see docs/DECISIONS.md -- so the
  // client has to poll for it instead of getting it inline in the save
  // response).
  const POLL_INTERVAL_MS = 1500;

  async function pollUnitUntilDone(unitId) {
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      const resp = await fetchWithTimeout(`/api/memory-units/${unitId}`);
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || "Couldn't check memory status");
      if (data.status !== "processing") return data;
    }
  }

  function buildCaptureCard() {
    const card = document.createElement("div");
    card.className = "card capture-card";
    card.innerHTML = `
      <p class="capture-saving muted">Saving...</p>
      <div class="capture-clarification hidden">
        <h2>One question</h2>
        <p class="muted">This memory isn't saved as searchable yet until this is answered.</p>
        <p class="capture-clarification-question"></p>
        <form class="capture-clarification-form">
          <input class="capture-clarification-answer" type="text" placeholder="Your answer" />
          <button type="submit">Submit answer</button>
          <button type="button" class="capture-clarification-discard">Discard this one</button>
        </form>
      </div>
      <p class="capture-processing muted hidden">The memory is being processed...</p>
      <div class="capture-result hidden">
        <h2 class="capture-result-heading">Saved.</h2>
        <p class="muted">Derived information (automatically extracted; your original text above is never changed by it):</p>
        <pre class="derived"></pre>
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
    const clarificationDiscard = card.querySelector(".capture-clarification-discard");
    const processingEl = card.querySelector(".capture-processing");
    const resultSection = card.querySelector(".capture-result");
    const resultHeading = card.querySelector(".capture-result-heading");
    const derivedEl = card.querySelector(".derived");
    const errorEl = card.querySelector(".capture-error");

    let allUnits = [];
    // Capture may split into several units; each may independently have
    // a pending clarification. Ask them one at a time rather than all at
    // once (spec: don't turn this into an interrogation).
    let pendingQueue = [];

    function showError(message) {
      hide(savingEl);
      hide(clarificationSection);
      hide(processingEl);
      showStatus(errorEl, message, true);
    }

    function showResult() {
      hide(processingEl);
      const ready = allUnits.filter((u) => u.status === "ready");
      const failed = allUnits.filter((u) => u.status === "failed");
      resultHeading.textContent =
        ready.length > 1 ? `Saved ${ready.length} memories.` : ready.length === 1 ? "Saved." : "Nothing saved.";
      renderUnits(derivedEl, ready);
      show(resultSection);
      if (failed.length > 0) {
        showStatus(errorEl, `${failed.length} memory unit(s) failed to process and were not saved.`, true);
      }
    }

    async function waitForProcessing() {
      hide(clarificationSection);
      show(processingEl);
      const inFlight = allUnits.filter((u) => u.status === "processing");
      try {
        const finished = await Promise.all(inFlight.map((u) => pollUnitUntilDone(u.id)));
        for (const unit of finished) {
          allUnits = allUnits.map((u) => (u.id === unit.id ? unit : u));
        }
      } catch (err) {
        showError(err.message);
        return;
      }
      showResult();
    }

    function askNextClarification() {
      if (pendingQueue.length === 0) {
        waitForProcessing();
        return;
      }
      hide(processingEl);
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

    clarificationForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const current = pendingQueue[0];
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
        showError(err.message);
      }
    });

    clarificationDiscard.addEventListener("click", async () => {
      const current = pendingQueue[0];
      try {
        const resp = await fetchWithTimeout(`/api/memory-units/${current.unitId}/discard`, { method: "POST" });
        const updatedUnit = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(updatedUnit.detail || "Failed to discard");
        allUnits = allUnits.map((u) => (u.id === updatedUnit.id ? updatedUnit : u));
        pendingQueue = pendingQueue.slice(1);
        askNextClarification();
      } catch (err) {
        showError(err.message);
      }
    });

    return async function start() {
      try {
        const resp = await fetchWithTimeout("/api/captures", { method: "POST", body: formData });
        const capture = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(capture.detail || "Failed to save");

        hide(savingEl);
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

    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const formData = new FormData(form);
      form.reset();

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
      showStatus(statusEl, "Recalling...");
      const ticker = setInterval(() => {
        showStatus(statusEl, `Recalling... (${formatElapsed(performance.now() - startedAt)} elapsed)`);
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

  return { initCreatePage, initRecallPage };
})();
