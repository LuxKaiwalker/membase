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

  // A hung request (e.g. a stuck DB transaction server-side) must never
  // leave the page silently spinning forever with no feedback -- that's
  // indistinguishable from a frozen UI. 90s is generous enough for a
  // real multi-round recall (several sequential LLM calls) but still
  // bounds the worst case to a visible error instead of an infinite wait.
  const REQUEST_TIMEOUT_MS = 90_000;

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
      throw new Error(data.detail ? JSON.stringify(data.detail) : `Request failed (${resp.status})`);
    }
    return data;
  }

  function renderUnits(derivedEl, units) {
    const lines = units.map((u) => ({
      content: u.authoritative_content,
      meaning: u.semantic && u.semantic.meaning,
      aliases: u.semantic && u.semantic.aliases,
    }));
    derivedEl.textContent = JSON.stringify(lines, null, 2);
  }

  function initCreatePage() {
    const form = document.getElementById("create-form");
    const statusEl = document.getElementById("status");
    const clarificationCard = document.getElementById("clarification");
    const clarificationQuestion = document.getElementById("clarification-question");
    const clarificationForm = document.getElementById("clarification-form");
    const clarificationAnswer = document.getElementById("clarification-answer");
    const clarificationDiscard = document.getElementById("clarification-discard");
    const resultCard = document.getElementById("result");
    const resultHeading = document.getElementById("result-heading");
    const derivedEl = document.getElementById("derived");

    // Capture may split into several units; each may independently have
    // a pending clarification. Ask them one at a time rather than all at
    // once (spec: don't turn this into an interrogation). A unit awaiting
    // clarification is NOT yet searchable -- "Saved" only ever counts
    // units that reached "ready".
    let pendingQueue = [];
    let allUnits = [];

    function showResult() {
      hide(clarificationCard);
      const ready = allUnits.filter((u) => u.status === "ready");
      resultHeading.textContent = ready.length > 1 ? `Saved ${ready.length} memories.` : ready.length === 1 ? "Saved." : "Nothing saved.";
      renderUnits(derivedEl, ready);
      show(resultCard);
    }

    function askNextClarification() {
      if (pendingQueue.length === 0) {
        showResult();
        return;
      }
      clarificationQuestion.textContent = pendingQueue[0].question;
      show(clarificationCard);
    }

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      hide(resultCard);
      hide(clarificationCard);
      showStatus(statusEl, "Saving...");

      const formData = new FormData(form);
      try {
        const resp = await fetchWithTimeout("/api/captures", { method: "POST", body: formData });
        const capture = await resp.json();
        if (!resp.ok) throw new Error(capture.detail || "Failed to save");

        hide(statusEl);
        form.reset();

        allUnits = capture.units;
        pendingQueue = capture.units
          .filter((u) => u.pending_clarification)
          .map((u) => ({ unitId: u.id, question: u.pending_clarification.question }));
        askNextClarification();
      } catch (err) {
        showStatus(statusEl, err.message, true);
      }
    });

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
        askNextClarification();
      } catch (err) {
        showStatus(statusEl, err.message, true);
      }
    });

    clarificationDiscard.addEventListener("click", async () => {
      const current = pendingQueue[0];
      try {
        const updatedUnit = await fetchWithTimeout(`/api/memory-units/${current.unitId}/discard`, {
          method: "POST",
        }).then((r) => r.json());
        allUnits = allUnits.map((u) => (u.id === updatedUnit.id ? updatedUnit : u));
        pendingQueue = pendingQueue.slice(1);
        askNextClarification();
      } catch (err) {
        showStatus(statusEl, err.message, true);
      }
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
      showStatus(statusEl, "Recalling...");
      const query = document.getElementById("query").value;

      try {
        const data = await postJSON("/api/recall", { query });
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
        showStatus(statusEl, err.message, true);
      }
    });
  }

  return { initCreatePage, initRecallPage };
})();
