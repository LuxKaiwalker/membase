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

  async function postJSON(url, body) {
    const resp = await fetch(url, {
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
    const resultCard = document.getElementById("result");
    const resultHeading = document.getElementById("result-heading");
    const derivedEl = document.getElementById("derived");

    // Capture may split into several units; each may independently have
    // a pending clarification. Ask them one at a time rather than all at
    // once (spec: don't turn this into an interrogation).
    let pendingQueue = [];
    let allUnits = [];

    function showResult() {
      hide(clarificationCard);
      resultHeading.textContent = allUnits.length > 1 ? `Saved ${allUnits.length} memories.` : "Saved.";
      renderUnits(derivedEl, allUnits);
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
        const resp = await fetch("/api/captures", { method: "POST", body: formData });
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
