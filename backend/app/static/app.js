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

  function renderDerived(derivedEl, memory) {
    derivedEl.textContent = JSON.stringify(memory.derived, null, 2);
  }

  function initCreatePage() {
    const form = document.getElementById("create-form");
    const statusEl = document.getElementById("status");
    const clarificationCard = document.getElementById("clarification");
    const clarificationQuestion = document.getElementById("clarification-question");
    const clarificationForm = document.getElementById("clarification-form");
    const clarificationAnswer = document.getElementById("clarification-answer");
    const resultCard = document.getElementById("result");
    const derivedEl = document.getElementById("derived");

    let currentMemoryId = null;

    function showResult(memory) {
      hide(clarificationCard);
      renderDerived(derivedEl, memory);
      show(resultCard);
    }

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      hide(resultCard);
      hide(clarificationCard);
      showStatus(statusEl, "Saving...");

      const formData = new FormData(form);
      try {
        const resp = await fetch("/api/memories", { method: "POST", body: formData });
        const memory = await resp.json();
        if (!resp.ok) throw new Error(memory.detail || "Failed to save memory");

        currentMemoryId = memory.id;
        hide(statusEl);
        form.reset();

        if (memory.pending_clarification) {
          clarificationQuestion.textContent = memory.pending_clarification.question;
          show(clarificationCard);
        } else {
          showResult(memory);
        }
      } catch (err) {
        showStatus(statusEl, err.message, true);
      }
    });

    clarificationForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      try {
        const memory = await postJSON(`/api/memories/${currentMemoryId}/clarify`, {
          answer: clarificationAnswer.value,
        });
        clarificationAnswer.value = "";
        showResult(memory);
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
      const summary = s.derived && s.derived.summary ? s.derived.summary : s.caption;
      const when = new Date(s.created_at).toLocaleDateString();
      li.innerHTML = `<strong>${escapeHtml(summary)}</strong><br>
        <span class="muted">Memory ${escapeHtml(s.id)} &middot; stored ${when}</span><br>
        <span class="caption">&ldquo;${escapeHtml(s.caption)}&rdquo;</span>`;
      listEl.appendChild(li);
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
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
