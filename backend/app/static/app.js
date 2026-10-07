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

    clarificationDiscard.addEventListener("click", async () => {
      const current = pendingQueue[0];
      try {
        const updatedUnit = await fetch(`/api/memory-units/${current.unitId}/discard`, { method: "POST" }).then((r) =>
          r.json()
        );
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

  const SESSION_STATUS_TEXT = {
    needs_user_cue: (s) =>
      s.last_candidate_count > 0
        ? `Found ${s.last_candidate_count} related memor${s.last_candidate_count === 1 ? "y" : "ies"} so far -- one more detail would help.`
        : "Looking for another angle to search by...",
    success: () => "Found it.",
    exhausted: () => "I don't have enough stored evidence to determine this.",
    cancelled: () => "Recall cancelled.",
    error: () => "Something went wrong.",
  };

  function renderTranscript(transcriptEl, session) {
    transcriptEl.innerHTML = "";
    for (const m of session.messages) {
      const div = document.createElement("div");
      const isFinal = m.role === "system" && (m.text === session.final_answer);
      div.className = `turn ${m.role}${isFinal ? " final" : ""}`;
      div.textContent = m.text;
      transcriptEl.appendChild(div);
    }
  }

  // Iterative recall: a query may only be a weak cue. The session walks
  // SEARCH -> (ANSWER | ASK_USER) -> ... until it succeeds, gives up, or
  // the user cancels -- see app/retrieval/session.py.
  function initRecallPage() {
    const queryForm = document.getElementById("recall-form");
    const statusEl = document.getElementById("status");
    const conversation = document.getElementById("conversation");
    const sessionStatusEl = document.getElementById("session-status");
    const transcriptEl = document.getElementById("transcript");
    const replyForm = document.getElementById("reply-form");
    const replyInput = document.getElementById("reply");
    const cancelBtn = document.getElementById("cancel-session");
    const startOverBtn = document.getElementById("start-over");
    const sourcesEl = document.getElementById("sources");
    const sourcesWrap = document.getElementById("sources-wrap");

    let sessionId = null;
    const TERMINAL = new Set(["success", "exhausted", "cancelled", "error"]);

    function render(session) {
      sessionId = session.id;
      renderTranscript(transcriptEl, session);
      const textFn = SESSION_STATUS_TEXT[session.status];
      sessionStatusEl.textContent = textFn ? textFn(session) : session.status;

      const terminal = TERMINAL.has(session.status);
      replyForm.classList.toggle("hidden", terminal);
      startOverBtn.classList.toggle("hidden", !terminal);

      if (session.status === "success" && session.sources.length > 0) {
        show(sourcesWrap);
        renderSources(sourcesEl, session.sources);
      } else {
        hide(sourcesWrap);
      }

      if (!terminal) {
        replyInput.value = "";
        replyInput.focus();
      }
      show(conversation);
    }

    queryForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      hide(conversation);
      showStatus(statusEl, "Searching your memories...");
      const query = document.getElementById("query").value;

      try {
        const session = await postJSON("/api/recall/sessions", { query });
        hide(statusEl);
        render(session);
      } catch (err) {
        showStatus(statusEl, err.message, true);
      }
    });

    replyForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (!replyInput.value.trim()) return;
      showStatus(statusEl, "Searching...");
      try {
        const session = await postJSON(`/api/recall/sessions/${sessionId}/messages`, {
          text: replyInput.value,
        });
        hide(statusEl);
        render(session);
      } catch (err) {
        showStatus(statusEl, err.message, true);
      }
    });

    cancelBtn.addEventListener("click", async () => {
      try {
        const session = await postJSON(`/api/recall/sessions/${sessionId}/cancel`, {});
        render(session);
      } catch (err) {
        showStatus(statusEl, err.message, true);
      }
    });

    startOverBtn.addEventListener("click", () => {
      sessionId = null;
      hide(conversation);
      queryForm.reset();
      document.getElementById("query").focus();
    });
  }

  return { initCreatePage, initRecallPage };
})();
