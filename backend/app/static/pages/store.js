// Store page (/app/store/) -- writing a new memory.
//
// Lifted verbatim out of the old single-file app.js (initCreatePage);
// behaviour is unchanged, only the file it lives in and the class names
// on the buttons it generates (now the shared .btn vocabulary from
// app.css) are new.
(() => {
  "use strict";

  const { hide, show, showStatus, fetchWithTimeout, buildAttachmentList } = window.membase;

  // POST /api/captures now returns the instant it has durably persisted
  // the capture's raw text and attachments -- zero LLM calls in that
  // request path at all (see docs/DECISIONS.md "Restore the multi-call
  // ingestion pipeline"). `capture.units` is therefore always `[]` on
  // return: there is nothing derived to preview here any more.
  // Segmentation, clarification, enrichment, embedding, and structuring
  // all happen afterward in one background task; the Library page is the
  // one place that progress is ever visualized (no polling here).

  function buildCaptureCard() {
    const card = document.createElement("div");
    card.className = "card capture-card";
    card.innerHTML = `
      <p class="capture-saving muted"><span class="spinner" aria-hidden="true"></span>Saving...</p>
      <div class="capture-result hidden">
        <h2 class="capture-result-heading">Saved.</h2>
        <p class="muted capture-attachment-list"></p>
        <p class="muted">
          Processing in the background -- check the Library tab shortly
          for extracted details (and to answer a follow-up question, if
          one comes up).
        </p>
        <p class="muted capture-safe-to-leave">
          Your words are already saved. Everything else keeps running on
          the server, so it's safe to close this page now.
        </p>
      </div>
      <p class="capture-error status error hidden"></p>
    `;
    return card;
  }

  // Drives one capture from submission to the result card, entirely
  // through this card's own elements -- nothing here is shared with any
  // other card, so several captures can be mid-flight at once.
  function runCapture(card, formData) {
    const savingEl = card.querySelector(".capture-saving");
    const resultSection = card.querySelector(".capture-result");
    const errorEl = card.querySelector(".capture-error");
    const attachmentListEl = card.querySelector(".capture-attachment-list");

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
        attachmentListEl.innerHTML = "";
        attachmentListEl.appendChild(buildAttachmentList(capture.attachments));
        show(resultSection);
      } catch (err) {
        showError(err.message);
      }
    };
  }

  function initStorePage() {
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
        remove.className = "btn btn-ghost btn-sm";
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

  window.membase.initStorePage = initStorePage;
})();
