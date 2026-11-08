// Library page (/app/library/) -- everything stored, newest first, with
// the per-unit processing state, any unanswered follow-up question, and
// deletion.
//
// Lifted verbatim out of the old single-file app.js (initCapturesPage,
// renamed along with the section itself: "Browse" -> "Library").
// Behaviour is unchanged. The status/structure badge class names are
// deliberately untouched (`status-badge status-badge-<state>`,
// `structure-badge-<state>`) -- app.css keys its badge variants off
// exactly those names, and two code paths below rewrite `className`
// wholesale when a unit's state changes.
(() => {
  "use strict";

  const {
    hide,
    show,
    showStatus,
    escapeHtml,
    fetchWithTimeout,
    postJSON,
    describeError,
    buildAttachmentList,
    isPublicMode,
  } = window.membase;

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
      const attachmentListPlaceholder = capture.attachments.length ? `<p class="muted capture-attachments"></p>` : "";
      const failedStructureUnits = capture.units.filter((u) => u.structure_status === "failed");
      const restructureButtons = failedStructureUnits
        .map(
          (u) =>
            `<button type="button" class="btn btn-secondary btn-sm capture-restructure" data-unit-id="${escapeHtml(u.id)}">Retry structuring</button>`
        )
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
            <input class="input capture-clarification-answer" type="text" placeholder="Your answer" />
            <button type="submit" class="btn btn-primary btn-sm capture-clarification-submit">Submit answer</button>
            <button type="button" class="btn btn-ghost btn-sm capture-clarification-discard">Discard this one</button>
          </form>
          <p class="capture-clarification-error status error hidden"></p>
        </div>`
        )
        .join("");
      card.innerHTML = `
        <p class="muted">${escapeHtml(when)}</p>
        <p class="capture-text">${escapeHtml(capture.raw_text)}</p>
        <p class="capture-badges">${unitBadges}</p>
        ${clarificationBlocks}
        ${attachmentListPlaceholder}
        ${restructureButtons}
        <p class="capture-restructure-error status error hidden"></p>
        <button type="button" class="btn btn-ghost btn-sm capture-delete">Delete</button>
      `;
      const attachmentsEl = card.querySelector(".capture-attachments");
      if (attachmentsEl) attachmentsEl.appendChild(buildAttachmentList(capture.attachments));

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
            const resp = await fetchWithTimeout(`/api/memory-units/${button.dataset.unitId}/restructure`, {
              method: "POST",
            });
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
      // Public mode: every control on this card that writes is disabled
      // rather than removed. Removing them would hide half of what the
      // Library actually is -- answering a clarification, discarding a
      // unit, retrying the structurer, deleting a capture are the page's
      // whole point -- but leaving them live would mean a click that can
      // only ever fail. Disabled-with-a-reason shows the affordance and
      // explains itself. (core.js refuses the request anyway, so this is
      // UI honesty, not the security boundary.)
      //
      // Listed explicitly rather than disabling every button on the
      // card: the attachment buttons buildAttachmentList() adds only
      // READ an attachment's bytes, which the snapshot exports and the
      // demo is specifically meant to show off. A blanket
      // querySelectorAll("button") would have switched off viewing the
      // demo's photographs.
      if (isPublicMode()) {
        const writeControls =
          ".capture-delete, .capture-restructure, .capture-clarification-submit, " +
          ".capture-clarification-discard, .capture-clarification-answer";
        for (const control of card.querySelectorAll(writeControls)) {
          control.disabled = true;
          control.title = "Needs a live backend — sign in to use a store you can change.";
        }
      }
      listEl.appendChild(card);
    }
  }

  function initLibraryPage() {
    const statusEl = document.getElementById("status");
    const listEl = document.getElementById("capture-list");
    const skipAllButton = document.getElementById("skip-all-followups");

    function updateSkipAllVisibility() {
      if (!skipAllButton) return;
      if (listEl.querySelector(".capture-clarification")) show(skipAllButton);
      else hide(skipAllButton);
      // Same reasoning as the per-card write controls below: a bulk
      // discard is the most destructive button on the page and cannot
      // work against a published snapshot.
      if (isPublicMode()) {
        skipAllButton.disabled = true;
        skipAllButton.title = "Needs a live backend — sign in to use a store you can change.";
      }
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
        const captures = await resp.json().catch(() => []);
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

  window.membase.initLibraryPage = initLibraryPage;
})();
