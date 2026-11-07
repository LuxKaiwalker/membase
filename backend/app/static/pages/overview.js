// ===========================================================================
// Overview page (/app/, nav key "overview") -- the home / memory overview.
//
// There is no dashboard endpoint, by design, so this page is composed from
// three independent reads:
//
//   GET /api/graph/stats        counts + entities_by_kind + top_entities
//   GET /api/captures           newest-first, no server-side pagination
//   GET /api/episodes?limit=5   the most recent episodes
//
// Each panel renders from its own response and reports its own failure, so
// one dead endpoint degrades one panel instead of blanking the page.
//
// Two deliberate non-features:
//
//   * **Recent retrieval sessions are not shown.** There is no list/search
//     endpoint for recall sessions -- docs/TODO.md records that as an
//     explicit scope decision ("recall sessions are application history,
//     not memory"). The panel says so in one line rather than sitting empty
//     or being faked from localStorage, which would be this page inventing
//     a history the backend does not have.
//   * **No KPI theatre.** These are counts of a personal memory store, not
//     metrics: no giant numerals, no deltas, no sparklines. A count of 16
//     means sixteen things, and the restraint is the honesty.
// ===========================================================================
(() => {
  "use strict";

  const { hide, showStatus, escapeHtml, fetchWithTimeout, describeError, buildAttachmentList } =
    window.membase;

  // Newest captures to show before sending the user to Library. The
  // endpoint has no pagination (docs/TODO.md: not warranted at this
  // volume), so the slice happens here.
  const RECENT_CAPTURE_LIMIT = 5;
  const RECENT_EPISODE_LIMIT = 5;
  const TOP_ENTITY_LIMIT = 10;
  const CAPTURE_EXCERPT_CHARS = 260;

  const KIND_ORDER = ["person", "location", "organization", "object", "topic", "concept"];

  function kindRank(kind) {
    const i = KIND_ORDER.indexOf(kind);
    return i === -1 ? KIND_ORDER.length : i;
  }

  function plural(n, singular, pluralForm) {
    return `${n} ${n === 1 ? singular : pluralForm || singular + "s"}`;
  }

  function excerpt(text) {
    const trimmed = (text || "").trim();
    if (trimmed.length <= CAPTURE_EXCERPT_CHARS) return trimmed;
    return trimmed.slice(0, CAPTURE_EXCERPT_CHARS).trimEnd() + "...";
  }

  // A section shell: heading, optional sub-line, a link out to the section
  // that owns the full view, and a body the caller fills.
  function buildPanel({ eyebrow, title, note, link, linkLabel, quiet }) {
    const section = document.createElement("section");
    section.className = `card ov-panel${quiet ? " card-quiet" : ""}`;
    section.innerHTML = `
      <div class="card-head">
        <div class="ov-panel-title">
          <p class="eyebrow">${escapeHtml(eyebrow)}</p>
          <h2>${escapeHtml(title)}</h2>
        </div>
        ${link ? `<a class="btn btn-ghost btn-sm" href="${escapeHtml(link)}">${escapeHtml(linkLabel)}</a>` : ""}
      </div>
      ${note ? `<p class="muted ov-panel-note">${escapeHtml(note)}</p>` : ""}
      <div class="ov-panel-body"></div>
    `;
    return section;
  }

  function panelError(section, err, elapsedMs) {
    const body = section.querySelector(".ov-panel-body");
    body.innerHTML = "";
    const p = document.createElement("p");
    p.className = "status error";
    p.textContent = describeError(err, elapsedMs);
    body.appendChild(p);
  }

  function panelEmpty(section, message) {
    const body = section.querySelector(".ov-panel-body");
    body.innerHTML = `<p class="empty-state">${escapeHtml(message)}</p>`;
  }

  // ---------------------------------------------------------------------
  // 1. Recent memories

  function renderRecentCaptures(section, captures) {
    const body = section.querySelector(".ov-panel-body");
    body.innerHTML = "";
    if (!captures.length) {
      panelEmpty(section, "Nothing saved yet.");
      return;
    }
    const list = document.createElement("ol");
    list.className = "ov-captures";
    for (const capture of captures.slice(0, RECENT_CAPTURE_LIMIT)) {
      const item = document.createElement("li");
      item.className = "ov-capture";
      const when = new Date(capture.created_at).toLocaleString();

      // The three states a capture's units can be in have to be told
      // apart: still being processed, processed into N memories, or
      // processed into none at all (see CaptureOut.processing_status).
      let unitsHtml;
      if (capture.processing_status === "pending" || capture.processing_status === "processing") {
        unitsHtml = `<span class="status-badge status-badge-processing">still processing</span>`;
      } else if (capture.processing_status === "failed") {
        unitsHtml = `<span class="status-badge status-badge-failed">processing failed</span>`;
      } else if (!capture.units.length) {
        unitsHtml = `<span class="pill">no separate memories extracted</span>`;
      } else {
        unitsHtml = capture.units
          .map(
            (unit) =>
              `<span class="status-badge status-badge-${escapeHtml(unit.status)}">${escapeHtml(unit.status)}</span>`
          )
          .join(" ");
      }

      item.innerHTML = `
        <p class="muted ov-capture-when">${escapeHtml(when)}</p>
        <p class="ov-capture-text">${escapeHtml(excerpt(capture.raw_text))}</p>
        <div class="row row-tight ov-capture-meta">
          ${unitsHtml}
          ${capture.units.length ? `<span class="pill">${escapeHtml(plural(capture.units.length, "memory", "memories"))}</span>` : ""}
        </div>
        ${capture.attachments.length ? `<p class="muted ov-capture-files"></p>` : ""}
      `;
      const filesEl = item.querySelector(".ov-capture-files");
      if (filesEl) filesEl.appendChild(buildAttachmentList(capture.attachments));
      list.appendChild(item);
    }
    body.appendChild(list);

    if (captures.length > RECENT_CAPTURE_LIMIT) {
      const more = document.createElement("p");
      more.className = "muted ov-more";
      more.textContent = `${captures.length - RECENT_CAPTURE_LIMIT} older note${captures.length - RECENT_CAPTURE_LIMIT === 1 ? "" : "s"} in the Library.`;
      body.appendChild(more);
    }
  }

  // ---------------------------------------------------------------------
  // 2. Memory statistics

  function kindPills(counts) {
    const entries = Object.entries(counts || {}).sort(
      (a, b) => b[1] - a[1] || kindRank(a[0]) - kindRank(b[0]) || a[0].localeCompare(b[0])
    );
    if (!entries.length) return "";
    return `<div class="row row-tight ov-kinds">${entries
      .map(
        ([kind, count]) =>
          `<span class="pill"><span class="ov-kind-name">${escapeHtml(kind)}</span> ${count}</span>`
      )
      .join("")}</div>`;
  }

  function renderStats(section, stats, timeline) {
    const body = section.querySelector(".ov-panel-body");
    const rows = [
      {
        label: "Searchable memories",
        value: stats.ready_units,
        note: "Memories that have finished processing and can be recalled.",
      },
      {
        label: "Entities",
        value: stats.entities,
        note: `${plural(stats.entity_links, "mention")} across those memories.`,
        extra: kindPills(stats.entities_by_kind),
      },
      {
        label: "Associations",
        value: stats.associations,
        note: "Edges linking one memory to another.",
        extra: kindPills(stats.associations_by_kind),
      },
      {
        label: "Relations",
        value: stats.relations,
        note: "Subject–predicate–object facts extracted from your notes.",
      },
    ];
    if (timeline) {
      rows.push({
        label: "Episodes",
        value: timeline.total,
        note:
          timeline.undated_count > 0
            ? `${plural(timeline.undated_count, "episode")} whose date isn’t known.`
            : "Every episode has a resolvable date.",
      });
    }

    body.innerHTML = `
      <dl class="ov-stats">
        ${rows
          .map(
            (row) => `
          <div class="ov-stat">
            <dt>${escapeHtml(row.label)}</dt>
            <dd>
              <span class="ov-stat-value">${Number(row.value) || 0}</span>
              ${row.note ? `<span class="muted ov-stat-note">${escapeHtml(row.note)}</span>` : ""}
              ${row.extra || ""}
            </dd>
          </div>`
          )
          .join("")}
      </dl>
    `;
  }

  // ---------------------------------------------------------------------
  // 3. What your memory is mostly about

  function renderTopEntities(section, stats) {
    const body = section.querySelector(".ov-panel-body");
    const entities = (stats.top_entities || []).slice(0, TOP_ENTITY_LIMIT);
    if (!entities.length) {
      panelEmpty(section, "No entities have been extracted yet.");
      return;
    }
    const max = Math.max(...entities.map((e) => e.mention_count || 0), 1);
    body.innerHTML = `
      <ul class="ov-entities">
        ${entities
          .map((entity) => {
            // The bar length is the mention count relative to the most
            // mentioned entity -- a comparison within this list only, not a
            // score of any kind.
            const share = Math.round(((entity.mention_count || 0) / max) * 100);
            return `
          <li class="ov-entity">
            <a class="ov-entity-name" href="/app/explorer/?entity=${encodeURIComponent(entity.id)}">${escapeHtml(entity.display_name)}</a>
            <span class="pill ov-entity-kind">${escapeHtml(entity.kind)}</span>
            <span class="ov-entity-bar" style="--ov-share: ${share}%" aria-hidden="true"></span>
            <span class="muted ov-entity-count">${escapeHtml(plural(entity.mention_count || 0, "mention"))}</span>
          </li>`;
          })
          .join("")}
      </ul>
    `;
  }

  // ---------------------------------------------------------------------
  // 4. Recent episodes

  function renderRecentEpisodes(section, timeline) {
    const body = section.querySelector(".ov-panel-body");
    const episodes = timeline.episodes || [];
    if (!episodes.length) {
      panelEmpty(section, "No episodes yet — they are derived once memories finish processing.");
      return;
    }
    // `date_display` verbatim: it was rendered server-side at exactly the
    // precision that is known (see backend/app/memory/temporal.py), so a
    // month-precision episode cannot be shown here as a specific day.
    body.innerHTML = `
      <ul class="ov-episodes">
        ${episodes
          .map((episode) => {
            const undated = episode.started_on === null;
            return `
          <li class="ov-episode">
            <span class="ov-episode-title">${escapeHtml(episode.title)}</span>
            <span class="ov-episode-when${undated ? " ov-episode-when-unknown" : ""}">${escapeHtml(episode.date_display)}</span>
            <span class="pill">${escapeHtml(plural(episode.member_count, "memory", "memories"))}</span>
          </li>`;
          })
          .join("")}
      </ul>
    `;
    if (timeline.undated_count > 0) {
      const note = document.createElement("p");
      note.className = "muted ov-more";
      note.textContent = `${plural(timeline.undated_count, "episode")} in the store have no resolvable date; the Timeline lists them as undated rather than placing them.`;
      body.appendChild(note);
    }
  }

  // ---------------------------------------------------------------------

  function initOverviewPage() {
    const root = document.getElementById("overview-root");
    if (!root) return;
    const statusEl = document.getElementById("status");

    const capturesPanel = buildPanel({
      eyebrow: "Recent memories",
      title: "The last few things you saved",
      link: "/app/library/",
      linkLabel: "All memories",
    });
    const statsPanel = buildPanel({
      eyebrow: "Memory statistics",
      title: "What is in the store",
    });
    const entitiesPanel = buildPanel({
      eyebrow: "Mostly about",
      title: "What your memory is mostly about",
      note: "The most-mentioned people, places and subjects. Open one in the Explorer to see what connects to it.",
      link: "/app/explorer/",
      linkLabel: "Explorer",
    });
    const episodesPanel = buildPanel({
      eyebrow: "Recent episodes",
      title: "Experiences, newest first",
      link: "/app/timeline/",
      linkLabel: "Timeline",
    });

    // The one panel with nothing to fetch. Stated plainly, once, and not
    // apologised for: there is no endpoint, because sessions are not
    // memory.
    const sessionsPanel = buildPanel({
      eyebrow: "Recent retrieval sessions",
      title: "Not kept",
      quiet: true,
    });
    sessionsPanel.querySelector(".ov-panel-body").innerHTML = `
      <p class="ov-sessions-note">
        Recall sessions are application history, not memory. There is no
        endpoint that lists or searches past sessions, so there is nothing
        here to show — and nothing reconstructed from this browser to stand in
        for it. A session is reachable by its own id while you are in it;
        anything you want to keep from one, you save as a memory.
      </p>
    `;

    // Every fetched panel starts in its own loading state; the sessions
    // panel never has one, because it never fetches anything.
    for (const panel of [capturesPanel, statsPanel, entitiesPanel, episodesPanel]) {
      panel.querySelector(".ov-panel-body").innerHTML =
        '<p class="muted"><span class="spinner" aria-hidden="true"></span>Loading...</p>';
    }

    async function getJSON(path) {
      const resp = await fetchWithTimeout(path);
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const err = new Error(data.detail || `Request failed (${resp.status})`);
        err.apiStatus = resp.status;
        throw err;
      }
      return data;
    }

    (async () => {
      const startedAt = performance.now();
      showStatus(statusEl, "Reading your memory store...");

      const [statsResult, capturesResult, timelineResult] = await Promise.allSettled([
        getJSON("/api/graph/stats"),
        getJSON("/api/captures"),
        getJSON(`/api/episodes?limit=${RECENT_EPISODE_LIMIT}`),
      ]);
      const elapsed = performance.now() - startedAt;

      const stats = statsResult.status === "fulfilled" ? statsResult.value : null;
      const captures =
        capturesResult.status === "fulfilled" && Array.isArray(capturesResult.value)
          ? capturesResult.value
          : null;
      const timeline = timelineResult.status === "fulfilled" ? timelineResult.value : null;

      // Everything failed: one page-level error, not four identical ones.
      if (!stats && !captures && !timeline) {
        root.innerHTML = "";
        showStatus(statusEl, describeError(statsResult.reason, elapsed), true);
        return;
      }
      hide(statusEl);

      // A genuinely fresh store. Shown as one honest statement instead of
      // five panels of zeroes, which read as a failure rather than as a new
      // install.
      const freshStore =
        stats !== null &&
        captures !== null &&
        stats.ready_units === 0 &&
        captures.length === 0 &&
        (!timeline || timeline.total === 0);
      if (freshStore) {
        root.innerHTML = "";
        const empty = document.createElement("div");
        empty.className = "card ov-empty";
        empty.innerHTML = `
          <p class="eyebrow">Nothing stored yet</p>
          <h2>Your memory store is empty</h2>
          <p>Write something down and membase will segment it, extract the people and
          places in it, connect it to whatever it relates to, and place it in time —
          all of which is derived from what you wrote, so there is nothing to
          configure first.</p>
          <p class="row ov-empty-actions">
            <a class="btn btn-primary" href="/app/store/">Store your first memory</a>
          </p>
        `;
        root.appendChild(empty);
        root.appendChild(sessionsPanel);
        return;
      }

      root.innerHTML = "";
      root.appendChild(capturesPanel);
      root.appendChild(statsPanel);
      root.appendChild(entitiesPanel);
      root.appendChild(episodesPanel);
      root.appendChild(sessionsPanel);

      if (captures) renderRecentCaptures(capturesPanel, captures);
      else panelError(capturesPanel, capturesResult.reason, elapsed);

      if (stats) {
        renderStats(statsPanel, stats, timeline);
        renderTopEntities(entitiesPanel, stats);
      } else {
        panelError(statsPanel, statsResult.reason, elapsed);
        panelError(entitiesPanel, statsResult.reason, elapsed);
      }

      if (timeline) renderRecentEpisodes(episodesPanel, timeline);
      else panelError(episodesPanel, timelineResult.reason, elapsed);
    })();
  }

  window.membase.initOverviewPage = initOverviewPage;
})();
