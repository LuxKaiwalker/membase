/*
  Explorer page: fetch the associative graph, drive the canvas renderer in
  static/graph-canvas.js, and run the filters, entity search and inspector.

  Division of labour:
    graph-canvas.js  -- layout, painting, hit-testing, camera. Knows nothing
                        about HTTP or about what an entity kind means.
    this file        -- data fetching, filter state, and turning a clicked
                        node or edge into readable prose.

  `graph-canvas.js` is injected and awaited here rather than listed in
  base.html, because base.html loads exactly core.js plus one page script and
  is shared by every section -- adding a second <script> there for a file only
  this page needs would make every other page pay for it.

  Two things this page is deliberately careful about, both because the backend
  goes to lengths to establish them and a UI can quietly undo either:

  * `truncated` on GraphOut means a node cap dropped part of the graph. It is
    surfaced as a notice. Presenting a partial graph as if it were the whole
    store is exactly what that flag exists to prevent.
  * An edge's `origin` (explicit / extracted / inferred) is rendered by line
    style, not colour, and always spelled out in words in the inspector. An
    association the system computed must never read as something the user
    stated.
*/
(function () {
  "use strict";

  const { escapeHtml, fetchWithTimeout, describeError, showStatus, hide, show, isPublicMode } =
    window.membase;

  const CANVAS_SRC = "/static/graph-canvas.js";
  const MAX_NODES = 400;
  const SEARCH_LIMIT = 12;

  // Entity kinds shown by default. `topic` and `concept` are off on purpose:
  // on a real store they outnumber the specific kinds several times over (the
  // demo corpus has ~83 topics and ~69 concepts against 2 people and 23
  // locations), so an unfiltered first view is an unreadable hairball. They
  // are not hidden silently -- the note under the chips says so, and one
  // click brings them back.
  const DEFAULT_OFF_KINDS = ["topic", "concept"];

  const KIND_LABELS = {
    person: "people",
    organization: "organizations",
    location: "locations",
    topic: "topics",
    concept: "concepts",
    object: "objects",
  };

  const ASSOCIATION_LABELS = {
    shared_entity: "shared entities",
    same_capture: "written together",
    structurer_link: "judged related",
  };

  const ORIGIN_WORDS = {
    explicit: "you stated this",
    extracted: "extracted from your words",
    inferred: "inferred by the system",
  };

  const ROLE_WORDS = {
    people: "named as a person",
    organizations: "named as an organization",
    locations: "named as a place",
    topics: "listed as a topic",
    definition_term: "a term this memory explains",
    relation_subject: "the subject of something this memory asserts",
    relation_object: "the object of something this memory asserts",
    visual_concepts: "visible in an attached image",
  };

  /* ------------------------------------------------------------------ utils */

  function loadRenderer() {
    if (window.membase.GraphCanvas) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${CANVAS_SRC}"]`);
      if (existing) {
        existing.addEventListener("load", () => resolve());
        existing.addEventListener("error", () => reject(new Error("renderer failed to load")));
        return;
      }
      const script = document.createElement("script");
      script.src = CANVAS_SRC;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("renderer failed to load"));
      document.head.appendChild(script);
    });
  }

  async function getJSON(path) {
    const response = await fetchWithTimeout(path, { method: "GET" });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
      const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      error.apiStatus = response.status;
      throw error;
    }
    return body;
  }

  function idPart(nodeId) {
    return String(nodeId).slice(String(nodeId).indexOf(":") + 1);
  }

  function pill(text, extraClass) {
    return `<span class="pill${extraClass ? ` ${extraClass}` : ""}">${escapeHtml(text)}</span>`;
  }

  function provenanceTag(origin) {
    const cls = origin === "inferred" ? "prov-inferred" : origin === "explicit" ? "prov-original" : "prov-extracted";
    return `<span class="provenance ${cls}">${escapeHtml(ORIGIN_WORDS[origin] || origin)}</span>`;
  }

  function dl(rows) {
    const body = rows
      .filter(([, value]) => value)
      .map(
        ([key, value]) =>
          `<div class="explorer-field"><span class="label">${escapeHtml(key)}</span><span>${value}</span></div>`
      )
      .join("");
    return body ? `<div class="stack-tight">${body}</div>` : "";
  }

  /* ------------------------------------------------------------------- page */

  function initExplorerPage() {
    const canvasEl = document.getElementById("explorer-canvas");
    if (!canvasEl) return;

    const statusEl = document.getElementById("explorer-status");
    const noticeEl = document.getElementById("explorer-notice");
    const emptyEl = document.getElementById("explorer-empty");
    const countsEl = document.getElementById("explorer-counts");
    const tooltipEl = document.getElementById("explorer-tooltip");
    const stageEl = document.getElementById("explorer-stage");
    const inspectorEl = document.getElementById("explorer-inspector");
    const inspectorBody = document.getElementById("explorer-inspector-body");
    const inspectorHeading = document.getElementById("explorer-inspector-heading");
    const inspectorClose = document.getElementById("explorer-inspector-close");
    const entityChipsEl = document.getElementById("explorer-entity-kinds");
    const entityKindsNote = document.getElementById("explorer-entity-kinds-note");
    const assocChipsEl = document.getElementById("explorer-association-kinds");
    const hopsEl = document.getElementById("explorer-hops");
    const hopsNote = document.getElementById("explorer-hops-note");
    const backBtn = document.getElementById("explorer-back");
    const fitBtn = document.getElementById("explorer-reset-view");
    const searchForm = document.getElementById("explorer-search-form");
    const searchInput = document.getElementById("explorer-search-input");
    const searchResults = document.getElementById("explorer-search-results");

    const state = {
      graph: null,
      stats: null,
      entityKinds: new Set(),
      associationKinds: new Set(),
      hops: 1,
      centerUnitId: null,
      canvas: null,
      inFlight: 0,
    };

    /* -- chips ------------------------------------------------------------ */

    function renderChips(container, available, selected, onToggle) {
      container.innerHTML = "";
      available.forEach(({ key, label, count }) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.setAttribute("aria-pressed", selected.has(key) ? "true" : "false");
        chip.textContent = count === undefined ? label : `${label} (${count})`;
        chip.addEventListener("click", () => {
          if (selected.has(key)) selected.delete(key);
          else selected.add(key);
          chip.setAttribute("aria-pressed", selected.has(key) ? "true" : "false");
          onToggle();
        });
        container.appendChild(chip);
      });
    }

    function updateKindNote() {
      const all = Object.keys(state.stats ? state.stats.entities_by_kind : {});
      const hidden = all.filter((kind) => !state.entityKinds.has(kind));
      entityKindsNote.textContent = hidden.length
        ? `Hiding ${hidden.map((k) => KIND_LABELS[k] || k).join(", ")} — click to include. ` +
          "They are the most numerous kinds, so leaving them out keeps the graph readable."
        : "Showing every entity kind.";
    }

    function buildFilters() {
      // The public snapshot cannot serve these honestly. The server
      // applies kind filters BEFORE the max_nodes cap, so filtering an
      // already-capped snapshot in the browser would show a different
      // graph than the live API returns for the same filter -- quietly,
      // with nothing on screen to say so. Hiding the controls and saying
      // why is the honest option; approximating them is not. The hop
      // selector and centring still work, because the exporter writes a
      // file per (hops, centre) pair (scripts/export_demo.py).
      if (isPublicMode()) {
        entityChipsEl.innerHTML = "";
        assocChipsEl.innerHTML = "";
        entityKindsNote.textContent =
          "Kind filters need a live backend: the server applies them before capping the graph, " +
          "so filtering the published snapshot here would show you something the real API would not. " +
          "This is the full default graph.";
        // Both sets stay empty on purpose -- query() only sends the
        // entity_kinds/association_kinds parameters when they are
        // non-empty, and the snapshot only has the unfiltered URLs.
        return;
      }
      const byKind = (state.stats && state.stats.entities_by_kind) || {};
      const entityAvailable = Object.keys(byKind)
        .sort((a, b) => byKind[b] - byKind[a])
        .map((kind) => ({ key: kind, label: KIND_LABELS[kind] || kind, count: byKind[kind] }));
      if (!state.entityKinds.size) {
        entityAvailable.forEach(({ key }) => {
          if (!DEFAULT_OFF_KINDS.includes(key)) state.entityKinds.add(key);
        });
      }
      renderChips(entityChipsEl, entityAvailable, state.entityKinds, () => {
        updateKindNote();
        load();
      });
      updateKindNote();

      const assocByKind = (state.stats && state.stats.associations_by_kind) || {};
      const assocAvailable = Object.keys(assocByKind)
        .sort()
        .map((kind) => ({
          key: kind,
          label: ASSOCIATION_LABELS[kind] || kind.replace(/_/g, " "),
          count: assocByKind[kind],
        }));
      if (!state.associationKinds.size) {
        assocAvailable.forEach(({ key }) => state.associationKinds.add(key));
      }
      renderChips(assocChipsEl, assocAvailable, state.associationKinds, load);
    }

    /* -- inspector -------------------------------------------------------- */

    function openInspector(heading, html) {
      inspectorHeading.textContent = heading;
      inspectorBody.innerHTML = html;
      show(inspectorClose);
      inspectorEl.focus({ preventScroll: true });
    }

    function clearInspector() {
      inspectorHeading.textContent = "Inspector";
      inspectorBody.innerHTML =
        '<p class="muted">Click a memory, an entity, or a link between them. ' +
        "Searching for an entity opens it here too.</p>";
      hide(inspectorClose);
      if (state.canvas) state.canvas.clearSelection();
    }

    function renderEdge(edge) {
      const from = state.canvas.nodeById.get(edge.source);
      const to = state.canvas.nodeById.get(edge.target);
      const what =
        edge.edge_type === "mentions"
          ? "This memory mentions this entity"
          : edge.edge_type === "association"
            ? "These two memories are linked"
            : "One memory asserts this about these two entities";
      openInspector(
        "Why these are connected",
        [
          `<p class="explorer-edge-ends">${escapeHtml(from ? from.label : edge.source)} ` +
            `<span class="muted">—</span> ${escapeHtml(to ? to.label : edge.target)}</p>`,
          `<p>${escapeHtml(what)}.</p>`,
          edge.explanation ? `<p class="explorer-why">${escapeHtml(edge.explanation)}</p>` : "",
          dl([
            ["Kind", pill(edge.label || edge.kind)],
            ["Where it came from", provenanceTag(edge.origin)],
            ["Strength", pill(Number(edge.weight).toFixed(2))],
          ]),
        ].join("")
      );
    }

    async function renderUnit(node) {
      const unitId = idPart(node.id);
      openInspector(
        "Memory",
        `<p class="explorer-content">${escapeHtml(node.label)}</p><p class="muted">Loading…</p>`
      );
      try {
        const [unit, associations, episodes] = await Promise.all([
          getJSON(`/api/memory-units/${unitId}`),
          getJSON(`/api/graph/units/${unitId}/associations?hops=${state.hops}&limit=12`),
          getJSON(`/api/episodes/for-unit/${unitId}`),
        ]);
        const semantic = unit.semantic || {};
        const listField = (key) =>
          Array.isArray(semantic[key]) && semantic[key].length
            ? semantic[key].map((v) => pill(v)).join(" ")
            : "";
        const assocHtml = associations.length
          ? associations
              .map(
                (a) =>
                  `<li><p>${escapeHtml(a.unit.authoritative_content)}</p>` +
                  `<p class="explorer-why">${escapeHtml(a.why || "Connected.")}</p>` +
                  `<p class="muted small">${a.hops} ${a.hops === 1 ? "hop" : "hops"} away · strength ${a.strength.toFixed(2)}</p>` +
                  `<button type="button" class="btn btn-ghost btn-sm" data-focus-unit="${escapeHtml(a.unit.id)}">Focus this memory</button></li>`
              )
              .join("")
          : '<li class="muted">Nothing is connected to this memory yet.</li>';
        const episodeHtml = episodes.length
          ? episodes
              .map(
                (e) =>
                  `<li>${escapeHtml(e.title)} <span class="muted small">${escapeHtml(e.date_display)} · ` +
                  `${escapeHtml(e.boundary_confidence)} boundaries</span></li>`
              )
              .join("")
          : '<li class="muted">Not part of any episode.</li>';

        openInspector(
          "Memory",
          [
            `<p class="explorer-content">${escapeHtml(unit.authoritative_content)}</p>`,
            semantic.meaning ? `<p class="muted">${escapeHtml(semantic.meaning)}</p>` : "",
            dl([
              ["People", listField("people")],
              ["Places", listField("locations")],
              ["Organizations", listField("organizations")],
              ["Topics", listField("topics")],
              ["Visible in image", listField("visual_concepts")],
              ["Recorded date", semantic.event_date ? pill(semantic.event_date) : ""],
            ]),
            `<p class="eyebrow explorer-section">Episodes</p><ul class="explorer-list">${episodeHtml}</ul>`,
            `<p class="eyebrow explorer-section">Connected memories</p><ul class="explorer-list">${assocHtml}</ul>`,
            `<button type="button" class="btn btn-secondary btn-sm" data-center-unit="${escapeHtml(unit.id)}">Explore outward from here</button>`,
          ].join("")
        );
      } catch (error) {
        openInspector("Memory", `<p class="error">${escapeHtml(describeError(error, 0))}</p>`);
      }
    }

    async function renderEntity(entityId) {
      openInspector("Entity", '<p class="muted">Loading…</p>');
      try {
        const detail = await getJSON(`/api/graph/entities/${entityId}`);
        const entity = detail.entity;
        const mentions = detail.mentions.length
          ? detail.mentions
              .map(
                (m) =>
                  `<li><p>${escapeHtml(m.content)}</p>` +
                  `<p class="muted small">${escapeHtml(ROLE_WORDS[m.role] || m.role)} · ${provenanceTag(m.origin)}</p>` +
                  `<button type="button" class="btn btn-ghost btn-sm" data-center-unit="${escapeHtml(m.unit_id)}">Explore from this memory</button></li>`
              )
              .join("")
          : '<li class="muted">No memory mentions this yet.</li>';
        const relations = detail.relations.length
          ? detail.relations
              .map(
                (r) =>
                  `<li>${escapeHtml(r.subject)} <em>${escapeHtml(r.predicate)}</em> ${escapeHtml(r.object)}` +
                  (r.occurred_on ? ` <span class="muted small">(${escapeHtml(r.occurred_on)})</span>` : "") +
                  `</li>`
              )
              .join("")
          : '<li class="muted">Nothing asserted about this.</li>';

        openInspector(
          entity.display_name,
          [
            dl([
              ["Kind", pill(KIND_LABELS[entity.kind] || entity.kind)],
              ["Mentioned in", pill(`${entity.mention_count} ${entity.mention_count === 1 ? "memory" : "memories"}`)],
              [
                "Also matched as",
                entity.aliases && entity.aliases.length ? entity.aliases.map((a) => pill(a)).join(" ") : "",
              ],
            ]),
            `<p class="eyebrow explorer-section">Memories mentioning it</p><ul class="explorer-list">${mentions}</ul>`,
            `<p class="eyebrow explorer-section">Asserted relationships</p><ul class="explorer-list">${relations}</ul>`,
          ].join("")
        );
      } catch (error) {
        openInspector("Entity", `<p class="error">${escapeHtml(describeError(error, 0))}</p>`);
      }
    }

    inspectorBody.addEventListener("click", (event) => {
      const focusTarget = event.target.closest("[data-focus-unit]");
      if (focusTarget) {
        const nodeId = `unit:${focusTarget.getAttribute("data-focus-unit")}`;
        if (state.canvas && state.canvas.nodeById.has(nodeId)) {
          state.canvas.selectNode(nodeId, { center: true });
          renderUnit(state.canvas.nodeById.get(nodeId));
        } else {
          centerOnUnit(focusTarget.getAttribute("data-focus-unit"));
        }
        return;
      }
      const centerTarget = event.target.closest("[data-center-unit]");
      if (centerTarget) centerOnUnit(centerTarget.getAttribute("data-center-unit"));
    });

    inspectorClose.addEventListener("click", clearInspector);

    /* -- fetching --------------------------------------------------------- */

    function centerOnUnit(unitId) {
      state.centerUnitId = unitId;
      show(backBtn);
      load();
    }

    function query() {
      const params = new URLSearchParams();
      params.set("max_nodes", String(MAX_NODES));
      params.set("hops", String(state.hops));
      if (state.centerUnitId) params.set("center_unit_id", state.centerUnitId);
      if (state.entityKinds.size) params.set("entity_kinds", [...state.entityKinds].join(","));
      if (state.associationKinds.size) params.set("association_kinds", [...state.associationKinds].join(","));
      return `/api/graph?${params.toString()}`;
    }

    function renderNotice(graph) {
      const parts = [];
      if (graph.truncated) {
        parts.push(
          `This view is incomplete: it shows ${graph.nodes.filter((n) => n.node_type === "unit").length} ` +
            `of ${graph.total_units} memories, capped at ${MAX_NODES} nodes. Narrow the filters to see a complete slice.`
        );
      }
      if (state.centerUnitId) {
        parts.push("Centred on one memory’s neighbourhood.");
      }
      if (parts.length) {
        noticeEl.textContent = parts.join(" ");
        show(noticeEl);
      } else {
        hide(noticeEl);
      }
    }

    async function load() {
      const started = performance.now();
      const token = ++state.inFlight;
      showStatus(statusEl, "Loading the graph…");
      try {
        if (!state.stats) {
          state.stats = await getJSON("/api/graph/stats");
          buildFilters();
        }
        const graph = await getJSON(query());
        if (token !== state.inFlight) return; // a newer request already won
        state.graph = graph;
        hide(statusEl);

        const unitCount = graph.nodes.filter((n) => n.node_type === "unit").length;
        const entityCount = graph.nodes.length - unitCount;
        countsEl.textContent = `${unitCount} memories · ${entityCount} entities · ${graph.edges.length} links`;
        hopsNote.textContent =
          state.hops === 1
            ? "Direct links only."
            : "Includes memories two steps away, which carry visibly less weight.";
        renderNotice(graph);

        if (!graph.nodes.length) {
          emptyEl.innerHTML = graph.total_units
            ? "<p>No memories match these filters. Try including more entity kinds.</p>"
            : '<p>Nothing stored yet. Add a memory on the <a href="/app/store/">Store</a> page and it will appear here once it has been processed.</p>';
          show(emptyEl);
        } else {
          hide(emptyEl);
        }

        state.canvas.setGraph(graph);
        canvasEl.setAttribute(
          "aria-label",
          `Memory graph: ${unitCount} memories and ${entityCount} entities, ${graph.edges.length} links. ` +
            "Use the entity search to inspect a specific entity."
        );
      } catch (error) {
        if (token !== state.inFlight) return;
        showStatus(statusEl, describeError(error, performance.now() - started), true);
      }
    }

    /* -- search ----------------------------------------------------------- */

    // Entity search against the published snapshot. The live endpoint
    // takes an arbitrary `q`, which a fixed set of files obviously
    // cannot enumerate, so the exporter writes the whole entity list
    // once (ordered exactly as the API orders it: mention_count desc,
    // then display_name) and the match happens here.
    //
    // One deliberate difference, not a bug to be fixed silently: the
    // backend normalizes `q` through app/memory/entities.py's
    // normalize_name() before matching, which also strips possessives
    // and collapses punctuation. Reimplementing that in JS would create
    // a second copy of entity normalization -- the exact kind of
    // duplicate rule that drifts -- so this does a plain
    // case-insensitive substring match over both the display and
    // normalized names instead. For ordinary words the two agree; for
    // "Nadia's" the live backend still finds "Nadia" and this does not.
    let snapshotEntities = null;
    async function searchSnapshot(term) {
      if (snapshotEntities === null) {
        snapshotEntities = await getJSON("/api/graph/entities?limit=500");
      }
      const needle = term.toLowerCase();
      return snapshotEntities
        .filter(
          (e) =>
            (e.display_name || "").toLowerCase().includes(needle) ||
            (e.normalized_name || "").toLowerCase().includes(needle)
        )
        .slice(0, SEARCH_LIMIT);
    }

    searchForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const term = searchInput.value.trim();
      if (!term) return;
      searchResults.innerHTML = '<li class="muted">Searching…</li>';
      show(searchResults);
      try {
        const entities = isPublicMode()
          ? await searchSnapshot(term)
          : await getJSON(
              `/api/graph/entities?q=${encodeURIComponent(term)}&limit=${SEARCH_LIMIT}`
            );
        if (!entities.length) {
          searchResults.innerHTML = `<li class="muted">Nothing matching “${escapeHtml(term)}”.</li>`;
          return;
        }
        searchResults.innerHTML = entities
          .map(
            (e) =>
              `<li><button type="button" class="btn btn-ghost btn-sm" data-entity="${escapeHtml(e.id)}">` +
              `${escapeHtml(e.display_name)} <span class="muted">${escapeHtml(KIND_LABELS[e.kind] || e.kind)} · ${e.mention_count}</span>` +
              `</button></li>`
          )
          .join("");
      } catch (error) {
        searchResults.innerHTML = `<li class="error">${escapeHtml(describeError(error, 0))}</li>`;
      }
    });

    searchResults.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-entity]");
      if (!button) return;
      const entityId = button.getAttribute("data-entity");
      await renderEntity(entityId);
      const nodeId = `entity:${entityId}`;
      if (state.canvas && state.canvas.nodeById.has(nodeId)) {
        state.canvas.selectNode(nodeId, { center: true });
        return;
      }
      // Not in the current view: the entity's kind is probably filtered out.
      // Turn its kind back on rather than silently doing nothing.
      try {
        const detail = await getJSON(`/api/graph/entities/${entityId}`);
        if (detail && detail.entity && !state.entityKinds.has(detail.entity.kind)) {
          state.entityKinds.add(detail.entity.kind);
          buildFilters();
          await load();
          if (state.canvas.nodeById.has(nodeId)) state.canvas.selectNode(nodeId, { center: true });
        }
      } catch {
        /* the inspector already shows the entity; recentring is best-effort */
      }
    });

    /* -- controls --------------------------------------------------------- */

    hopsEl.addEventListener("change", () => {
      state.hops = Number(hopsEl.value) || 1;
      load();
    });

    backBtn.addEventListener("click", () => {
      state.centerUnitId = null;
      hide(backBtn);
      clearInspector();
      load();
    });

    fitBtn.addEventListener("click", () => {
      if (state.canvas) state.canvas.fitToView();
    });

    /* -- tooltip ---------------------------------------------------------- */

    function positionTooltip(text, point) {
      if (!text) {
        hide(tooltipEl);
        return;
      }
      const rect = stageEl.getBoundingClientRect();
      tooltipEl.textContent = text;
      tooltipEl.style.left = `${Math.min(rect.width - 16, Math.max(8, point.clientX - rect.left + 12))}px`;
      tooltipEl.style.top = `${Math.max(8, point.clientY - rect.top + 12)}px`;
      show(tooltipEl);
    }

    /* -- boot ------------------------------------------------------------- */

    clearInspector();

    loadRenderer()
      .then(() => {
        state.canvas = new window.membase.GraphCanvas(canvasEl, {
          onSelectNode: (node) => {
            if (node.node_type === "unit") renderUnit(node);
            else renderEntity(idPart(node.id));
          },
          onSelectEdge: (edge) => renderEdge(edge),
          onSelectNone: clearInspector,
          onHover: (target, point) => {
            if (!target) {
              positionTooltip(null);
              return;
            }
            positionTooltip(
              target.type === "node"
                ? target.node.label
                : target.edge.explanation || target.edge.label || "link",
              point
            );
          },
          onSettled: (metrics) => {
            // Logged, not displayed: the useful claim here ("the animation
            // loop really does stop") is verifiable from the console without
            // putting diagnostics in the user's way.
            console.info(
              `membase explorer: settled after ${metrics.steps} steps / ${metrics.frames} frames` +
                (metrics.settleMs === null ? "" : ` in ${Math.round(metrics.settleMs)}ms`) +
                ` (${metrics.nodes} nodes, ${metrics.edges} edges, running=${metrics.running})`
            );
          },
        });
        return load();
      })
      .catch((error) => {
        showStatus(statusEl, describeError(error, 0), true);
      });
  }

  window.membase.initExplorerPage = initExplorerPage;
})();
