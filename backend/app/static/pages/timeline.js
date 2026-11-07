// ===========================================================================
// Timeline page (/app/timeline/, nav key "timeline").
//
// Renders GET /api/episodes as a chronological rail, newest first, and
// GET /api/episodes/{id} as an episode's members when one is expanded.
//
// The one rule this file exists to not break: the backend goes to real
// lengths never to invent a date (see backend/app/memory/temporal.py and
// backend/app/memory/episodes.py), and the UI must not undo that.
// Concretely, three things are deliberate here:
//
//   1. Dates are NEVER formatted from the raw ISO value. Every visible
//      date is the server's pre-rendered `date_display`, which was
//      produced by temporal.py::describe at exactly the precision that is
//      actually known. The ISO `started_on`/`ended_on` fields are read for
//      ordering and for gap arithmetic only -- never printed. A
//      month-precision episode carries `started_on: "2024-02-01"`, and
//      printing that would claim a day the user never recorded.
//   2. An episode with no resolvable date (`started_on === null`) is shown
//      in its own labelled section at the end -- not hidden, and not given
//      a position on the rail, because any position would be invented.
//   3. `boundary_confidence` is rendered with the provenance vocabulary so
//      an inferred span cannot look like a stated one, and an episode whose
//      span is open (some member has no date at all) says so in words.
//
// Shared helpers come from static/core.js via window.membase; the component
// vocabulary (.card/.btn/.chip/.pill/.provenance/.confidence) comes from
// static/app.css. No dependency, no bundler, no type="module".
// ===========================================================================
(() => {
  "use strict";

  const {
    hide,
    show,
    showStatus,
    escapeHtml,
    fetchWithTimeout,
    describeError,
    openAttachment,
    buildAttachmentList,
  } = window.membase;

  // Mirrors backend/app/memory/temporal.py's PRECISION_ORDER, loosest
  // first, so two precisions can be compared the same way the backend
  // compares them.
  const PRECISION_ORDER = ["unknown", "year", "season", "month", "day"];

  // What each precision actually licenses saying. Shown next to a span so
  // "2021" is understood as "the year, and only the year" rather than as a
  // date that happens to be written short.
  const PRECISION_NOTE = {
    day: "exact day known",
    month: "month known, not the day",
    season: "season known, not the month",
    year: "year known, nothing finer",
  };

  // boundary_confidence -> how the span is allowed to look. The mapping is
  // the point of the page: `asserted` is the user's own claim, `inferred`
  // was read out of their words, `uncertain` means the episode may really
  // extend past what is recorded.
  const BOUNDARY = {
    asserted: {
      cls: "prov-original",
      label: "dates stated",
      title: "You wrote these dates yourself.",
    },
    inferred: {
      cls: "prov-extracted",
      label: "dates extracted",
      title: "Every memory here has a date, but the dates were read out of your notes rather than entered as dates.",
    },
    uncertain: {
      cls: "prov-inferred",
      label: "span open",
      title: "At least one memory in this episode has no date, so the episode may extend past the dates shown.",
    },
  };

  // An episode with no dated member at all is also `boundary_confidence:
  // "uncertain"`, but "span open" would misdescribe it: there is no span to
  // be open at one end. Its own label says what is actually true.
  const BOUNDARY_UNDATED = {
    cls: "prov-inferred",
    label: "no date recorded",
    title: "No memory in this episode has a resolvable date, so the episode has no position in time.",
  };

  // Membership confidence -> the shared three-segment meter. A category,
  // never a percentage (see app.css section 10).
  const MEMBER_CONFIDENCE = {
    certain: { cls: "conf-strong", label: "certain" },
    likely: { cls: "conf-moderate", label: "likely" },
    uncertain: { cls: "conf-weak", label: "uncertain" },
  };

  // How a memory came to be in the episode.
  const MEMBER_ORIGIN = {
    explicit: { cls: "prov-original", label: "grouped by you" },
    extracted: { cls: "prov-extracted", label: "grouped from your note" },
    inferred: { cls: "prov-inferred", label: "grouped by inference" },
  };

  const DAY_MS = 86_400_000;
  // Below this, two episodes are near enough that a labelled gap would be
  // noise; the connector still scales, so density is visible either way.
  const GAP_LABEL_MIN_DAYS = 40;

  function looserPrecision(a, b) {
    const ia = Math.max(0, PRECISION_ORDER.indexOf(a));
    const ib = Math.max(0, PRECISION_ORDER.indexOf(b));
    return PRECISION_ORDER[Math.min(ia, ib)];
  }

  // Parses an ISO date for arithmetic and ordering ONLY. Nothing derived
  // from this is ever displayed as a date.
  function isoParts(iso) {
    if (typeof iso !== "string") return null;
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
    if (!m) return null;
    return {
      year: Number(m[1]),
      month: Number(m[2]),
      time: Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])),
    };
  }

  // The time between two episodes, expressed at the coarser of their two
  // precisions. Both anchors are *representative* dates (temporal.py picks
  // the first of the month / mid-season / first of the year so the value is
  // sortable), so reporting "47 days" between two year-precision episodes
  // would be arithmetic on something that was never claimed. The unit
  // therefore degrades with the precision.
  function describeGap(newer, older) {
    const a = isoParts(newer.started_on);
    const b = isoParts(older.ended_on || older.started_on);
    if (!a || !b) return null;
    const days = Math.round((a.time - b.time) / DAY_MS);
    if (days <= 0) return { days: 0, label: "", overlaps: true };

    const precision = looserPrecision(newer.date_precision, older.date_precision);
    let label = "";
    if (precision === "year") {
      const years = a.year - b.year;
      label = years <= 0 ? "same year" : `about ${years} year${years === 1 ? "" : "s"} earlier`;
    } else if (precision === "month" || precision === "season") {
      const months = (a.year - b.year) * 12 + (a.month - b.month);
      if (months <= 0) label = "same month";
      else if (months < 12) label = `about ${months} month${months === 1 ? "" : "s"} earlier`;
      else {
        const years = Math.floor(months / 12);
        const rest = months % 12;
        label =
          `about ${years} year${years === 1 ? "" : "s"}` +
          (rest ? ` ${rest} month${rest === 1 ? "" : "s"}` : "") +
          " earlier";
      }
    } else if (days < 60) {
      label = `${days} day${days === 1 ? "" : "s"} earlier`;
    } else {
      const months = Math.round(days / 30);
      label = months < 12 ? `about ${months} months earlier` : `about ${Math.round(days / 365)} years earlier`;
    }
    return {
      days,
      label,
      overlaps: false,
      title: `Computed from each span's representative anchor date; the coarser of the two spans is known to the ${precision}.`,
    };
  }

  // Connector height, scaled logarithmically so a week and a decade are
  // visibly different without a decade pushing the next card off-screen.
  function gapHeight(days) {
    if (!days || days <= 0) return 14;
    return Math.round(Math.min(104, 18 + 24 * Math.log2(1 + days / 7)));
  }

  function provenanceTag(spec) {
    if (!spec) return "";
    return `<span class="provenance ${spec.cls}" title="${escapeHtml(spec.title || spec.label)}">${escapeHtml(spec.label)}</span>`;
  }

  function episodeMetaHtml(episode) {
    const boundary =
      episode.started_on === null ? BOUNDARY_UNDATED : BOUNDARY[episode.boundary_confidence];
    const parts = [provenanceTag(boundary)];
    const noun = episode.member_count === 1 ? "memory" : "memories";
    parts.push(`<span class="pill">${episode.member_count} ${noun}</span>`);
    const note = PRECISION_NOTE[episode.date_precision];
    if (note) parts.push(`<span class="pill">${escapeHtml(note)}</span>`);
    return parts.filter(Boolean).join(" ");
  }

  function anchorsHtml(episode) {
    if (!episode.anchors || !episode.anchors.length) return "";
    const chips = episode.anchors
      .map(
        (name) =>
          `<button type="button" class="chip tl-anchor" data-anchor="${escapeHtml(name)}">${escapeHtml(name)}</button>`
      )
      .join("");
    return `<div class="row row-tight tl-anchors" aria-label="What this episode is about">${chips}</div>`;
  }

  // ---------------------------------------------------------------------
  // Episode detail (members, their own dates, why each is here, photos)

  // Object URLs for inline images, kept for the life of the page so
  // collapsing and re-expanding an episode does not re-download a photo.
  // The bytes have to come through fetchWithTimeout -- it is the only thing
  // that attaches the Authorization header this backend needs, which a
  // plain <img src> cannot carry.
  const imageUrlCache = new Map();

  async function loadInlineImage(attachment, imgEl, captionEl) {
    try {
      let url = imageUrlCache.get(attachment.id);
      if (!url) {
        const resp = await fetchWithTimeout(`/api/attachments/${attachment.id}/file`);
        if (!resp.ok) throw new Error(`Failed to load image (${resp.status})`);
        url = URL.createObjectURL(await resp.blob());
        imageUrlCache.set(attachment.id, url);
      }
      imgEl.src = url;
      imgEl.classList.remove("hidden");
    } catch (err) {
      captionEl.classList.add("error");
      captionEl.textContent = `${attachment.original_filename} — ${err.message}`;
    }
  }

  // Images shown inline; anything else gets the shared attachment list
  // (view/download through core.js). `seen` is shared across the whole
  // episode: two members from the same submission carry the same capture
  // attachments, and showing one photograph twice reads as a bug rather
  // than as the shared note it actually is.
  function buildMemberAttachments(member, seen) {
    const wrap = document.createElement("div");
    const attachments = member.attachments || [];
    if (!attachments.length) return wrap;

    const fresh = attachments.filter((a) => !seen.has(a.id));
    const repeated = attachments.filter((a) => seen.has(a.id));
    attachments.forEach((a) => seen.add(a.id));

    const images = fresh.filter((a) => (a.content_type || "").startsWith("image/"));
    const others = fresh.filter((a) => !(a.content_type || "").startsWith("image/"));

    if (images.length) {
      const gallery = document.createElement("div");
      gallery.className = "tl-gallery";
      for (const attachment of images) {
        const figure = document.createElement("figure");
        figure.className = "tl-photo";
        const button = document.createElement("button");
        button.type = "button";
        button.className = "tl-photo-button";
        button.title = `Open ${attachment.original_filename} full size`;
        const img = document.createElement("img");
        img.className = "hidden";
        // Deliberately NOT loading="lazy": the bytes are already in hand by
        // the time `src` is set (it is a blob: URL), so lazy only buys a
        // blank box for any photograph below the fold.
        img.alt = `Photograph attached to this memory: ${attachment.original_filename}`;
        button.appendChild(img);
        const caption = document.createElement("figcaption");
        caption.className = "muted";
        caption.textContent = attachment.original_filename;
        button.addEventListener("click", () => openAttachment(attachment, null));
        figure.appendChild(button);
        figure.appendChild(caption);
        gallery.appendChild(figure);
        loadInlineImage(attachment, img, caption);
      }
      wrap.appendChild(gallery);
    }

    if (others.length) {
      const line = document.createElement("p");
      line.className = "muted";
      line.appendChild(buildAttachmentList(others));
      wrap.appendChild(line);
    }

    if (repeated.length && !fresh.length) {
      const line = document.createElement("p");
      line.className = "muted";
      line.textContent =
        repeated.length === 1
          ? `Came with the same note as a memory above (${repeated[0].original_filename}).`
          : "Came with the same note as a memory above.";
      wrap.appendChild(line);
    }
    return wrap;
  }

  function renderDetail(container, detail, onNavigate) {
    const episode = detail.episode;
    container.innerHTML = "";

    const stack = document.createElement("div");
    stack.className = "stack tl-detail-body";

    if (episode.title_source === "derived") {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent =
        "Title composed from this episode's most frequent people and places plus its date span — a label, not a written summary.";
      stack.appendChild(note);
    }

    const membersWrap = document.createElement("ol");
    membersWrap.className = "tl-members";
    const seenAttachments = new Set();

    for (const member of detail.members) {
      const li = document.createElement("li");
      li.className = "card card-quiet tl-member";
      const confidence = MEMBER_CONFIDENCE[member.confidence] || {
        cls: "conf-insufficient",
        label: member.confidence || "unknown",
      };
      const origin = MEMBER_ORIGIN[member.origin];
      const undated = member.occurred_on === null || member.date_precision === "unknown";
      const precisionNote = PRECISION_NOTE[member.date_precision];
      li.innerHTML = `
        <div class="row row-between tl-member-head">
          <span class="tl-member-when${undated ? " tl-member-when-unknown" : ""}">${escapeHtml(member.date_display)}</span>
          <span class="row row-tight">
            ${provenanceTag(origin)}
            <span class="confidence ${confidence.cls}">${escapeHtml(confidence.label)}</span>
          </span>
        </div>
        <p class="tl-member-content">${escapeHtml(member.content)}</p>
        ${precisionNote && !undated ? `<p class="muted">${escapeHtml(precisionNote)}</p>` : ""}
        ${member.explanation ? `<p class="muted tl-member-why">Why it's here: ${escapeHtml(member.explanation)}</p>` : ""}
        <div class="tl-member-files"></div>
      `;
      li.querySelector(".tl-member-files").appendChild(buildMemberAttachments(member, seenAttachments));
      membersWrap.appendChild(li);
    }

    if (!detail.members.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No searchable memories left in this episode.";
      stack.appendChild(empty);
    } else {
      stack.appendChild(membersWrap);
    }

    // Related episodes. Deliberately empty for an undated episode: an
    // unknown span cannot be said to overlap anything, and the UI says that
    // rather than showing an empty box.
    const related = document.createElement("div");
    related.className = "stack stack-tight tl-related";
    const relatedHead = document.createElement("h3");
    relatedHead.textContent = "Related episodes";
    related.appendChild(relatedHead);
    if (detail.overlapping && detail.overlapping.length) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "Episodes whose span overlaps this one's.";
      related.appendChild(note);
      const list = document.createElement("div");
      list.className = "row tl-related-list";
      for (const other of detail.overlapping) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn btn-secondary btn-sm tl-related-item";
        button.textContent = `${other.title} · ${other.date_display}`;
        button.addEventListener("click", () => onNavigate(other.id));
        list.appendChild(button);
      }
      related.appendChild(list);
    } else if (episode.started_on === null) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent =
        "None — this episode has no known span, and an unknown span can't be said to overlap anything.";
      related.appendChild(note);
    } else {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "None — no other episode's span overlaps this one's.";
      related.appendChild(note);
    }
    stack.appendChild(related);

    container.appendChild(stack);
  }

  // ---------------------------------------------------------------------

  function initTimelinePage() {
    const root = document.getElementById("timeline-root");
    if (!root) return;
    const statusEl = document.getElementById("status");
    const countsEl = document.getElementById("tl-counts");
    const controls = document.getElementById("tl-controls");
    const searchEl = document.getElementById("tl-search");
    const undatedToggle = document.getElementById("tl-undated-toggle");
    const pageSizeEl = document.getElementById("tl-page-size");
    const pager = document.getElementById("tl-pager");
    const newerButton = document.getElementById("tl-newer");
    const olderButton = document.getElementById("tl-older");
    const pageLabel = document.getElementById("tl-page-label");

    // Per-page closure state, same discipline as pages/library.js: nothing
    // about this page is attached to window.
    const state = {
      limit: Number(pageSizeEl.value) || 25,
      offset: 0,
      total: 0,
      undatedCount: 0,
      includeUndated: true,
      query: "",
      episodes: [],
      loaded: false,
      expanded: new Set(),
      details: new Map(),
    };

    function matchesQuery(episode) {
      if (!state.query) return true;
      const haystack = [episode.title, episode.summary, episode.date_display, ...(episode.anchors || [])]
        .join(" ")
        .toLowerCase();
      return haystack.includes(state.query);
    }

    function node(markerClass, bodyHtml, extraClass) {
      const el = document.createElement("div");
      el.className = `tl-node${extraClass ? " " + extraClass : ""}`;
      el.innerHTML = `<div class="tl-marker ${markerClass}" aria-hidden="true"></div><div class="tl-body">${bodyHtml}</div>`;
      return el;
    }

    function buildEpisodeNode(episode) {
      const undated = episode.started_on === null;
      const el = node(undated ? "tl-marker-undated" : "tl-marker-dated", "", "tl-node-episode");
      const body = el.querySelector(".tl-body");
      const article = document.createElement("article");
      article.className = "card tl-episode";
      article.dataset.episodeId = episode.id;
      const openNote =
        episode.boundary_confidence === "uncertain" && !undated
          ? `<p class="muted tl-open-note">At least one memory here has no date, so this episode may extend past ${escapeHtml(episode.date_display)}.</p>`
          : "";
      const undatedNote = undated
        ? `<p class="muted tl-open-note">No memory in this episode has a resolvable date, so it has no place on the rail above.</p>`
        : "";
      article.innerHTML = `
        <button type="button" class="tl-head" aria-expanded="false">
          <span class="tl-head-main">
            <span class="tl-title">${escapeHtml(episode.title)}</span>
            <span class="tl-when${undated ? " tl-when-unknown" : ""}">${escapeHtml(episode.date_display)}</span>
          </span>
          <span class="row row-tight tl-head-meta">${episodeMetaHtml(episode)}</span>
          <span class="tl-caret" aria-hidden="true"></span>
        </button>
        <p class="tl-summary secondary">${escapeHtml(episode.summary)}</p>
        ${openNote}${undatedNote}
        ${anchorsHtml(episode)}
        <div class="tl-detail hidden"></div>
      `;

      const toggle = article.querySelector(".tl-head");
      const detailEl = article.querySelector(".tl-detail");

      for (const chip of article.querySelectorAll(".tl-anchor")) {
        chip.addEventListener("click", () => {
          searchEl.value = chip.dataset.anchor;
          state.query = chip.dataset.anchor.toLowerCase();
          render();
        });
      }

      async function expand() {
        state.expanded.add(episode.id);
        toggle.setAttribute("aria-expanded", "true");
        show(detailEl);
        if (state.details.has(episode.id)) {
          renderDetail(detailEl, state.details.get(episode.id), navigateTo);
          return;
        }
        detailEl.innerHTML = '<p class="muted"><span class="spinner" aria-hidden="true"></span>Loading this episode’s memories...</p>';
        const startedAt = performance.now();
        try {
          const resp = await fetchWithTimeout(`/api/episodes/${episode.id}`);
          const data = await resp.json().catch(() => ({}));
          if (!resp.ok) {
            const err = new Error(data.detail || `Failed to load episode (${resp.status})`);
            err.apiStatus = resp.status;
            throw err;
          }
          state.details.set(episode.id, data);
          renderDetail(detailEl, data, navigateTo);
        } catch (err) {
          detailEl.innerHTML = "";
          const p = document.createElement("p");
          p.className = "status error";
          p.textContent = describeError(err, performance.now() - startedAt);
          detailEl.appendChild(p);
        }
      }

      function collapse() {
        state.expanded.delete(episode.id);
        toggle.setAttribute("aria-expanded", "false");
        hide(detailEl);
      }

      toggle.addEventListener("click", () => {
        if (state.expanded.has(episode.id)) collapse();
        else expand();
      });

      body.appendChild(article);
      if (state.expanded.has(episode.id)) expand();
      return el;
    }

    // "Related episodes" navigation: the overlapping episode is almost
    // always on the page already (the rail is paginated, not virtualised),
    // so jump to it and open it. If the current filter or page is hiding
    // it, say so instead of silently doing nothing.
    function navigateTo(episodeId) {
      state.expanded.add(episodeId);
      const existing = root.querySelector(`[data-episode-id="${CSS.escape(episodeId)}"]`);
      if (!existing) {
        // Clearing the filter is enough whenever the episode is on this
        // page at all; if it is on another page, the counts line below
        // still tells the user where they are.
        searchEl.value = "";
        state.query = "";
        render();
      }
      const target = root.querySelector(`[data-episode-id="${CSS.escape(episodeId)}"]`);
      if (!target) return;
      const toggle = target.querySelector(".tl-head");
      if (toggle && toggle.getAttribute("aria-expanded") !== "true") toggle.click();
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      target.classList.add("tl-episode-flash");
      setTimeout(() => target.classList.remove("tl-episode-flash"), 1200);
    }

    function renderCounts(shown, datedShown) {
      const pieces = [];
      if (state.query) {
        // Only the match count while filtering: mentioning the store-wide
        // undated total here reads as "and N of these matches are undated",
        // which is a different claim.
        const undatedShown = shown - datedShown;
        pieces.push(`${shown} of ${state.episodes.length} loaded episode${state.episodes.length === 1 ? "" : "s"} match this filter`);
        if (undatedShown > 0) pieces.push(`${undatedShown} of them undated`);
        countsEl.textContent = pieces.join(" · ");
        show(countsEl);
        return;
      }
      const first = state.total === 0 ? 0 : state.offset + 1;
      const last = state.offset + state.episodes.length;
      pieces.push(`Showing ${first}–${last} of ${state.total} episode${state.total === 1 ? "" : "s"}`);
      if (state.undatedCount > 0) {
        pieces.push(
          state.includeUndated
            ? `${state.undatedCount} episode${state.undatedCount === 1 ? "" : "s"} whose date isn’t known`
            : `${state.undatedCount} episode${state.undatedCount === 1 ? "" : "s"} whose date isn’t known (hidden)`
        );
      }
      if (datedShown === 0 && shown > 0) pieces.push("none of them placeable in time");
      countsEl.textContent = pieces.join(" · ");
      show(countsEl);
    }

    function renderPager() {
      const pages = Math.max(1, Math.ceil(state.total / state.limit));
      const page = Math.floor(state.offset / state.limit) + 1;
      if (pages <= 1) {
        hide(pager);
        return;
      }
      show(pager);
      pageLabel.textContent = `Page ${page} of ${pages}`;
      newerButton.disabled = state.offset === 0;
      olderButton.disabled = state.offset + state.limit >= state.total;
    }

    function render() {
      root.innerHTML = "";

      const visible = state.episodes.filter(matchesQuery);
      const dated = visible.filter((e) => e.started_on !== null);
      const undated = visible.filter((e) => e.started_on === null);

      renderCounts(visible.length, dated.length);
      renderPager();

      if (!state.episodes.length) {
        // Honest empty state: nothing stored (or nothing processed into an
        // episode yet) is a different statement from "loading" and from
        // "the request failed", both of which live in #status.
        const empty = document.createElement("div");
        empty.className = "card tl-empty";
        empty.innerHTML = `
          <p class="eyebrow">Nothing on the timeline yet</p>
          <p>Episodes are derived from the memories you store — grouped by what they
          are about and when they happened, with no manual curation step. None
          exist yet, which means nothing has been stored and finished processing.</p>
          <p><a class="btn btn-primary" href="/app/store/">Store your first memory</a></p>
        `;
        root.appendChild(empty);
        return;
      }

      if (!visible.length) {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = `No loaded episode matches “${state.query}”.`;
        root.appendChild(empty);
        return;
      }

      let previous = null;
      let previousYear = null;
      for (const episode of dated) {
        const parts = isoParts(episode.started_on);
        const year = parts ? parts.year : null;

        if (previous) {
          const gap = describeGap(previous, episode);
          const height = gapHeight(gap ? gap.days : 0);
          const labelHtml =
            gap && gap.label && gap.days >= GAP_LABEL_MIN_DAYS
              ? `<p class="muted tl-gap-label" title="${escapeHtml(gap.title || "")}">${escapeHtml(gap.label)}</p>`
              : "";
          const gapNode = node("tl-marker-gap", labelHtml, "tl-node-gap");
          gapNode.style.setProperty("--tl-gap-size", `${height}px`);
          root.appendChild(gapNode);
        }

        // Year markers plus the scaled connector are the position cue: the
        // rail is not evenly spaced, so adjacency on screen means adjacency
        // in time.
        if (year !== null && year !== previousYear) {
          root.appendChild(node("tl-marker-year", `<p class="tl-year">${escapeHtml(String(year))}</p>`, "tl-node-year"));
          previousYear = year;
        }

        root.appendChild(buildEpisodeNode(episode));
        previous = episode;
      }

      if (undated.length) {
        const head = node(
          "tl-marker-break",
          `<div class="tl-section-head">
             <h2>Date not known</h2>
             <p class="muted">${undated.length} episode${undated.length === 1 ? "" : "s"} with no resolvable date, listed here rather than placed above — putting them on the rail would mean choosing a date for them.</p>
           </div>`,
          "tl-node-break"
        );
        root.appendChild(head);
        for (const episode of undated) root.appendChild(buildEpisodeNode(episode));
      }
    }

    async function load() {
      const startedAt = performance.now();
      showStatus(statusEl, "Loading the timeline...");
      try {
        const params = new URLSearchParams({
          limit: String(state.limit),
          offset: String(state.offset),
          include_undated: String(state.includeUndated),
        });
        const resp = await fetchWithTimeout(`/api/episodes?${params}`);
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          const err = new Error(data.detail || `Failed to load the timeline (${resp.status})`);
          err.apiStatus = resp.status;
          throw err;
        }
        state.episodes = Array.isArray(data.episodes) ? data.episodes : [];
        state.total = Number(data.total) || 0;
        state.undatedCount = Number(data.undated_count) || 0;
        state.loaded = true;
        // An offset left past the end (page size raised, or undated
        // episodes excluded) would render a blank rail that looks like an
        // empty store.
        if (state.offset > 0 && state.offset >= state.total) {
          state.offset = Math.max(0, (Math.ceil(state.total / state.limit) - 1) * state.limit);
          return load();
        }
        hide(statusEl);
        render();
      } catch (err) {
        root.innerHTML = "";
        hide(countsEl);
        hide(pager);
        showStatus(statusEl, describeError(err, performance.now() - startedAt), true);
      }
    }

    controls.addEventListener("submit", (ev) => ev.preventDefault());

    searchEl.addEventListener("input", () => {
      state.query = searchEl.value.trim().toLowerCase();
      if (state.loaded) render();
    });

    undatedToggle.addEventListener("click", () => {
      state.includeUndated = !state.includeUndated;
      undatedToggle.setAttribute("aria-pressed", String(state.includeUndated));
      state.offset = 0;
      load();
    });

    pageSizeEl.addEventListener("change", () => {
      state.limit = Number(pageSizeEl.value) || 25;
      state.offset = 0;
      load();
    });

    newerButton.addEventListener("click", () => {
      state.offset = Math.max(0, state.offset - state.limit);
      load();
    });

    olderButton.addEventListener("click", () => {
      if (state.offset + state.limit >= state.total) return;
      state.offset += state.limit;
      load();
    });

    load();
  }

  window.membase.initTimelinePage = initTimelinePage;
})();
