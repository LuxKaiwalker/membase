// Evaluation page (/app/evaluation/, nav key "evaluation") -- the
// multi-strategy retrieval benchmark's results.
//
// The one page under /app/ with no API behind it. The backend is API-only
// and exposes no evaluation endpoint: these numbers are produced offline by
// eval/strategy_benchmark.py, which seeds a 44-memory corpus through the
// real ingestion pipeline into an isolated `membase_eval` database (15-25
// minutes before a single query runs) and writes
// eval/strategy_benchmark_results.json. scripts/build_frontend.py copies
// that artefact to static/evaluation-results.json when it exists, so this
// file is a renderer for a published measurement, not a client for a
// service. When the artefact is absent the page says so and gives the
// command, rather than looking broken.
//
// Every metric definition and every caveat rendered here is taken from
// docs/EVALUATION.md ("Metrics, exactly" and "Caveats"). That document is
// the source of truth; this file must not invent its own definitions, and
// must not reduce the metrics to one number -- the trade-offs between them
// are the finding, and a composite score is exactly what would hide the
// one thing this benchmark exists to show.
//
// Defensive about the payload on purpose: the runner is still being
// iterated on, so every field is read through a helper that renders "n/a"
// (or omits the row) rather than throwing or printing `undefined`.
(() => {
  "use strict";

  const { escapeHtml } = window.membase;

  // Same-origin static artefact, deliberately NOT fetched through
  // membase.fetchWithTimeout: that helper prefixes the API base URL
  // (https://membase-api.vxliu.com) and attaches the Basic Auth header,
  // both of which are wrong here. This file sits next to the page on the
  // static site, so a plain relative fetch is the correct call -- and it
  // must keep working with no credentials stored at all.
  const RESULTS_URL = "/static/evaluation-results.json";

  const BENCHMARK_COMMAND = "scripts/run-strategy-benchmark.sh";

  // Canonical order, matching STRATEGIES in eval/strategy_benchmark.py.
  // Only used as a fallback / sort key -- the payload's own meta.strategies
  // wins, so a strategy added to the runner still shows up here.
  const STRATEGY_ORDER = ["vector", "semantic_rerank", "associative", "adaptive"];

  const STRATEGY_NOTES = {
    vector:
      "Pure pgvector cosine top-k at the production threshold, then one answer call. " +
      "No full-text, no reformulation, no graph. The conventional baseline.",
    semantic_rerank:
      "The existing one-shot agent: hybrid vector+full-text fusion plus " +
      "sufficiency-driven query reformulation. No graph, no clarification.",
    associative:
      "Association-graph expansion and structured evidence collection. " +
      "Multi-hop, but single-shot — it never asks the user anything.",
    adaptive:
      "The full iterative RecallSession, driven by a simulated user that " +
      "answers whatever clarifying question the session actually asks.",
  };

  // Canonical order, matching CATEGORIES in eval/strategy_benchmark.py.
  const CATEGORY_ORDER = [
    "exact",
    "semantic",
    "indirect_association",
    "multi_hop",
    "temporal",
    "ambiguous",
    "vague_episodic",
    "unanswerable",
  ];

  const CATEGORY_NOTES = {
    exact: "the query uses the memory's own words or proper nouns",
    semantic: "paraphrase, with little or no lexical overlap",
    indirect_association: "the fact lives in a memory linked to the one the query matches",
    multi_hop: "two steps: memory A names the entity, memory B holds the answer",
    temporal: "date- and ordering-sensitive, including season-only dates",
    ambiguous: "several stored memories stay plausible; a clarifying question resolves it",
    vague_episodic: "a feeling or a fragment rather than a fact",
    unanswerable: "nothing stored answers it; the correct behaviour is to abstain",
  };

  // dir: "up"   = higher is better
  //      "down" = LOWER is better (rendered as such everywhere, never mixed
  //               into the same visual group as an up-metric)
  //      "cost" = neither better nor worse, just what it cost
  const METRICS = {
    recall_at_1: { label: "recall@1", kind: "pct", dir: "up" },
    recall_at_5: { label: "recall@5", kind: "pct", dir: "up" },
    correctness: { label: "answer correctness", kind: "pct", dir: "up" },
    false_retrieval_rate: { label: "false retrieval rate", kind: "pct", dir: "down" },
    abstention_rate_answerable: { label: "abstention (answerable)", kind: "pct", dir: "down" },
    mean_questions: { label: "mean clarifying Qs", kind: "num", dir: "cost" },
    mean_iterations: { label: "mean iterations", kind: "num", dir: "cost" },
    mean_llm_calls: { label: "mean LLM calls", kind: "num", dir: "cost" },
    mean_llm_seconds: { label: "mean LLM seconds", kind: "num", dir: "cost" },
    mean_wall_seconds: { label: "mean wall seconds", kind: "num", dir: "cost" },
  };

  // Column order of the comparison table. recall and false retrieval sit
  // next to each other deliberately -- see renderComparison().
  const TABLE_COLUMNS = [
    "recall_at_1",
    "recall_at_5",
    "correctness",
    "false_retrieval_rate",
    "abstention_rate_answerable",
    "mean_questions",
    "mean_iterations",
    "mean_llm_calls",
    "mean_llm_seconds",
    "mean_wall_seconds",
  ];

  const METRIC_DEFINITIONS = [
    ["recall@1 / recall@5", "over answerable queries: the fraction where the declared relevant memories intersect the strategy's own top-1 / top-5 ranked sources."],
    ["answer correctness", "over answerable queries: the answer is grounded and contains at least one expected substring. An abstention is never correct on an answerable query."],
    ["false retrieval rate", "over unanswerable queries: the fraction answered instead of abstained. Lower is better. This is the only metric that stops “answer everything” from looking like a win."],
    ["abstention (answerable)", "over answerable queries: the fraction where the strategy abstained. The mirror image of false retrieval; together they say whether a strategy's grounding decision is calibrated or just biased."],
    ["mean clarifying Qs", "over all queries. Structurally zero for the first three strategies — only the iterative one can ask."],
    ["mean iterations", "over all queries. The session's own round count for the iterative strategy, 1 by construction for the vector baseline, and inferred from the LLM call log for the other two — inferred, not measured."],
    ["mean LLM calls / seconds / wall seconds", "over all queries, from a timing decorator around the real provider. LLM seconds is summed per-call latency; wall seconds includes the SQL and Python work too."],
  ];

  const CAVEATS = [
    ["The corpus was authored by the same system being evaluated.", "Every memory, every query and every expected substring was written by an LLM while building this benchmark. The ambiguous pairs, multi-hop chains and lexical distractors are the hard cases this system's author could imagine, not a sample of the hard cases a real person's notes contain. This is the single largest caveat and running the benchmark more times does not fix it."],
    ["Correctness is a substring proxy, not a judge.", "It marks a correct answer wrong for using a synonym the expected list does not contain, and a wrong answer right for naming the expected token in passing. An LLM judge was rejected because it would add its own unmeasured error to every number here — the trade is that individual verdicts should be read from the answer text, not trusted blindly."],
    ["The simulated user is an upper bound on a real user's helpfulness.", "It answers from the ground-truth memory's real post-ingestion semantic fields, so it is always consistent, never wrong, never vague and never misremembers. A real person struggling to recall a memory may well not know. Whatever the iterative strategy gains from clarification here is the ceiling, not the expected value."],
    ["Clarification at ingestion time was suppressed.", "The corpus was seeded with clarification off, so no unit was parked awaiting an answer and silently removed from every strategy's reach. That keeps it a retrieval benchmark, but it is a real deviation from production ingestion."],
    ["The vector baseline is not production code.", "It is assembled inside the benchmark; production has no reason to offer a non-fusing, non-iterating search path."],
    ["Recall is measured against a hand-declared relevant set that may not be exhaustive.", "A strategy can retrieve a genuinely reasonable memory nobody listed and be scored as a miss."],
    ["recall@k is a lower bound for the iterative strategy on some queries.", "A session still asking questions after the simulated turn cap snapshotted no candidate list, so those queries report an empty retrieved set even though the session did retrieve each round. Re-running retrieval to fill that in would measure something other than what the session actually delivered to its user, which was nothing."],
    ["The seconds are not portable.", "One machine, shared with everything else, including whatever the model server's prompt cache was doing. The ratios between strategies are the finding; the absolute seconds are not."],
    ["A small store.", "Graph traversal, the relevance threshold's behaviour and the fused score's absolute scale all depend on corpus size. Nothing here predicts behaviour at ten thousand memories."],
  ];

  // -------------------------------------------------------------------------
  // Value helpers. Everything the payload carries is treated as possibly
  // missing or null: `recall_1` is null on an unanswerable query,
  // `false_retrieval` is null on an answerable one, and a whole rate is
  // null when its denominator was empty.
  // -------------------------------------------------------------------------

  function isNum(v) {
    return typeof v === "number" && Number.isFinite(v);
  }

  function pct(v, dp) {
    return isNum(v) ? `${(v * 100).toFixed(dp === undefined ? 1 : dp)}%` : "n/a";
  }

  function fixed(v, dp) {
    return isNum(v) ? v.toFixed(dp === undefined ? 2 : dp) : "n/a";
  }

  function formatMetric(key, value) {
    return (METRICS[key] || {}).kind === "num" ? fixed(value) : pct(value);
  }

  function plural(n, one, many) {
    return `${n} ${n === 1 ? one : many}`;
  }

  // Mean / min / max of one metric for one strategy across the repeats,
  // ignoring runs where it was null. Mirrors print_summary's agg() in
  // eval/strategy_benchmark.py so the page and the terminal report cannot
  // disagree about what a number means.
  function across(summaries, strategy, key) {
    const values = summaries
      .map((s) => (s && s[strategy] ? s[strategy][key] : null))
      .filter(isNum);
    if (!values.length) return null;
    const total = values.reduce((a, b) => a + b, 0);
    return {
      mean: total / values.length,
      lo: Math.min.apply(null, values),
      hi: Math.max.apply(null, values),
      runs: values.length,
    };
  }

  function categoryAcross(perCategory, category, strategy, key) {
    const values = perCategory
      .map((run) => {
        const cell = run && run[category] ? run[category][strategy] : null;
        return cell ? cell[key] : null;
      })
      .filter(isNum);
    if (!values.length) return null;
    const total = values.reduce((a, b) => a + b, 0);
    return { mean: total / values.length, lo: Math.min.apply(null, values), hi: Math.max.apply(null, values) };
  }

  // Same definition as succeeded() in the runner: a correct grounded answer
  // on an answerable query, a correct abstention on an unanswerable one.
  // Recomputed rather than trusted only when the payload omits the flag,
  // so a runner change that drops the field degrades instead of breaking.
  function didSucceed(row) {
    if (typeof row.succeeded === "boolean") return row.succeeded;
    return row.answerable ? !!row.correct : !!row.abstained;
  }

  function orderedBy(canonical, present) {
    const known = canonical.filter((k) => present.indexOf(k) !== -1);
    const extra = present.filter((k) => canonical.indexOf(k) === -1).sort();
    return known.concat(extra);
  }

  function resolveStrategies(payload) {
    const meta = payload.meta || {};
    if (Array.isArray(meta.strategies) && meta.strategies.length) return meta.strategies.slice();
    const first = (payload.summary_per_run || [])[0];
    if (first && typeof first === "object") return orderedBy(STRATEGY_ORDER, Object.keys(first));
    return STRATEGY_ORDER.slice();
  }

  function strategyLabel(name) {
    return `<span class="mono ev-strategy-name">${escapeHtml(name)}</span>`;
  }

  // -------------------------------------------------------------------------
  // Bars. Plain CSS: the fill's width is an inline percentage, nothing else.
  // No charting library, no SVG, no CDN -- this repo's whole frontend build
  // is `pip install jinja2`.
  // -------------------------------------------------------------------------

  // `stat` is an across() result. When there is more than one repeat the
  // observed min-max is drawn as a band on the same track, because that
  // spread is the noise floor a between-strategy gap has to clear.
  function bar(label, stat, kind, dir) {
    const value = stat ? stat.mean : null;
    const text = kind === "num" ? fixed(value) : pct(value);
    const width = isNum(value) ? Math.max(0, Math.min(100, value * 100)) : 0;
    const showRange = stat && stat.runs > 1 && stat.hi > stat.lo;
    const band = showRange
      ? `<span class="ev-bar-range" style="left: ${(stat.lo * 100).toFixed(2)}%; width: ${((stat.hi - stat.lo) * 100).toFixed(2)}%"></span>`
      : "";
    const rangeText = showRange ? `<span class="ev-bar-spread">${pct(stat.lo, 0)}–${pct(stat.hi, 0)}</span>` : "";
    return `
      <div class="ev-bar">
        <span class="ev-bar-label mono">${escapeHtml(label)}</span>
        <span class="ev-bar-track" role="img" aria-label="${escapeHtml(`${label}: ${text}`)}">
          <span class="ev-bar-fill ev-bar-fill-${dir === "down" ? "down" : "up"}" style="width: ${width.toFixed(2)}%"></span>
          ${band}
        </span>
        <span class="ev-bar-value">${escapeHtml(text)}${rangeText}</span>
      </div>`;
  }

  function barGroup(title, note, metricKeys, strategies, summaries) {
    const groups = metricKeys
      .map((key) => {
        const meta = METRICS[key] || { label: key, kind: "pct", dir: "up" };
        const bars = strategies
          .map((s) => bar(s, across(summaries, s, key), meta.kind, meta.dir))
          .join("");
        return `
          <div class="ev-bar-metric">
            <h4 class="ev-bar-metric-title">${escapeHtml(meta.label)}</h4>
            <div class="ev-bars">${bars}</div>
          </div>`;
      })
      .join("");
    return `
      <div class="card card-quiet ev-bar-panel">
        <p class="eyebrow">${escapeHtml(title)}</p>
        <p class="muted ev-bar-note">${note}</p>
        ${groups}
      </div>`;
  }

  // -------------------------------------------------------------------------
  // Sections
  // -------------------------------------------------------------------------

  function renderProvenance(payload) {
    const meta = payload.meta || {};
    const graph = meta.graph || {};
    const summaries = payload.summary_per_run || [];
    const repeats = isNum(meta.repeat) ? meta.repeat : summaries.length || null;
    const queriesRun = Array.isArray(meta.queries_run) ? meta.queries_run.length : null;
    const skipped = Array.isArray(meta.queries_skipped) ? meta.queries_skipped : [];
    const firstSummary = summaries[0] || {};
    const anyStrategy = Object.keys(firstSummary)[0];
    const counts = anyStrategy ? firstSummary[anyStrategy] : null;

    // The runner records time.strftime("%Y-%m-%dT%H:%M:%S") -- a local
    // wall-clock string with no zone. Shown verbatim rather than parsed
    // into a Date, which would silently reinterpret it in the viewer's
    // timezone and show a time the benchmark never ran at.
    const when = typeof meta.generated_at === "string" && meta.generated_at
      ? `<span class="mono">${escapeHtml(meta.generated_at)}</span> <span class="muted">(the benchmark machine's local clock — the runner records no timezone)</span>`
      : `<span class="muted">This results file carries no timestamp, so when it was generated is unknown.</span>`;

    const rows = [];
    rows.push(["Generated", when]);
    if (isNum(meta.memories_seeded)) {
      rows.push([
        "Corpus",
        `${escapeHtml(plural(meta.memories_seeded, "memory", "memories"))} from <span class="mono">${escapeHtml(meta.corpus_file || "the corpus file")}</span>` +
          (isNum(meta.ready_units) ? `, ingested as ${escapeHtml(plural(meta.ready_units, "retrievable unit", "retrievable units"))}` : ""),
      ]);
    }
    if (queriesRun !== null) {
      rows.push([
        "Query set",
        `${escapeHtml(plural(queriesRun, "query", "queries"))} from <span class="mono">${escapeHtml(meta.queries_file || "the query file")}</span>` +
          (counts && isNum(counts.n_answerable) && isNum(counts.n_unanswerable)
            ? ` — ${counts.n_answerable} answerable, ${counts.n_unanswerable} unanswerable`
            : "") +
          (skipped.length
            ? `<br /><span class="ev-flag">Skipped ${escapeHtml(plural(skipped.length, "query", "queries"))} whose ground truth was not seeded: <span class="mono">${escapeHtml(skipped.join(", "))}</span></span>`
            : ""),
      ]);
    }
    if (repeats !== null) {
      rows.push(["Repeats", `${escapeHtml(plural(repeats, "run", "runs"))} of the whole query set against the same stored corpus`]);
    }
    if (isNum(graph.entities) || isNum(graph.unit_associations)) {
      rows.push([
        "Association graph",
        `${escapeHtml(String(graph.entities))} entities, ${escapeHtml(String(graph.unit_entity_links))} unit–entity links, ${escapeHtml(String(graph.unit_associations))} unit associations`,
      ]);
    }
    if (typeof meta.fingerprint === "string" && meta.fingerprint) {
      rows.push([
        "Corpus fingerprint",
        `<span class="mono">${escapeHtml(meta.fingerprint.slice(0, 16))}</span> ` +
          `<span class="muted">— a hash of the corpus file and the run's size limit. ` +
          `If the corpus has been edited since, this no longer matches it and these results are stale.</span>`,
      ]);
    }
    if (meta.seeding_skipped_cache_hit === true) {
      rows.push([
        "Seeding",
        `<span class="muted">Reused a cached ingest rather than re-seeding` +
          (isNum(meta.seed_seconds_original_ingest) ? ` (the original ingest took ${Math.round(meta.seed_seconds_original_ingest)}s)` : "") +
          `.</span>`,
      ]);
    } else if (isNum(meta.seed_seconds_this_run) && meta.seed_seconds_this_run > 0) {
      rows.push(["Seeding", `<span class="muted">Re-ingested the corpus for this run, taking ${Math.round(meta.seed_seconds_this_run)}s.</span>`]);
    }

    // Two conditions that can quietly invalidate a run. Flagged here rather
    // than left for a reader to infer from the numbers.
    const flags = [];
    if (isNum(meta.limit) && meta.limit) {
      flags.push(
        `This was a <strong>limited run</strong> (<span class="mono">--limit ${escapeHtml(String(meta.limit))}</span>): only part of the corpus was seeded, and only the queries it can answer were run. A limited run is a plumbing smoke test, not a measurement of the full query set.`
      );
    }
    if (graph.unit_associations === 0) {
      flags.push(
        `The association graph is <strong>empty</strong>, so the multi-hop strategy had nothing to walk. Its numbers here are not a measurement of graph expansion.`
      );
    }
    if (isNum(graph.degraded_units) && graph.degraded_units > 0) {
      flags.push(
        `${escapeHtml(plural(graph.degraded_units, "unit", "units"))} finished ingestion with incomplete semantic enrichment (the model was unavailable or timed out). Those memories were matched against partial information and are penalised in reranking, which is grounds to re-seed rather than to believe these numbers.`
      );
    }

    return `
      <div class="card ev-panel">
        <div class="card-head">
          <div class="ev-panel-title">
            <p class="eyebrow">Where these numbers came from</p>
            <h2>This run</h2>
          </div>
        </div>
        <dl class="ev-meta">
          ${rows.map(([k, v]) => `<div class="ev-meta-row"><dt>${escapeHtml(k)}</dt><dd>${v}</dd></div>`).join("")}
        </dl>
        ${flags.length ? `<ul class="ev-flags">${flags.map((f) => `<li class="ev-flag">${f}</li>`).join("")}</ul>` : ""}
      </div>`;
  }

  function renderNoise(payload) {
    const summaries = payload.summary_per_run || [];
    const meta = payload.meta || {};
    const repeats = isNum(meta.repeat) ? meta.repeat : summaries.length;
    if (repeats > 1) {
      return `
        <div class="card ev-panel ev-noise">
          <p class="eyebrow">Reading the spread</p>
          <p>
            The query set was run <strong>${escapeHtml(plural(repeats, "time", "times"))}</strong> against the same stored
            corpus. Every figure below is the mean across those runs, and the
            range beside it is the observed minimum and maximum &mdash; the
            sampling-noise floor for this corpus and this model.
          </p>
          <p class="ev-rule">
            A gap between two strategies that is smaller than the range within
            one strategy is not a finding.
          </p>
        </div>`;
    }
    return `
      <div class="card ev-panel ev-noise">
        <p class="eyebrow">Reading these numbers</p>
        <p>
          This is a <strong>single run</strong>, so there is no spread to show and
          no noise floor to compare against. The generation model is not
          deterministic: on a query set this size, a one- or two-case flip
          between strategies is expected with no code change at all.
        </p>
        <p class="ev-rule">
          A single run cannot distinguish a real difference from model sampling
          noise. Treat any ordering below as a hypothesis, not a result.
          Re-run with <span class="mono">${escapeHtml(BENCHMARK_COMMAND)} --repeat 3</span>
          to see the noise floor first.
        </p>
      </div>`;
  }

  function renderHeadline(payload, strategies) {
    const summaries = payload.summary_per_run || [];
    const first = summaries[0] || {};
    const anyStrategy = strategies.find((s) => first[s]) || Object.keys(first)[0];
    const counts = anyStrategy ? first[anyStrategy] : null;
    const nUnanswerable = counts && isNum(counts.n_unanswerable) ? counts.n_unanswerable : null;
    const nAnswerable = counts && isNum(counts.n_answerable) ? counts.n_answerable : null;

    const upNote =
      `Measured over ${nAnswerable === null ? "the answerable queries" : escapeHtml(plural(nAnswerable, "answerable query", "answerable queries"))}. ` +
      `Higher is better.`;
    const downNote =
      `False retrieval is measured over ${nUnanswerable === null ? "the unanswerable queries" : escapeHtml(plural(nUnanswerable, "unanswerable query", "unanswerable queries"))} &mdash; ` +
      `ones where nothing stored answers the question and the correct behaviour is to say so. ` +
      `<strong>Lower is better.</strong>`;

    return `
      <div class="card ev-panel">
        <div class="card-head">
          <div class="ev-panel-title">
            <p class="eyebrow">Headline metrics</p>
            <h2>What each strategy found, and what it cost to find it</h2>
          </div>
        </div>
        <p class="ev-panel-note secondary">
          These two columns carry equal weight on purpose. A strategy can buy
          everything in the left column by simply never abstaining, and it pays
          for that on the right. There is no combined score here, and there
          will not be one: the trade between the two columns <em>is</em> the
          finding.
        </p>
        <div class="grid ev-bar-grid">
          ${barGroup("Higher is better", upNote, ["recall_at_1", "recall_at_5", "correctness"], strategies, summaries)}
          ${barGroup("Lower is better — the price of answering", downNote, ["false_retrieval_rate", "abstention_rate_answerable"], strategies, summaries)}
        </div>
      </div>`;
  }

  function renderComparison(payload, strategies) {
    const summaries = payload.summary_per_run || [];
    const multi = summaries.length > 1;

    const head = TABLE_COLUMNS.map((key) => {
      const meta = METRICS[key] || { label: key, dir: "cost" };
      const direction =
        meta.dir === "up"
          ? `<span class="ev-dir ev-dir-up">higher is better</span>`
          : meta.dir === "down"
            ? `<span class="ev-dir ev-dir-down">lower is better</span>`
            : `<span class="ev-dir ev-dir-cost">cost</span>`;
      return `<th scope="col"><span class="ev-col-label">${escapeHtml(meta.label)}</span>${direction}</th>`;
    }).join("");

    const body = strategies
      .map((strategy) => {
        const cells = TABLE_COLUMNS.map((key) => {
          const stat = across(summaries, strategy, key);
          const meta = METRICS[key] || { kind: "pct", dir: "cost" };
          if (!stat) {
            return `<td class="ev-num ev-na">n/a</td>`;
          }
          const main = meta.kind === "num" ? fixed(stat.mean) : pct(stat.mean);
          const spread =
            multi && stat.hi > stat.lo
              ? `<span class="ev-spread">${escapeHtml(
                  meta.kind === "num" ? `${fixed(stat.lo)}–${fixed(stat.hi)}` : `${pct(stat.lo, 0)}–${pct(stat.hi, 0)}`
                )}</span>`
              : "";
          return `<td class="ev-num ev-dir-cell-${meta.dir}">${escapeHtml(main)}${spread}</td>`;
        }).join("");
        return `
          <tr>
            <th scope="row" class="ev-row-head">
              ${strategyLabel(strategy)}
              <span class="ev-strategy-note">${escapeHtml(STRATEGY_NOTES[strategy] || "")}</span>
            </th>
            ${cells}
          </tr>`;
      })
      .join("");

    return `
      <div class="card ev-panel">
        <div class="card-head">
          <div class="ev-panel-title">
            <p class="eyebrow">Strategy comparison</p>
            <h2>All four strategies, every metric</h2>
          </div>
        </div>
        <p class="ev-panel-note secondary">
          ${multi
            ? "Mean across runs, with the observed minimum&ndash;maximum beneath each figure."
            : "A single run, so each figure is one observation with no spread."}
          A dash means the metric had no denominator in this run &mdash; not a zero.
        </p>
        <div class="scroll-x ev-scroll">
          <table class="ev-table ev-table-compare">
            <thead><tr><th scope="col" class="ev-row-head-col">strategy</th>${head}</tr></thead>
            <tbody>${body}</tbody>
          </table>
        </div>
        <dl class="ev-defs">
          ${METRIC_DEFINITIONS.map(
            ([term, def]) => `<div class="ev-def"><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(def)}</dd></div>`
          ).join("")}
        </dl>
      </div>`;
  }

  function renderCategories(payload, strategies) {
    const perCategory = payload.per_category_per_run || [];
    if (!perCategory.length) {
      return `
        <div class="card ev-panel">
          <p class="eyebrow">Per-category breakdown</p>
          <p class="muted">This results file carries no per-category data.</p>
        </div>`;
    }
    const present = Object.keys(perCategory[0] || {});
    const categories = orderedBy(CATEGORY_ORDER, present);

    const head = strategies.map((s) => `<th scope="col">${strategyLabel(s)}</th>`).join("");

    const body = categories
      .map((category) => {
        const unanswerable = category === "unanswerable";
        const nCell = perCategory[0][category] || {};
        const firstStrategy = strategies.find((s) => nCell[s]);
        const n = firstStrategy && isNum(nCell[firstStrategy].n) ? nCell[firstStrategy].n : null;

        const cells = strategies
          .map((strategy) => {
            const primaryKey = unanswerable ? "abstain_ok" : "correctness";
            const primary = categoryAcross(perCategory, category, strategy, primaryKey);
            if (!primary) return `<td class="ev-cat-cell ev-na">n/a</td>`;
            const recall = unanswerable ? null : categoryAcross(perCategory, category, strategy, "recall_at_5");
            const width = Math.max(0, Math.min(100, primary.mean * 100));
            const secondary = recall
              ? `<span class="ev-cat-secondary">recall@5 ${escapeHtml(pct(recall.mean, 0))}</span>`
              : unanswerable
                ? `<span class="ev-cat-secondary">correct abstentions</span>`
                : "";
            return `
              <td class="ev-cat-cell">
                <span class="ev-cat-value">${escapeHtml(pct(primary.mean, 0))}</span>
                <span class="ev-bar-track ev-cat-track" role="img" aria-label="${escapeHtml(`${strategy} on ${category}: ${pct(primary.mean, 0)}`)}">
                  <span class="ev-bar-fill ev-bar-fill-up" style="width: ${width.toFixed(2)}%"></span>
                </span>
                ${secondary}
              </td>`;
          })
          .join("");

        return `
          <tr>
            <th scope="row" class="ev-row-head">
              <span class="mono ev-cat-name">${escapeHtml(category)}</span>
              ${n === null ? "" : `<span class="pill">${escapeHtml(plural(n, "query", "queries"))}</span>`}
              <span class="ev-cat-note">${escapeHtml(CATEGORY_NOTES[category] || "")}</span>
            </th>
            ${cells}
          </tr>`;
      })
      .join("");

    return `
      <div class="card ev-panel">
        <div class="card-head">
          <div class="ev-panel-title">
            <p class="eyebrow">Per-category breakdown</p>
            <h2>Where each strategy actually earns its average</h2>
          </div>
        </div>
        <p class="ev-panel-note secondary">
          The most informative view on this page. An overall winner that loses
          on <span class="mono">multi_hop</span> or <span class="mono">ambiguous</span>
          queries is only visible here. The figure is answer correctness, with
          recall@5 beneath it; for <span class="mono">unanswerable</span> queries
          it is the correct-abstention rate instead, since there is nothing to
          retrieve and abstaining is the right answer.
        </p>
        <div class="scroll-x ev-scroll">
          <table class="ev-table ev-table-category">
            <thead><tr><th scope="col" class="ev-row-head-col">category</th>${head}</tr></thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      </div>`;
  }

  // For one ordered pair: every query the first strategy got right and the
  // second got wrong, with how many of the repeats it flipped in. Mirrors
  // print_win_sets() in the runner.
  function winSet(runs, a, b) {
    const counts = new Map();
    for (const rows of runs) {
      if (!Array.isArray(rows)) continue;
      const lookup = new Map();
      const ids = [];
      for (const row of rows) {
        lookup.set(`${row.query_id} ${row.strategy}`, row);
        if (ids.indexOf(row.query_id) === -1) ids.push(row.query_id);
      }
      for (const qid of ids) {
        const ra = lookup.get(`${qid} ${a}`);
        const rb = lookup.get(`${qid} ${b}`);
        if (!ra || !rb) continue;
        if (didSucceed(ra) && !didSucceed(rb)) {
          const entry = counts.get(qid) || { flips: 0, question: ra.question, category: ra.category, won: ra, lost: rb };
          entry.flips += 1;
          counts.set(qid, entry);
        }
      }
    }
    return Array.from(counts.entries())
      .map(([qid, entry]) => Object.assign({ qid }, entry))
      .sort((x, y) => y.flips - x.flips || String(x.qid).localeCompare(String(y.qid), undefined, { numeric: true }));
  }

  function renderWinList(runs, a, b, nRuns) {
    const entries = winSet(runs, a, b);
    if (!entries.length) {
      return `<p class="muted ev-win-empty">No query that <span class="mono">${escapeHtml(a)}</span> got right and <span class="mono">${escapeHtml(b)}</span> got wrong.</p>`;
    }
    const items = entries
      .map((e) => {
        // How many of the repeats this query actually flipped in. Below the
        // full count it is sampling noise rather than a finding, so the
        // marker is warning-toned -- said once in the panel note above
        // rather than repeated as a sentence on every row.
        const flips =
          nRuns > 1
            ? `<span class="pill ${e.flips === nRuns ? "pill-strong" : "ev-win-flips-partial"}">${e.flips}/${nRuns} runs</span>`
            : "";
        const answers = [
          e.won && typeof e.won.answer === "string" ? [a, e.won.answer, true] : null,
          e.lost && typeof e.lost.answer === "string" ? [b, e.lost.answer, false] : null,
        ].filter(Boolean);
        const detail = answers.length
          ? `<details class="ev-win-detail">
               <summary>What each one answered</summary>
               ${answers
                 .map(
                   ([who, text, ok]) => `
                 <div class="ev-win-answer">
                   <p class="eyebrow">${escapeHtml(who)} ${ok ? "— counted correct" : "— counted wrong"}</p>
                   <p class="ev-win-answer-text">${escapeHtml(text) || '<span class="muted">(empty)</span>'}</p>
                 </div>`
                 )
                 .join("")}
               <p class="muted">Correctness is a substring test, not a judge &mdash; read these before trusting an individual verdict.</p>
             </details>`
          : "";
        return `
          <li class="ev-win-item">
            <div class="row row-tight ev-win-head">
              <span class="mono ev-win-id">${escapeHtml(String(e.qid))}</span>
              <span class="badge badge-neutral">${escapeHtml(e.category || "")}</span>
              ${flips}
            </div>
            <p class="ev-win-question">${escapeHtml(e.question || "")}</p>
            ${detail}
          </li>`;
      })
      .join("");
    return `<ul class="ev-wins">${items}</ul>`;
  }

  function renderWinSets(payload, strategies) {
    const runs = payload.runs || [];
    if (!runs.length) {
      return `
        <div class="card ev-panel">
          <p class="eyebrow">Pairwise win sets</p>
          <p class="muted">This results file carries no per-query rows, so win sets cannot be computed.</p>
        </div>`;
    }
    const a = strategies[0];
    const b = strategies.length > 1 ? strategies[1] : strategies[0];
    const options = (selected) =>
      strategies.map((s) => `<option value="${escapeHtml(s)}"${s === selected ? " selected" : ""}>${escapeHtml(s)}</option>`).join("");

    return `
      <div class="card ev-panel">
        <div class="card-head">
          <div class="ev-panel-title">
            <p class="eyebrow">Pairwise win sets</p>
            <h2>Which specific queries one strategy got right and the other got wrong</h2>
          </div>
        </div>
        <p class="ev-panel-note secondary">
          An aggregate gap of a few points says nothing about whether two
          strategies fail on the same queries or on different ones. Only this
          distinguishes &ldquo;strictly better&rdquo; from &ldquo;differently
          wrong&rdquo;. <em>Got it right</em> means a correct grounded answer on
          an answerable query, and a correct abstention on an unanswerable one.
          ${runs.length > 1
            ? `The <span class="pill ev-win-flips-partial">k/${escapeHtml(String(runs.length))} runs</span>
               marker is how many of the repeats a query flipped in; anything
               below ${escapeHtml(String(runs.length))}/${escapeHtml(String(runs.length))} is sampling noise, not a finding.`
            : `With a single run there is no way to tell a stable win from one the model would not repeat.`}
        </p>
        <div class="row ev-pick">
          <label class="ev-pick-field">
            <span class="label">Strategy A</span>
            <select id="ev-pick-a">${options(a)}</select>
          </label>
          <label class="ev-pick-field">
            <span class="label">Strategy B</span>
            <select id="ev-pick-b">${options(b)}</select>
          </label>
        </div>
        <div id="ev-win-sets" class="grid ev-win-grid"></div>
      </div>`;
  }

  function paintWinSets(payload, container, a, b) {
    const runs = payload.runs || [];
    const nRuns = runs.length;
    if (a === b) {
      container.innerHTML = `<p class="muted">Pick two different strategies to compare.</p>`;
      return;
    }
    container.innerHTML = `
      <div class="card card-quiet ev-win-panel">
        <p class="eyebrow"><span class="mono">${escapeHtml(a)}</span> right, <span class="mono">${escapeHtml(b)}</span> wrong</p>
        ${renderWinList(runs, a, b, nRuns)}
      </div>
      <div class="card card-quiet ev-win-panel">
        <p class="eyebrow"><span class="mono">${escapeHtml(b)}</span> right, <span class="mono">${escapeHtml(a)}</span> wrong</p>
        ${renderWinList(runs, b, a, nRuns)}
      </div>`;
  }

  function renderCaveats() {
    return `
      <div class="card ev-panel ev-caveats">
        <div class="card-head">
          <div class="ev-panel-title">
            <p class="eyebrow">Caveats</p>
            <h2>Reasons not to over-read any of this</h2>
          </div>
        </div>
        <dl class="ev-defs">
          ${CAVEATS.map(([term, def]) => `<div class="ev-def"><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(def)}</dd></div>`).join("")}
        </dl>
      </div>`;
  }

  // The artefact is not published, so there is nothing to render. This is a
  // normal state, not an error: the benchmark takes 20+ minutes, needs a
  // database and a local model, and is never run as part of the build.
  function renderEmpty(reason) {
    return `
      <div class="card placeholder ev-empty">
        <p class="eyebrow">No results published</p>
        <p>
          ${escapeHtml(reason)}
        </p>
        <p>
          This page renders a file produced <strong>offline</strong>. The
          benchmark seeds a corpus through the real ingestion pipeline into an
          isolated database and runs four retrieval strategies over the whole
          query set, which takes upwards of twenty minutes and needs both a
          database and a local model &mdash; so it is never run as part of the
          site build, and a build on a machine that has never run it simply
          publishes no results.
        </p>
        <p>To produce them, from the repository root:</p>
        <pre class="derived ev-command">nix develop
${escapeHtml(BENCHMARK_COMMAND)} --self-check   # validate the fixtures only
${escapeHtml(BENCHMARK_COMMAND)}                # full run
${escapeHtml(BENCHMARK_COMMAND)} --repeat 3     # full run, three times, with the noise floor</pre>
        <p class="muted">
          The run writes <span class="mono">eval/strategy_benchmark_results.json</span>;
          the next frontend build copies it to
          <span class="mono">static/evaluation-results.json</span> and this page
          picks it up. See <span class="mono">docs/EVALUATION.md</span> for the
          metric definitions and the caveats.
        </p>
      </div>`;
  }

  function render(root, payload) {
    const strategies = resolveStrategies(payload);
    if (!strategies.length || !(payload.summary_per_run || []).length) {
      root.innerHTML = renderEmpty(
        "A results file was found, but it carries no per-strategy summary — it may have been written by an older or interrupted run."
      );
      return;
    }

    root.innerHTML = [
      renderProvenance(payload),
      renderNoise(payload),
      renderHeadline(payload, strategies),
      renderComparison(payload, strategies),
      renderCategories(payload, strategies),
      renderWinSets(payload, strategies),
      renderCaveats(),
    ].join("");

    const pickA = document.getElementById("ev-pick-a");
    const pickB = document.getElementById("ev-pick-b");
    const winContainer = document.getElementById("ev-win-sets");
    if (pickA && pickB && winContainer) {
      const repaint = () => paintWinSets(payload, winContainer, pickA.value, pickB.value);
      pickA.addEventListener("change", repaint);
      pickB.addEventListener("change", repaint);
      repaint();
    }
  }

  async function initEvaluationPage() {
    const root = document.getElementById("evaluation-root");
    if (!root) return;
    root.innerHTML = `<p class="status"><span class="spinner" aria-hidden="true"></span>Loading benchmark results...</p>`;

    let payload;
    try {
      const resp = await fetch(RESULTS_URL, { cache: "no-cache" });
      if (resp.status === 404) {
        root.innerHTML = renderEmpty("No benchmark results have been published with this build of the site.");
        return;
      }
      if (!resp.ok) {
        root.innerHTML = renderEmpty(
          `The published results file could not be read (HTTP ${resp.status}).`
        );
        return;
      }
      payload = await resp.json();
    } catch (err) {
      // Covers a malformed artefact, and a page opened straight off disk
      // via file:// where a relative fetch is blocked. Neither is worth an
      // alarming error state -- the page has nothing to show either way.
      root.innerHTML = renderEmpty(
        `The published results file could not be loaded or parsed (${err && err.message ? err.message : "unknown error"}).`
      );
      return;
    }

    if (!payload || typeof payload !== "object") {
      root.innerHTML = renderEmpty("The published results file is not a JSON object.");
      return;
    }
    render(root, payload);
  }

  window.membase.initEvaluationPage = initEvaluationPage;
})();
