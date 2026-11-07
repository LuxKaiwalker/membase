// ===========================================================================
// graph-canvas.js -- the force-directed <canvas> renderer behind the
// Explorer page. Nothing in here knows about the membase API, the filter
// chips, or the inspector panel: it takes a {nodes, edges} snapshot in the
// shape GraphOut already has, lays it out, draws it, and reports what the
// pointer is over. pages/explorer.js owns all of that.
//
// Loaded by pages/explorer.js (which injects a <script> tag for it and
// awaits the load), NOT by base.html -- base.html's registry only knows
// how to load core.js plus one page script, and this file is a page-local
// concern rather than something every section needs. It registers
// `membase.GraphCanvas` and nothing else.
//
// Zero dependencies, by repo rule: no npm, no bundler, no CDN script, no
// vendored library. The layout below is a plain velocity-Verlet-ish
// spring/repulsion integrator, ~120 lines, which is all a few hundred
// nodes need -- see LAYOUT below for the parameters and why each exists.
//
// Two things this deliberately does NOT do:
//
//   * It does not run requestAnimationFrame forever. The simulation cools
//     on a fixed schedule and the loop is CANCELLED once the layout has
//     settled; interaction (drag, resize, a new graph) re-heats it. A
//     background tab of this page burns no frames at all.
//   * It does not hardcode a colour. Every stroke and fill is read out of
//     app.css's custom properties at runtime via getComputedStyle, and
//     re-read when the colour scheme changes.
// ===========================================================================
(() => {
  "use strict";

  const TAU = Math.PI * 2;

  // --- LAYOUT -------------------------------------------------------------
  // A force-directed layout is three forces and two brakes:
  //
  //   repulsion   every node pushes every nearby node away (inverse-square,
  //               so it dominates at short range and vanishes at long range)
  //   springs     every edge pulls its two endpoints toward a rest length
  //   gravity     a weak pull toward the origin, which keeps disconnected
  //               components (a memory with no associations, an entity
  //               mentioned once) from drifting off to infinity instead of
  //               just sitting beside the rest of the graph
  //   damping     velocity is multiplied down every step, so the system
  //               loses energy instead of oscillating
  //   alpha       a global cooling factor scaling every force, decaying
  //               geometrically -- this is what makes the layout SETTLE and
  //               therefore what makes stopping the loop possible at all
  //
  // Repulsion is the only O(n^2) term, so it gets a uniform spatial grid
  // with a hard cutoff: a node only considers the nine grid cells around
  // it. At the cutoff distance the inverse-square force is already ~1% of
  // its short-range value, so the cutoff changes the result negligibly
  // while turning the hot loop linear in practice. No Barnes-Hut tree
  // needed at this scale.
  const LAYOUT = {
    repulsion: 5600,      // numerator of the inverse-square repulsion
    repulsionCutoff: 300, // world units past which repulsion is ignored
    springStiffness: 0.05,
    gravity: 0.02,
    damping: 0.86,
    alphaStart: 1,
    alphaDecay: 0.982,    // ~245 steps to alphaMin
    alphaMin: 0.012,
    maxSpeed: 30,         // world units/step, guards against a blow-up
    maxSteps: 1600,       // absolute termination guarantee
    // Rest lengths per edge type. A memory sits close to the things it
    // mentions; two memories associated only by a shared entity sit further
    // apart, so the entity between them has somewhere to be.
    restMentions: 76,
    restRelation: 58,
    restAssociation: 165,
    // Settled = cooled down AND no longer visibly moving. Both, because
    // alpha alone can hit its floor while a dense cluster is still
    // unwinding, and a velocity test alone can pass for one lucky step.
    quietSpeed: 0.08,
    quietSteps: 3,
    // Steps run before the first paint, so the opening frame is a rough
    // layout rather than the phyllotaxis spiral every graph starts from.
    presteps: 24,
  };

  const ZOOM_MIN = 0.15;
  const ZOOM_MAX = 4;
  const EDGE_HIT_PX = 7;      // screen px of slop for picking an edge
  const CLICK_SLOP_PX = 4;    // drag shorter than this counts as a click
  const MAX_LABELS = 22;      // highest-degree nodes that get a resting label
  const LABEL_CHARS = 28;

  // Entity kind -> shape. SHAPE is the primary channel, colour the
  // secondary one, for the same reason app.css's provenance vocabulary
  // uses a glyph and a border style and a colour all at once: the
  // distinction has to survive greyscale and colour-blindness. The legend
  // in explorer.html uses the matching unicode glyph for each of these.
  const SHAPES = {
    memory: "note",
    person: "circle",
    organization: "hexagon",
    location: "triangle",
    topic: "diamond",
    concept: "ring",
    object: "square",
  };

  // Entity kind -> app.css token. Deliberately reusing the semantic tokens
  // rather than inventing seven new hues: app.css owns the palette, and
  // shape already carries the distinction if two kinds land on a similar
  // colour.
  const KIND_TOKENS = {
    memory: "--accent",
    person: "--success",
    organization: "--warning",
    location: "--danger",
    topic: "--accent-strong",
    concept: "--text-secondary",
    object: "--text-muted",
  };

  // Reads the live values of app.css's tokens. Called at construction and
  // again whenever the colour scheme flips, because the canvas is pixels
  // and gets no `@media (prefers-color-scheme)` for free the way the DOM
  // does. The fallback is the canvas element's own computed `color` -- a
  // real token-derived value, so no literal colour is ever hardcoded here.
  function readPalette(el) {
    const cs = getComputedStyle(document.documentElement);
    const fallback = getComputedStyle(el).color;
    const v = (name) => cs.getPropertyValue(name).trim() || fallback;
    const palette = {
      bg: v("--surface-sunken"),
      edgeExplicit: v("--text-secondary"),
      edgeExtracted: v("--accent-edge"),
      edgeInferred: v("--warning"),
      edgeHighlight: v("--accent-strong"),
      nodeOutline: v("--surface-sunken"),
      labelText: v("--text-primary"),
      labelHalo: v("--surface-sunken"),
      selectRing: v("--accent-strong"),
      pinRing: v("--text-muted"),
      font: cs.getPropertyValue("--font-sans").trim() || "system-ui, sans-serif",
      kinds: {},
    };
    for (const [kind, token] of Object.entries(KIND_TOKENS)) palette.kinds[kind] = v(token);
    return palette;
  }

  function kindColor(palette, node) {
    return palette.kinds[node.kind] || palette.kinds.concept;
  }

  // Origin is the authority distinction -- an association the system
  // inferred must not be drawn the way something the user stated is. Line
  // STYLE carries it (solid vs. dashed), exactly as app.css's `.prov-*`
  // classes do with border-style, with colour and thickness reinforcing
  // rather than replacing that. Thickness additionally encodes `weight`.
  function edgeStyle(palette, edge) {
    if (edge.origin === "inferred") {
      return { color: palette.edgeInferred, dash: [5, 4], width: 0.8 + 1.8 * edge.weight };
    }
    if (edge.origin === "explicit") {
      return { color: palette.edgeExplicit, dash: null, width: 1.2 + 2.2 * edge.weight };
    }
    // "extracted" (and anything unexpected): solid, but thinner and
    // quieter than a user-stated link.
    return { color: palette.edgeExtracted, dash: null, width: 0.7 + 1.4 * edge.weight };
  }

  function restLengthFor(edge) {
    if (edge.edge_type === "mentions") return LAYOUT.restMentions;
    if (edge.edge_type === "relation") return LAYOUT.restRelation;
    return LAYOUT.restAssociation;
  }

  function radiusFor(node) {
    const base = node.node_type === "unit" ? 7 : 5;
    const r = base + 3.3 * Math.sqrt(Math.max(0, node.degree));
    return Math.min(node.node_type === "unit" ? 27 : 23, r);
  }

  function truncate(text, max) {
    const s = String(text || "");
    return s.length <= max ? s : s.slice(0, max - 1).trimEnd() + "…";
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function polygonPath(ctx, x, y, r, sides, rotation) {
    ctx.beginPath();
    for (let i = 0; i < sides; i += 1) {
      const a = rotation + (i * TAU) / sides;
      const px = x + Math.cos(a) * r;
      const py = y + Math.sin(a) * r;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
  }

  function shapePath(ctx, x, y, r, shape) {
    switch (shape) {
      case "note":
        roundRect(ctx, x - r * 1.15, y - r * 0.72, r * 2.3, r * 1.44, Math.max(2, r * 0.32));
        break;
      case "square":
        roundRect(ctx, x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7, Math.max(1, r * 0.2));
        break;
      case "triangle":
        polygonPath(ctx, x, y + r * 0.18, r * 1.18, 3, -Math.PI / 2);
        break;
      case "diamond":
        polygonPath(ctx, x, y, r * 1.12, 4, -Math.PI / 2);
        break;
      case "hexagon":
        polygonPath(ctx, x, y, r * 1.08, 6, -Math.PI / 2);
        break;
      case "ring":
      case "circle":
      default:
        ctx.beginPath();
        ctx.arc(x, y, r, 0, TAU);
        break;
    }
  }

  // Distance from point p to segment ab, in whatever units all three are
  // expressed in. Used for edge hit-testing.
  function distToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - ax, py - ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  class GraphCanvas {
    constructor(canvas, handlers = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.handlers = handlers;

      this.nodes = [];
      this.edges = [];
      this.nodeById = new Map();

      // Camera: world -> screen is `world * scale + offset`, applied by
      // hand rather than via ctx transforms, so label type and line widths
      // stay a constant number of screen pixels at every zoom level and
      // hit-testing is the exact inverse of one multiply and one add.
      this.scale = 1;
      this.offsetX = 0;
      this.offsetY = 0;
      this.autoFit = true;

      this.width = 0;
      this.height = 0;
      this.dpr = 1;

      this.alpha = 0;
      this.quiet = 0;
      this.steps = 0;
      this.running = false;
      this.rafId = null;
      this.pendingRender = null;
      this.pausedByVisibility = false;

      // Instrumentation. `frames` counts actual paints, which is how the
      // "the loop really does stop" claim is verifiable from the console
      // rather than taken on trust.
      this.frames = 0;
      this.settleMs = null;
      this.startedAt = null;

      this.hovered = null;
      this.hoveredEdge = null;
      this.selectedNode = null;
      this.selectedEdge = null;
      this.focusId = null;
      this.focusSet = null;

      this.drag = null;
      this.pan = null;

      this.reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      this.palette = readPalette(canvas);

      this._bind();
      this.resize();
    }

    // ---------------------------------------------------------------- setup

    _bind() {
      const c = this.canvas;
      this._onPointerDown = (ev) => this._pointerDown(ev);
      this._onPointerMove = (ev) => this._pointerMove(ev);
      this._onPointerUp = (ev) => this._pointerUp(ev);
      this._onPointerLeave = () => this._pointerLeave();
      this._onWheel = (ev) => this._wheel(ev);
      this._onDblClick = (ev) => this._dblClick(ev);
      this._onKeyDown = (ev) => this._keyDown(ev);

      c.addEventListener("pointerdown", this._onPointerDown);
      c.addEventListener("pointermove", this._onPointerMove);
      c.addEventListener("pointerup", this._onPointerUp);
      c.addEventListener("pointercancel", this._onPointerUp);
      c.addEventListener("pointerleave", this._onPointerLeave);
      // Not passive: zooming the graph must not also scroll the page.
      c.addEventListener("wheel", this._onWheel, { passive: false });
      c.addEventListener("dblclick", this._onDblClick);
      c.addEventListener("keydown", this._onKeyDown);

      this._onVisibility = () => {
        if (document.hidden) {
          if (this.running) {
            this.pausedByVisibility = true;
            this._stopLoop();
          }
        } else if (this.pausedByVisibility) {
          this.pausedByVisibility = false;
          this._startLoop();
        }
      };
      document.addEventListener("visibilitychange", this._onVisibility);

      // The canvas is pixels, so a colour-scheme flip has to be handled in
      // JS; app.css's own `@media (prefers-color-scheme: light)` block
      // cannot reach inside a <canvas>.
      this._schemeQuery = window.matchMedia("(prefers-color-scheme: dark)");
      this._onScheme = () => {
        this.palette = readPalette(this.canvas);
        this.requestRender();
      };
      if (this._schemeQuery.addEventListener) {
        this._schemeQuery.addEventListener("change", this._onScheme);
      }

      this._observer = new ResizeObserver(() => this.resize());
      this._observer.observe(this.canvas.parentElement || this.canvas);
    }

    destroy() {
      const c = this.canvas;
      c.removeEventListener("pointerdown", this._onPointerDown);
      c.removeEventListener("pointermove", this._onPointerMove);
      c.removeEventListener("pointerup", this._onPointerUp);
      c.removeEventListener("pointercancel", this._onPointerUp);
      c.removeEventListener("pointerleave", this._onPointerLeave);
      c.removeEventListener("wheel", this._onWheel);
      c.removeEventListener("dblclick", this._onDblClick);
      c.removeEventListener("keydown", this._onKeyDown);
      document.removeEventListener("visibilitychange", this._onVisibility);
      if (this._schemeQuery.removeEventListener) {
        this._schemeQuery.removeEventListener("change", this._onScheme);
      }
      if (this._observer) this._observer.disconnect();
      this._stopLoop();
      if (this.pendingRender !== null) cancelAnimationFrame(this.pendingRender);
    }

    // A devicePixelRatio > 1 display needs a backing store that many times
    // larger than the CSS box, with every drawing coordinate still
    // expressed in CSS pixels -- otherwise lines are blurry and, worse,
    // hit-testing silently disagrees with what was drawn.
    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const cssW = Math.max(1, Math.round(rect.width));
      const cssH = Math.max(1, Math.round(rect.height));
      const dpr = window.devicePixelRatio || 1;
      if (cssW === this.width && cssH === this.height && dpr === this.dpr) return;
      const first = this.width === 0;
      this.width = cssW;
      this.height = cssH;
      this.dpr = dpr;
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
      if (!first && this.nodes.length) {
        // A resize changes what "fits", so give the layout a nudge and
        // re-fit unless the user has taken the camera over.
        if (this.autoFit) this.fitToView();
        this.reheat(0.25);
      }
      this.requestRender();
    }

    // --------------------------------------------------------------- graph

    setGraph(snapshot) {
      const previous = this.nodeById;
      this.nodes = [];
      this.nodeById = new Map();
      const count = snapshot.nodes.length;
      snapshot.nodes.forEach((raw, i) => {
        // Phyllotaxis seeding: deterministic (no Math.random, so the same
        // graph always lays out the same way, which makes a visual
        // regression reproducible), and evenly spread, which matters
        // because a force layout started from coincident points spends its
        // first dozen steps just exploding.
        const angle = i * 2.3999632297;
        const radius = 26 * Math.sqrt(i + 0.5);
        const carried = previous.get(raw.id);
        const node = {
          ...raw,
          x: carried ? carried.x : Math.cos(angle) * radius,
          y: carried ? carried.y : Math.sin(angle) * radius,
          vx: 0,
          vy: 0,
          // A node the user dropped somewhere stays there across a
          // re-fetch; everything else is free again.
          pinned: carried ? carried.pinned : false,
          mass: 1 + Math.min(12, raw.degree || 0) * 0.3,
          r: radiusFor(raw),
          shape: SHAPES[raw.kind] || "circle",
          index: i,
          neighbors: new Set(),
          incident: [],
        };
        this.nodes.push(node);
        this.nodeById.set(node.id, node);
      });

      this.edges = [];
      snapshot.edges.forEach((raw, i) => {
        const s = this.nodeById.get(raw.source);
        const t = this.nodeById.get(raw.target);
        // An edge whose endpoint was filtered out of the snapshot is not
        // drawable; dropping it here is the only sane thing, and the
        // backend's own filtering already keeps this rare.
        if (!s || !t || s === t) return;
        const edge = { ...raw, s, t, key: i, rest: restLengthFor(raw), k: LAYOUT.springStiffness * (0.35 + Math.min(1, raw.weight)) };
        this.edges.push(edge);
        s.neighbors.add(t.id);
        t.neighbors.add(s.id);
        s.incident.push(edge);
        t.incident.push(edge);
      });

      // Only the best-connected handful get a resting label; everything
      // else labels on hover, on selection, or when zoomed in. A label per
      // node at this density is unreadable noise.
      const byDegree = [...this.nodes].sort((a, b) => (b.degree || 0) - (a.degree || 0));
      byDegree.forEach((n, i) => {
        n.showLabel = i < MAX_LABELS;
      });

      this.selectedNode = this.selectedNode ? this.nodeById.get(this.selectedNode.id) || null : null;
      this.selectedEdge = null;
      this.hovered = null;
      this.hoveredEdge = null;
      if (this.focusId && !this.nodeById.has(this.focusId)) this.setFocus(null);
      else this.setFocus(this.focusId);

      this.frames = 0;
      this.steps = 0;
      this.quiet = 0;
      this.settleMs = null;
      this.startedAt = performance.now();
      this.autoFit = true;

      if (!this.nodes.length) {
        this.alpha = 0;
        this._stopLoop();
        this.requestRender();
        this._report();
        return;
      }

      // prefers-reduced-motion: do the whole settling in one synchronous
      // burst and paint the converged layout once. No animation at all --
      // the result is identical, the motion is not.
      if (this.reduceMotion) {
        this.alpha = LAYOUT.alphaStart;
        while (!this._settled()) this._step();
        this.settleMs = performance.now() - this.startedAt;
        this.fitToView();
        this.requestRender();
        this._report();
        return;
      }

      this.alpha = LAYOUT.alphaStart;
      for (let i = 0; i < LAYOUT.presteps && !this._settled(); i += 1) this._step();
      this.fitToView();
      this._startLoop();
    }

    // --------------------------------------------------------------- forces

    _step() {
      const { nodes, edges } = this;
      const alpha = this.alpha;
      const n = nodes.length;
      for (let i = 0; i < n; i += 1) {
        nodes[i].fx = 0;
        nodes[i].fy = 0;
      }

      // Repulsion, over a uniform grid whose cell size is the cutoff, so
      // each node only looks at the 3x3 block of cells around it.
      const cell = LAYOUT.repulsionCutoff;
      const grid = new Map();
      for (let i = 0; i < n; i += 1) {
        const node = nodes[i];
        const key = `${Math.floor(node.x / cell)},${Math.floor(node.y / cell)}`;
        const bucket = grid.get(key);
        if (bucket) bucket.push(node);
        else grid.set(key, [node]);
      }
      const cutoffSq = cell * cell;
      for (let i = 0; i < n; i += 1) {
        const a = nodes[i];
        const cx = Math.floor(a.x / cell);
        const cy = Math.floor(a.y / cell);
        for (let gx = cx - 1; gx <= cx + 1; gx += 1) {
          for (let gy = cy - 1; gy <= cy + 1; gy += 1) {
            const bucket = grid.get(`${gx},${gy}`);
            if (!bucket) continue;
            for (let j = 0; j < bucket.length; j += 1) {
              const b = bucket[j];
              if (b === a) continue;
              let dx = a.x - b.x;
              let dy = a.y - b.y;
              let dSq = dx * dx + dy * dy;
              if (dSq > cutoffSq) continue;
              if (dSq < 0.01) {
                // Exactly coincident: nudge apart deterministically using
                // the index, so the pair separates instead of dividing by
                // zero forever.
                dx = ((a.index % 7) - 3) * 0.1 + 0.05;
                dy = ((a.index % 5) - 2) * 0.1 + 0.05;
                dSq = dx * dx + dy * dy;
              }
              const d = Math.sqrt(dSq);
              // Inverse square, scaled by both masses: a hub pushes harder
              // and is harder to push, which keeps high-degree nodes from
              // being squeezed into the middle of their own neighbours.
              const f = (LAYOUT.repulsion * a.mass * b.mass) / dSq;
              a.fx += (dx / d) * f;
              a.fy += (dy / d) * f;
            }
          }
        }
      }

      // Springs along edges.
      for (let i = 0; i < edges.length; i += 1) {
        const e = edges[i];
        const dx = e.t.x - e.s.x;
        const dy = e.t.y - e.s.y;
        const d = Math.hypot(dx, dy) || 0.01;
        const f = e.k * (d - e.rest);
        const ux = (dx / d) * f;
        const uy = (dy / d) * f;
        e.s.fx += ux;
        e.s.fy += uy;
        e.t.fx -= ux;
        e.t.fy -= uy;
      }

      // Centring gravity, proportional to distance so it is negligible in
      // the middle and firm at the edges.
      for (let i = 0; i < n; i += 1) {
        const node = nodes[i];
        node.fx -= node.x * LAYOUT.gravity * node.mass;
        node.fy -= node.y * LAYOUT.gravity * node.mass;
      }

      // Integrate. Velocity picks up alpha-scaled acceleration and is then
      // damped, so energy leaves the system on two independent schedules
      // (cooling and friction) and the layout cannot ring forever.
      let maxMove = 0;
      for (let i = 0; i < n; i += 1) {
        const node = nodes[i];
        if (node.pinned || node === (this.drag && this.drag.node)) {
          node.vx = 0;
          node.vy = 0;
          continue;
        }
        node.vx = (node.vx + (alpha * node.fx) / node.mass) * LAYOUT.damping;
        node.vy = (node.vy + (alpha * node.fy) / node.mass) * LAYOUT.damping;
        const speed = Math.hypot(node.vx, node.vy);
        if (speed > LAYOUT.maxSpeed) {
          node.vx = (node.vx / speed) * LAYOUT.maxSpeed;
          node.vy = (node.vy / speed) * LAYOUT.maxSpeed;
        }
        node.x += node.vx;
        node.y += node.vy;
        if (speed > maxMove) maxMove = speed;
      }

      this.alpha *= LAYOUT.alphaDecay;
      this.steps += 1;
      this.lastMaxMove = maxMove;
      if (this.alpha < LAYOUT.alphaMin && maxMove < LAYOUT.quietSpeed) this.quiet += 1;
      else this.quiet = 0;
    }

    _settled() {
      if (!this.nodes.length) return true;
      if (this.steps >= LAYOUT.maxSteps) return true;
      return this.quiet >= LAYOUT.quietSteps;
    }

    // ----------------------------------------------------------- the loop

    _startLoop() {
      if (this.running || !this.nodes.length) return;
      if (document.hidden) {
        // Nothing to animate for nobody to see. Resumed by the
        // visibilitychange handler.
        this.pausedByVisibility = true;
        return;
      }
      this.running = true;
      if (this.startedAt === null) this.startedAt = performance.now();
      const tick = () => {
        if (!this.running) return;
        // A couple of integration steps per frame: converges in fewer
        // frames without making any single frame long.
        this._step();
        this._step();
        this._paint();
        if (this.autoFit && this.steps % 8 === 0) this.fitToView();
        if (this._settled()) {
          this.settleMs = performance.now() - this.startedAt;
          this._stopLoop();
          // The final paint happens with the layout at rest, so what is on
          // screen is exactly the converged state and nothing further is
          // scheduled.
          if (this.autoFit) this.fitToView();
          this._paint();
          this._report();
          return;
        }
        this.rafId = requestAnimationFrame(tick);
      };
      this.rafId = requestAnimationFrame(tick);
    }

    _stopLoop() {
      this.running = false;
      if (this.rafId !== null) {
        cancelAnimationFrame(this.rafId);
        this.rafId = null;
      }
    }

    // Re-heat after an interaction that invalidates the settled layout.
    reheat(amount = 0.45) {
      if (!this.nodes.length || this.reduceMotion) {
        // Under reduced motion, re-settle synchronously instead of
        // animating back to rest.
        if (this.nodes.length && this.reduceMotion) {
          this.alpha = Math.max(this.alpha, amount);
          this.quiet = 0;
          const limit = this.steps + 400;
          while (!this._settled() && this.steps < limit) this._step();
          this.requestRender();
        }
        return;
      }
      this.alpha = Math.max(this.alpha, amount);
      this.quiet = 0;
      if (this.steps >= LAYOUT.maxSteps) this.steps = 0;
      if (this.startedAt === null) this.startedAt = performance.now();
      this._startLoop();
    }

    // One-off repaint for something that changed appearance but not
    // positions (hover, selection, pan, zoom, colour scheme). Coalesced
    // into a single frame and never starts the simulation.
    requestRender() {
      if (this.running || this.pendingRender !== null) return;
      this.pendingRender = requestAnimationFrame(() => {
        this.pendingRender = null;
        this._paint();
      });
    }

    _report() {
      if (this.handlers.onSettled) {
        this.handlers.onSettled(this.getMetrics());
      }
    }

    getMetrics() {
      return {
        nodes: this.nodes.length,
        edges: this.edges.length,
        steps: this.steps,
        frames: this.frames,
        settleMs: this.settleMs,
        running: this.running,
        reduceMotion: this.reduceMotion,
      };
    }

    // --------------------------------------------------------------- camera

    toWorld(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: (clientX - rect.left - this.offsetX) / this.scale,
        y: (clientY - rect.top - this.offsetY) / this.scale,
      };
    }

    fitToView() {
      if (!this.nodes.length || !this.width) return;
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (const n of this.nodes) {
        if (n.x - n.r < minX) minX = n.x - n.r;
        if (n.y - n.r < minY) minY = n.y - n.r;
        if (n.x + n.r > maxX) maxX = n.x + n.r;
        if (n.y + n.r > maxY) maxY = n.y + n.r;
      }
      const pad = 28;
      const w = Math.max(1, maxX - minX);
      const h = Math.max(1, maxY - minY);
      const scale = Math.min((this.width - pad * 2) / w, (this.height - pad * 2) / h);
      this.scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, scale));
      this.offsetX = this.width / 2 - ((minX + maxX) / 2) * this.scale;
      this.offsetY = this.height / 2 - ((minY + maxY) / 2) * this.scale;
    }

    resetView() {
      this.autoFit = true;
      this.fitToView();
      this.requestRender();
    }

    // Centres the camera on one node without changing the layout -- how
    // "recentre on this entity's neighbourhood" and the keyboard entity
    // search both land somewhere visible.
    centerOn(nodeId, zoom) {
      const node = this.nodeById.get(nodeId);
      if (!node) return false;
      this.autoFit = false;
      if (zoom) this.scale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom));
      this.offsetX = this.width / 2 - node.x * this.scale;
      this.offsetY = this.height / 2 - node.y * this.scale;
      this.requestRender();
      return true;
    }

    _zoomBy(factor, anchorX, anchorY) {
      const next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, this.scale * factor));
      if (next === this.scale) return;
      // Keep the world point under the anchor fixed, so zooming feels like
      // it happens where the pointer is rather than at the canvas centre.
      const wx = (anchorX - this.offsetX) / this.scale;
      const wy = (anchorY - this.offsetY) / this.scale;
      this.scale = next;
      this.offsetX = anchorX - wx * this.scale;
      this.offsetY = anchorY - wy * this.scale;
      this.autoFit = false;
      this.requestRender();
    }

    // ------------------------------------------------------------ selection

    setFocus(nodeId) {
      this.focusId = nodeId || null;
      if (!this.focusId) {
        this.focusSet = null;
      } else {
        const node = this.nodeById.get(this.focusId);
        if (!node) {
          this.focusSet = null;
          this.focusId = null;
        } else {
          this.focusSet = new Set([node.id, ...node.neighbors]);
        }
      }
      this.requestRender();
    }

    selectNode(nodeId, { center = false } = {}) {
      const node = nodeId ? this.nodeById.get(nodeId) : null;
      this.selectedNode = node || null;
      this.selectedEdge = null;
      if (node && center) this.centerOn(node.id, Math.max(this.scale, 0.9));
      this.requestRender();
      return !!node;
    }

    selectEdgeByKey(key) {
      this.selectedEdge = this.edges.find((e) => e.key === key) || null;
      if (this.selectedEdge) this.selectedNode = null;
      this.requestRender();
    }

    clearSelection() {
      this.selectedNode = null;
      this.selectedEdge = null;
      this.requestRender();
    }

    // ------------------------------------------------------------ hit tests

    nodeAt(worldX, worldY) {
      // Back to front, so the node drawn on top is the one picked. The
      // test is the plain point-in-circle the brief asks for; a memory's
      // rounded-rect body is slightly wider than its hit circle, which is
      // the right way round (no dead pixels inside the shape's core).
      for (let i = this.nodes.length - 1; i >= 0; i -= 1) {
        const n = this.nodes[i];
        const hit = Math.max(n.r, 6 / this.scale);
        if (Math.hypot(worldX - n.x, worldY - n.y) <= hit) return n;
      }
      return null;
    }

    edgeAt(worldX, worldY) {
      const slop = EDGE_HIT_PX / this.scale;
      let best = null;
      let bestD = Infinity;
      for (const e of this.edges) {
        const d = distToSegment(worldX, worldY, e.s.x, e.s.y, e.t.x, e.t.y);
        if (d < slop && d < bestD) {
          bestD = d;
          best = e;
        }
      }
      return best;
    }

    // -------------------------------------------------------------- events

    _pointerDown(ev) {
      if (ev.button !== 0) return;
      this.canvas.focus({ preventScroll: true });
      const w = this.toWorld(ev.clientX, ev.clientY);
      const node = this.nodeAt(w.x, w.y);
      try {
        this.canvas.setPointerCapture(ev.pointerId);
      } catch {
        // Capture is a nicety; dragging still works through the
        // document-level pointer events without it.
      }
      if (node) {
        this.drag = { node, moved: 0, startX: ev.clientX, startY: ev.clientY, wasPinned: node.pinned };
        this.canvas.style.cursor = "grabbing";
        this.reheat(0.3);
      } else {
        this.pan = { x: ev.clientX, y: ev.clientY, moved: 0 };
        this.canvas.style.cursor = "grabbing";
      }
    }

    _pointerMove(ev) {
      if (this.drag) {
        const w = this.toWorld(ev.clientX, ev.clientY);
        this.drag.node.x = w.x;
        this.drag.node.y = w.y;
        this.drag.node.vx = 0;
        this.drag.node.vy = 0;
        this.drag.moved = Math.max(
          this.drag.moved,
          Math.hypot(ev.clientX - this.drag.startX, ev.clientY - this.drag.startY)
        );
        this.autoFit = false;
        // Dragging a node invalidates the settled layout for everything
        // attached to it, so the simulation comes back to life.
        this.reheat(0.3);
        if (!this.running) this.requestRender();
        return;
      }
      if (this.pan) {
        this.offsetX += ev.clientX - this.pan.x;
        this.offsetY += ev.clientY - this.pan.y;
        this.pan.moved += Math.abs(ev.clientX - this.pan.x) + Math.abs(ev.clientY - this.pan.y);
        this.pan.x = ev.clientX;
        this.pan.y = ev.clientY;
        this.autoFit = false;
        this.requestRender();
        return;
      }
      const w = this.toWorld(ev.clientX, ev.clientY);
      const node = this.nodeAt(w.x, w.y);
      const edge = node ? null : this.edgeAt(w.x, w.y);
      if (node === this.hovered && edge === this.hoveredEdge) return;
      this.hovered = node;
      this.hoveredEdge = edge;
      this.canvas.style.cursor = node || edge ? "pointer" : "grab";
      this.requestRender();
      if (this.handlers.onHover) {
        this.handlers.onHover(
          node ? { type: "node", node } : edge ? { type: "edge", edge } : null,
          { clientX: ev.clientX, clientY: ev.clientY }
        );
      }
    }

    _pointerUp(ev) {
      try {
        this.canvas.releasePointerCapture(ev.pointerId);
      } catch {
        // See _pointerDown: capture is optional.
      }
      if (this.drag) {
        const { node, moved } = this.drag;
        this.drag = null;
        this.canvas.style.cursor = "pointer";
        if (moved <= CLICK_SLOP_PX) {
          // A press that did not move is a click, not a drag: select,
          // and leave the node's pinned state alone.
          this.selectedNode = node;
          this.selectedEdge = null;
          this.requestRender();
          if (this.handlers.onSelectNode) this.handlers.onSelectNode(node);
        } else {
          // Dropped somewhere deliberately: pin it there. Double-click
          // releases it again.
          node.pinned = true;
          this.requestRender();
          if (this.handlers.onPin) this.handlers.onPin(node);
        }
        return;
      }
      if (this.pan) {
        const moved = this.pan.moved;
        this.pan = null;
        this.canvas.style.cursor = "grab";
        if (moved <= CLICK_SLOP_PX) {
          const w = this.toWorld(ev.clientX, ev.clientY);
          const edge = this.edgeAt(w.x, w.y);
          if (edge) {
            this.selectedEdge = edge;
            this.selectedNode = null;
            this.requestRender();
            if (this.handlers.onSelectEdge) this.handlers.onSelectEdge(edge);
          } else {
            this.selectedEdge = null;
            this.selectedNode = null;
            this.requestRender();
            if (this.handlers.onSelectNone) this.handlers.onSelectNone();
          }
        }
      }
    }

    _pointerLeave() {
      if (this.hovered || this.hoveredEdge) {
        this.hovered = null;
        this.hoveredEdge = null;
        this.requestRender();
        if (this.handlers.onHover) this.handlers.onHover(null, null);
      }
    }

    _wheel(ev) {
      ev.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const factor = Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0016));
      this._zoomBy(factor, ev.clientX - rect.left, ev.clientY - rect.top);
    }

    _dblClick(ev) {
      const w = this.toWorld(ev.clientX, ev.clientY);
      const node = this.nodeAt(w.x, w.y);
      if (node && node.pinned) {
        node.pinned = false;
        this.reheat(0.4);
        if (this.handlers.onPin) this.handlers.onPin(node);
      }
    }

    // The canvas is mouse-first by design (the entity search in
    // explorer.js is the keyboard route to the same information), but pan
    // and zoom are cheap to offer from the keyboard once it is focusable,
    // so a keyboard user is not stuck with whatever the initial fit was.
    _keyDown(ev) {
      const stepPx = ev.shiftKey ? 120 : 40;
      let handled = true;
      switch (ev.key) {
        case "ArrowLeft": this.offsetX += stepPx; this.autoFit = false; break;
        case "ArrowRight": this.offsetX -= stepPx; this.autoFit = false; break;
        case "ArrowUp": this.offsetY += stepPx; this.autoFit = false; break;
        case "ArrowDown": this.offsetY -= stepPx; this.autoFit = false; break;
        case "+":
        case "=": this._zoomBy(1.2, this.width / 2, this.height / 2); break;
        case "-":
        case "_": this._zoomBy(1 / 1.2, this.width / 2, this.height / 2); break;
        case "0": this.resetView(); break;
        default: handled = false;
      }
      if (handled) {
        ev.preventDefault();
        this.requestRender();
      }
    }

    // -------------------------------------------------------------- drawing

    _paint() {
      const ctx = this.ctx;
      if (!ctx) return;
      this.frames += 1;
      const p = this.palette;
      // All drawing coordinates are CSS pixels; this one transform maps
      // them onto the devicePixelRatio-scaled backing store.
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.width, this.height);
      ctx.fillStyle = p.bg;
      ctx.fillRect(0, 0, this.width, this.height);
      if (!this.nodes.length) return;

      const sx = (x) => x * this.scale + this.offsetX;
      const sy = (y) => y * this.scale + this.offsetY;
      const focus = this.focusSet;
      const sel = this.selectedNode;
      const hov = this.hovered;
      const emphasisId = (sel && sel.id) || (hov && hov.id) || null;

      // --- edges
      ctx.lineCap = "round";
      for (const e of this.edges) {
        const highlighted =
          e === this.selectedEdge ||
          e === this.hoveredEdge ||
          (emphasisId && (e.s.id === emphasisId || e.t.id === emphasisId));
        const dimmedByFocus = focus && !(focus.has(e.s.id) && focus.has(e.t.id));
        const dimmedBySelection = emphasisId && !highlighted;
        const style = edgeStyle(p, e);
        ctx.globalAlpha = highlighted ? 1 : dimmedByFocus ? 0.08 : dimmedBySelection ? 0.18 : 0.5;
        ctx.strokeStyle = highlighted ? p.edgeHighlight : style.color;
        ctx.lineWidth = highlighted ? style.width + 1.1 : style.width;
        // The dash pattern is what separates an inferred link from a
        // stated one, so it survives highlighting.
        if (style.dash) ctx.setLineDash(style.dash);
        else ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(sx(e.s.x), sy(e.s.y));
        ctx.lineTo(sx(e.t.x), sy(e.t.y));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // --- nodes
      for (const n of this.nodes) {
        const dimmed = focus && !focus.has(n.id);
        const r = Math.max(2.5, n.r * this.scale);
        const x = sx(n.x);
        const y = sy(n.y);
        if (x < -40 || y < -40 || x > this.width + 40 || y > this.height + 40) continue;
        ctx.globalAlpha = dimmed ? 0.16 : 1;
        const color = kindColor(p, n);
        shapePath(ctx, x, y, r, n.shape);
        if (n.shape === "ring") {
          // "concept" is drawn hollow -- one more non-colour channel, and
          // concepts are the most numerous entity kind, so the lightest
          // mark is the right one for them.
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(1.2, r * 0.3);
          ctx.stroke();
        } else {
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = p.nodeOutline;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        if (n.pinned) {
          // Pinned where the user dropped it: a thin outer ring, so the
          // state is visible rather than something you have to remember.
          ctx.strokeStyle = p.pinRing;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(x, y, r + 3.5, 0, TAU);
          ctx.stroke();
        }
        if (n === sel || n === hov) {
          ctx.strokeStyle = p.selectRing;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, r + (n.pinned ? 6.5 : 5), 0, TAU);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // --- labels, last, so nothing paints over them
      ctx.font = `500 11px ${p.font}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const n of this.nodes) {
        const dimmed = focus && !focus.has(n.id);
        if (dimmed) continue;
        const wanted =
          n === sel ||
          n === hov ||
          n.showLabel ||
          this.scale > 1.5 ||
          (focus && focus.has(n.id)) ||
          (this.selectedEdge && (this.selectedEdge.s === n || this.selectedEdge.t === n));
        if (!wanted) continue;
        const r = Math.max(2.5, n.r * this.scale);
        const x = sx(n.x);
        const y = sy(n.y) + r + 9;
        if (x < -80 || y < 0 || x > this.width + 80 || y > this.height) continue;
        const text = truncate(n.label, n === sel || n === hov ? 52 : LABEL_CHARS);
        // A halo rather than a filled plate: keeps the edge underneath
        // readable while making the type legible against it.
        ctx.lineWidth = 3;
        ctx.strokeStyle = p.labelHalo;
        ctx.strokeText(text, x, y);
        ctx.fillStyle = n === sel || n === hov ? p.labelText : p.labelText;
        ctx.globalAlpha = n === sel || n === hov || n.showLabel ? 1 : 0.75;
        ctx.fillText(text, x, y);
        ctx.globalAlpha = 1;
      }

      // --- the hovered or selected edge's label, drawn at its midpoint:
      // the "why are these connected?" answer starts here and continues in
      // the inspector.
      const labelEdge = this.selectedEdge || this.hoveredEdge;
      if (labelEdge && labelEdge.label) {
        const mx = (sx(labelEdge.s.x) + sx(labelEdge.t.x)) / 2;
        const my = (sy(labelEdge.s.y) + sy(labelEdge.t.y)) / 2;
        const text = truncate(labelEdge.label, 34);
        ctx.font = `500 11px ${p.font}`;
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = p.labelHalo;
        ctx.strokeText(text, mx, my);
        ctx.fillStyle = p.edgeHighlight;
        ctx.fillText(text, mx, my);
      }
    }
  }

  window.membase.GraphCanvas = GraphCanvas;
  window.membase.GRAPH_SHAPES = SHAPES;
})();
