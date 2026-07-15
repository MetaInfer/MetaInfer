// SVG state-machine graph. Ported from the legacy dashboard but
// reimplemented as a Preact component using htm templates.
//
// All connectors are orthogonal elbows. Geometry is derived from a few
// base dimensions + the node list the backend sends — no per-node
// hardcoded x or viewBox. If a phase is added in phases.py, the diagram
// resizes itself.

import { html } from "htm/preact";
import { useEffect, useMemo } from "preact/hooks";
import { setPhaseLabels, labelFor, escapeHtml } from "app/utils";

const SVG_NS = "http://www.w3.org/2000/svg";
const GEO = {
  nodeW: 150, nodeH: 50,
  gapX: 20,
  arcClearance: 110,
  labelSpace: 50,
  margin: 20,
};
const COLORS = {
  forward: "#8b949e", pass: "#3fb950", fail: "#d29922",
  backward: "#d29922", self: "#a371f7", abort: "#f85149",
};

function edgeClass(e, nodeOrder) {
  if (e.to === "failed" || e.from === "failed") return "abort";
  if (e.from === e.to) return "self";
  const i = nodeOrder.indexOf(e.from);
  const j = nodeOrder.indexOf(e.to);
  if (i === -1 || j === -1) return "forward";
  if (j < i) return "backward";
  const lbl = (e.label || "").toLowerCase();
  if (lbl === "ok" || lbl === "pass" || lbl === "c ok → perf") return "pass";
  if (lbl.includes("fail") || lbl.includes("redo") || lbl.includes("replan")) return "fail";
  return "forward";
}

function layout(nodes) {
  const n = nodes.length;
  const innerW = n * GEO.nodeW + Math.max(0, n - 1) * GEO.gapX;
  const W = 2 * GEO.margin + innerW;
  const arcTop = GEO.margin;
  const mainTopY = arcTop + GEO.arcClearance;
  const mainY = mainTopY + GEO.nodeH / 2;
  const mainBottomY = mainTopY + GEO.nodeH;
  const H = mainBottomY + GEO.labelSpace + GEO.margin;
  const mainX = {};
  nodes.forEach((node, i) => {
    mainX[node.id] = GEO.margin + GEO.nodeW / 2 + i * (GEO.nodeW + GEO.gapX);
  });
  return {
    W, H, mainX, mainY,
    band: {
      backwardHigh: arcTop + 15,
      backwardLow: arcTop + 60,
      selfLoopTop: arcTop + 30,
      forwardLabel: mainBottomY + 18,
    },
  };
}

function nodeAnchor(id, L) {
  const cx = L.mainX[id];
  const cy = L.mainY;
  const hw = GEO.nodeW / 2, hh = GEO.nodeH / 2;
  return {
    cx, cy,
    top: cy - hh, bottom: cy + hh,
    left: cx - hw, right: cx + hw,
  };
}

function elbowPath(cls, fromId, toId, nodeOrder, L) {
  const s = nodeAnchor(fromId, L);
  const t = nodeAnchor(toId, L);
  const band = L.band;
  if (cls === "self") {
    const cx = s.cx;
    const topY = s.top;
    const w = 30, h = topY - band.selfLoopTop;
    const xR = cx + w / 2, xL = cx - w / 2, yT = topY - h;
    return `M ${xR} ${topY} L ${xR} ${yT} L ${xL} ${yT} L ${xL} ${topY}`;
  }
  if (cls === "backward") {
    const hops = Math.abs(nodeOrder.indexOf(fromId) - nodeOrder.indexOf(toId));
    const apexY = hops >= 3 ? band.backwardHigh : band.backwardLow;
    return `M ${s.cx} ${s.top} L ${s.cx} ${apexY} L ${t.cx} ${apexY} L ${t.cx} ${t.top}`;
  }
  return `M ${s.right} ${s.cy} L ${t.left} ${s.cy}`;
}

function labelPos(cls, fromId, toId, nodeOrder, L) {
  const s = nodeAnchor(fromId, L);
  const t = nodeAnchor(toId, L);
  const band = L.band;
  if (cls === "backward") {
    const hops = Math.abs(nodeOrder.indexOf(fromId) - nodeOrder.indexOf(toId));
    const apexY = hops >= 3 ? band.backwardHigh : band.backwardLow;
    return { x: (s.cx + t.cx) / 2, y: apexY - 8 };
  }
  return { x: (s.right + t.left) / 2, y: band.forwardLabel };
}

function splitNodeLabel(label) {
  const idx = label.indexOf(":");
  if (idx === -1) return [label];
  return [label.slice(0, idx), label.slice(idx + 1).trim()];
}

// Build the SVG element tree imperatively. Preact's SVG support is decent
// but the elbow geometry is already imperative — rendering the SVG once
// per props change via createElementNS is simpler than spelling every
// attribute out in htm.
function buildSvg(graph) {
  const mainNodes = graph.nodes || [];
  const edges = graph.edges || [];
  const nodeOrder = mainNodes.map((n) => n.id);
  const L = layout(mainNodes);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${L.W} ${L.H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  // Arrow markers, one per color.
  const defs = document.createElementNS(SVG_NS, "defs");
  for (const [cls, color] of Object.entries(COLORS)) {
    const m = document.createElementNS(SVG_NS, "marker");
    m.setAttribute("id", `arrow-${cls}`);
    m.setAttribute("viewBox", "0 0 10 10");
    m.setAttribute("refX", "9");
    m.setAttribute("refY", "5");
    m.setAttribute("markerWidth", "8");
    m.setAttribute("markerHeight", "8");
    m.setAttribute("orient", "auto-start-reverse");
    const p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    p.setAttribute("fill", color);
    m.appendChild(p);
    defs.appendChild(m);
  }
  svg.appendChild(defs);

  // Filter out abort edges (X→failed). They're noisy; the run-level
  // failure is already surfaced elsewhere.
  const visible = edges.filter((e) => edgeClass(e, nodeOrder) !== "abort");

  for (const e of visible) {
    const cls = edgeClass(e, nodeOrder);
    const isActive = graph.active_edge
      && graph.active_edge.from === e.from
      && graph.active_edge.to === e.to
      && e.label.split(" / ").includes(graph.active_edge.label);
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("class", `edge ${cls}${isActive ? " active" : ""}`);
    path.setAttribute("stroke", COLORS[cls]);
    path.setAttribute("marker-end", `url(#arrow-${cls})`);
    path.setAttribute("fill", "none");
    path.setAttribute("d", elbowPath(cls, e.from, e.to, nodeOrder, L));
    svg.appendChild(path);

    if (cls === "forward" || cls === "pass" || cls === "fail" || cls === "backward") {
      const lp = labelPos(cls, e.from, e.to, nodeOrder, L);
      const txt = document.createElementNS(SVG_NS, "text");
      txt.setAttribute("class", `edge-label${isActive ? " active" : ""}`);
      txt.setAttribute("x", lp.x);
      txt.setAttribute("y", lp.y);
      txt.textContent = e.label;
      svg.appendChild(txt);
    }
  }

  // Main-row nodes on top so arrowheads terminate cleanly.
  for (const n of mainNodes) {
    const cx = L.mainX[n.id];
    const cy = L.mainY;
    const isActive = n.id === graph.current;
    const g = document.createElementNS(SVG_NS, "g");
    const rect = document.createElementNS(SVG_NS, "rect");
    const cls = ["node-rect", isActive ? "active" : ""].filter(Boolean).join(" ");
    rect.setAttribute("class", cls);
    rect.setAttribute("x", cx - GEO.nodeW / 2);
    rect.setAttribute("y", cy - GEO.nodeH / 2);
    rect.setAttribute("width", GEO.nodeW);
    rect.setAttribute("height", GEO.nodeH);
    rect.setAttribute("rx", "8");
    g.appendChild(rect);
    const lines = splitNodeLabel(n.label);
    lines.forEach((line, i) => {
      const y = cy + (i - (lines.length - 1) / 2) * 16;
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("class", "node-label");
      t.setAttribute("x", cx);
      t.setAttribute("y", y);
      t.textContent = line;
      g.appendChild(t);
    });
    if (isActive) {
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("class", "node-dot");
      dot.setAttribute("cx", cx - GEO.nodeW / 2 + 10);
      dot.setAttribute("cy", cy - GEO.nodeH / 2 + 10);
      dot.setAttribute("r", "4");
      g.appendChild(dot);
    }
    svg.appendChild(g);
  }
  return svg;
}

export function StateGraph({ graph }) {
  // Cache labels for other panels.
  useEffect(() => {
    if (!graph) return;
    setPhaseLabels(graph.nodes || []);
    setPhaseLabels(graph.terminal_nodes || []);
  }, [graph]);

  const ref = (el) => {
    if (!el) return;
    el.innerHTML = "";
    if (graph && (graph.nodes || []).length) {
      el.appendChild(buildSvg(graph));
    }
  };

  const caption = (() => {
    if (!graph) return null;
    if (graph.error) {
      return html`<span class="text-err">⚠ ${graph.error}</span>`;
    }
    const cur = graph.current;
    if (cur === "finished") return "Run finished.";
    if (cur === "failed") return "Run ended in failure.";
    if (cur === "idle") return "Not started.";
    if (graph.active_edge) {
      const ae = graph.active_edge;
      const outcome = graph.last_outcome ? ` (outcome: ${graph.last_outcome})` : "";
      // Flat single-line template — avoid nested <span> which trips up
      // htm's child-stack bookkeeping. Build the label as a plain string
      // and pass it through a single ${...} interpolation.
      const text = `last transition: ${ae.from} —[${ae.label}]→ ${ae.to}${outcome}`;
      return html`<span>${text}</span>`;
    }
    return `Current phase: ${labelFor(cur)}`;
  })();

  const placeholder = !graph
    ? html`<div class="state-graph-placeholder"><p class="muted">Loading state graph…</p></div>`
    : (graph.error
      ? html`<div class="state-graph-placeholder"><p class="text-err">⚠ ${graph.error}</p></div>`
      : null);

  return html`<div class="state-graph" ref=${ref}>${placeholder}</div>${caption ? html`<div class="state-graph-caption">${caption}</div>` : null}`;
}
