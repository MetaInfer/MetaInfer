// SVG state-machine graph for evolve-kernel 8-phase flow.
// Preact component using htm — follows the same pattern as cpp-gf-state-graph.js.
// Layout: two rows — bootstrap phases (A-D) on top, optimization loop (E-H) below.

import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { setPhaseLabels, labelFor } from "app/utils";

const SVG_NS = "http://www.w3.org/2000/svg";

// Phase labels for the graph
const PHASE_LABELS = {
  "A_gen_correctness_harness": "A: Gen\nCorrectness Harness",
  "B_review_correctness_harness": "B: Review\nCorrectness Harness",
  "C_gen_perf_harness": "C: Gen\nPerf Harness",
  "D_review_perf_harness": "D: Review\nPerf Harness",
  "E_select_kernel": "E: Select\nKernel",
  "F_optimize": "F: Optimize\nKernel",
  "G_verify_correctness": "G: Verify\nCorrectness",
  "H_measure_perf": "H: Measure\nPerf + Update",
};

const BOOTSTRAP = ["A_gen_correctness_harness", "B_review_correctness_harness", "C_gen_perf_harness", "D_review_perf_harness"];
const OPTIMIZE = ["E_select_kernel", "F_optimize", "G_verify_correctness", "H_measure_perf"];

// Layout constants
const NODE_W = 130, NODE_H = 50, GAP_X = 24, GAP_Y = 80;
const MARGIN = 24;

function layout(nodes) {
  const row1 = nodes.filter(n => BOOTSTRAP.includes(n.id));
  const row2 = nodes.filter(n => OPTIMIZE.includes(n.id));

  const maxCols = Math.max(row1.length, row2.length);
  const totalW = maxCols * NODE_W + (maxCols - 1) * GAP_X;
  const W = 2 * MARGIN + totalW;
  const H = 2 * MARGIN + 2 * NODE_H + GAP_Y + 60;

  const row1Y = MARGIN + NODE_H / 2;
  const row2Y = MARGIN + NODE_H + GAP_Y + NODE_H / 2;
  const labelY = H - 20;

  const pos = {};
  row1.forEach((n, i) => {
    pos[n.id] = { x: MARGIN + NODE_W / 2 + i * (NODE_W + GAP_X), y: row1Y, row: 1 };
  });
  row2.forEach((n, i) => {
    pos[n.id] = { x: MARGIN + NODE_W / 2 + i * (NODE_W + GAP_X), y: row2Y, row: 2 };
  });

  return { W, H, pos, labelY };
}

function nodeRect(id, cx, cy, isActive) {
  const x = cx - NODE_W / 2;
  const y = cy - NODE_H / 2;
  const label = PHASE_LABELS[id] || id;
  const lines = label.split("\n");

  const g = document.createElementNS(SVG_NS, "g");
  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  rect.setAttribute("width", NODE_W);
  rect.setAttribute("height", NODE_H);
  rect.setAttribute("rx", "8");
  rect.setAttribute("class", `node-rect${isActive ? " active" : ""}`);
  g.appendChild(rect);

  lines.forEach((line, i) => {
    const ty = cy + (i - (lines.length - 1) / 2) * 14;
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("class", "node-label");
    t.setAttribute("x", cx);
    t.setAttribute("y", ty);
    t.setAttribute("text-anchor", "middle");
    t.textContent = line;
    g.appendChild(t);
  });

  if (isActive) {
    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("class", "node-dot");
    dot.setAttribute("cx", x + 8);
    dot.setAttribute("cy", y + 8);
    dot.setAttribute("r", "4");
    g.appendChild(dot);
  }

  return g;
}

function arrowDefs() {
  const defs = document.createElementNS(SVG_NS, "defs");
  const colors = { forward: "#8b949e", pass: "#3fb950", fail: "#d29922", self: "#a371f7" };
  for (const [cls, color] of Object.entries(colors)) {
    const m = document.createElementNS(SVG_NS, "marker");
    m.setAttribute("id", `arrow-${cls}`);
    m.setAttribute("viewBox", "0 0 10 10");
    m.setAttribute("refX", "9");
    m.setAttribute("refY", "5");
    m.setAttribute("markerWidth", "6");
    m.setAttribute("markerHeight", "6");
    m.setAttribute("orient", "auto-start-reverse");
    const p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    p.setAttribute("fill", color);
    m.appendChild(p);
    defs.appendChild(m);
  }
  return defs;
}

function drawEdge(fromPos, toPos, cls) {
  const path = document.createElementNS(SVG_NS, "path");
  const sx = fromPos.x, sy = fromPos.y;
  const tx = toPos.x, ty = toPos.y;

  // Auto-route
  const midY = (fromPos.row === 1 && toPos.row === 2) ? sy + (ty - sy) / 2 : sy - 20;

  if (fromPos.row === 1 && toPos.row === 2 && sx === tx) {
    // Vertical drop from bootstrap to optimize
    path.setAttribute("d", `M ${sx} ${sy + NODE_H / 2} L ${sx} ${midY} L ${tx} ${midY} L ${tx} ${ty - NODE_H / 2}`);
  } else if (fromPos.row === 2 && toPos.row === 1) {
    // Loop back (H→E)
    const apexY = ty - 40;
    path.setAttribute("d", `M ${sx} ${sy - NODE_H / 2} L ${sx} ${apexY} L ${tx} ${apexY} L ${tx} ${ty - NODE_H / 2}`);
  } else if (fromPos.row === toPos.row && sx < tx) {
    // Horizontal forward
    path.setAttribute("d", `M ${sx + NODE_W / 2} ${sy} L ${tx - NODE_W / 2} ${ty}`);
  } else if (fromPos.row === toPos.row && sx > tx) {
    // Horizontal backward
    const apexY = sy - (fromPos.row === 2 ? 50 : 40);
    path.setAttribute("d", `M ${sx + NODE_W / 2} ${sy} L ${sx + NODE_W / 2 + 10} ${sy} L ${sx + NODE_W / 2 + 10} ${apexY} L ${tx - NODE_W / 2 - 10} ${apexY} L ${tx - NODE_W / 2 - 10} ${ty} L ${tx - NODE_W / 2} ${ty}`);
  } else if (fromPos.row === toPos.row && sx === tx) {
    // Self-loop
    const cx = sx + NODE_W / 2 + 15;
    path.setAttribute("d", `M ${sx + NODE_W / 2} ${sy - 8} L ${cx} ${sy - 8} L ${cx} ${sy + 8} L ${sx + NODE_W / 2} ${sy + 8}`);
  } else {
    // Diagonal
    path.setAttribute("d", `M ${sx} ${sy + NODE_H / 2} L ${sx} ${sy + 30} L ${tx} ${ty - 30} L ${tx} ${ty - NODE_H / 2}`);
  }

  path.setAttribute("stroke", cls === "pass" ? "#3fb950" : cls === "fail" ? "#d29922" : "#8b949e");
  path.setAttribute("fill", "none");
  path.setAttribute("marker-end", `url(#arrow-${cls})`);
  path.setAttribute("class", `edge ${cls}`);
  return path;
}

function buildSvg(graph) {
  const mainNodes = graph.nodes || [];
  const edges = graph.edges || [];
  const L = layout(mainNodes);
  const nodeOrder = mainNodes.map(n => n.id);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${L.W} ${L.H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.appendChild(arrowDefs());

  // Draw edges (subset of key transitions for visual clarity)
  for (const e of edges) {
    const fromIdx = nodeOrder.indexOf(e.from);
    const toIdx = nodeOrder.indexOf(e.to);
    if (fromIdx === -1 || toIdx === -1) continue;
    const fromPos = L.pos[e.from];
    const toPos = L.pos[e.to];
    if (!fromPos || !toPos) continue;

    const isActive = graph.active_edge
      && graph.active_edge.from === e.from
      && graph.active_edge.to === e.to;

    let cls = "forward";
    if (toIdx < fromIdx) cls = "fail";
    else if (e.label && e.label.includes("pass")) cls = "pass";
    else if (fromIdx === toIdx) cls = "self";

    svg.appendChild(drawEdge(fromPos, toPos, cls));
  }

  // Draw nodes (on top)
  for (const n of mainNodes) {
    const p = L.pos[n.id];
    if (!p) continue;
    svg.appendChild(nodeRect(n.id, p.x, p.y, n.id === graph.current));
  }

  // Row labels
  const addLabel = (text, x, y) => {
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", x);
    t.setAttribute("y", y);
    t.setAttribute("class", "muted");
    t.setAttribute("font-size", "11");
    t.setAttribute("text-anchor", "middle");
    t.textContent = text;
    svg.appendChild(t);
  };

  // Bootstrap label
  const bNodes = mainNodes.filter(n => BOOTSTRAP.includes(n.id));
  if (bNodes.length > 0) {
    const bx = L.pos[bNodes[0].id].x;
    addLabel("Bootstrap (runs once)", bx - NODE_W / 2 - 10, L.pos[bNodes[0].id].y + NODE_H / 2 + 14);
  }

  // Optimization label
  const oNodes = mainNodes.filter(n => OPTIMIZE.includes(n.id));
  if (oNodes.length > 0) {
    const ox = L.pos[oNodes[0].id].x;
    addLabel("Optimization Loop", ox - NODE_W / 2 - 10, L.pos[oNodes[0].id].y + NODE_H / 2 + 14);
  }

  return svg;
}

export function StateGraph({ graph }) {
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
    if (!graph) return html`<span class="muted">Loading state graph…</span>`;
    if (graph.error) return html`<span class="text-err">⚠ ${graph.error}</span>`;
    const cur = graph.current;
    if (cur === "finished") return "Run finished.";
    if (cur === "idle") return "Not started.";
    const label = PHASE_LABELS[cur] || cur;
    return html`<span>Current: <strong>${label.replace(/\n/g, " ")}</strong></span>`;
  })();

  return html`
    <div class="state-graph" ref=${ref}></div>
    ${caption ? html`<div class="state-graph-caption">${caption}</div>` : null}
  `;
}
