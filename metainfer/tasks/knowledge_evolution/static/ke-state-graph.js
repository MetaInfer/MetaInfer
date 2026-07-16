// SVG state-machine graph for knowledge-evolution.
// Renders the 4-phase state machine (A→B→C→D) with highlighted current phase
// and active transition edge. Geometry is derived from the graph_payload()
// node/edge lists the backend sends — no hardcoded positions.

import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { setPhaseLabels } from "app/utils";

const SVG_NS = "http://www.w3.org/2000/svg";
const GEO = {
  nodeW: 140, nodeH: 42,
  gapX: 12,
  arcClearance: 50,
  belowClearance: 28,
  labelSpace: 28,
  margin: 12,
};
const COLORS = {
  forward: "#8b949e", pass: "#3fb950", fail: "#d29922",
  backward: "#d29922", self: "#a371f7", abort: "#f85149",
};

function edgeClass(e, nodeOrder) {
  if (e.to === "finished") {
    const lbl = (e.label || "").toLowerCase();
    if (lbl.includes("ok") || lbl.includes("pass") || lbl.includes("verified") || lbl.includes("sufficient")) return "pass";
    return "fail";
  }
  if (e.from === e.to) return "self";
  const i = nodeOrder.indexOf(e.from);
  const j = nodeOrder.indexOf(e.to);
  if (i === -1 || j === -1) return "forward";
  if (j < i) return "backward";
  const lbl = (e.label || "").toLowerCase();
  if (lbl.includes("ok") || lbl.includes("pass") || lbl.includes("consolidated") || lbl.includes("verify")) return "pass";
  if (lbl.includes("fail") || lbl.includes("retry") || lbl.includes("re-enrich")) return "fail";
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
  const belowApexY = mainBottomY + GEO.belowClearance;
  const H = belowApexY + GEO.labelSpace + GEO.margin;
  const mainX = {};
  nodes.forEach((node, i) => {
    mainX[node.id] = GEO.margin + GEO.nodeW / 2 + i * (GEO.nodeW + GEO.gapX);
  });
  return { W, H, mainX, mainY, mainBottomY, band: { selfLoopTop: arcTop + 18 }, belowApexY };
}

function nodeAnchor(id, L) {
  const cx = L.mainX[id], cy = L.mainY, hw = GEO.nodeW / 2, hh = GEO.nodeH / 2;
  return { cx, cy, top: cy - hh, bottom: cy + hh, left: cx - hw, right: cx + hw };
}

function elbowPath(cls, fromId, toId, nodeOrder, L) {
  const s = nodeAnchor(fromId, L), t = nodeAnchor(toId, L);
  if (cls === "self") {
    const cx = s.cx, topY = s.top, w = 30, h = topY - L.band.selfLoopTop;
    return `M ${cx + w / 2} ${topY} L ${cx + w / 2} ${topY - h} L ${cx - w / 2} ${topY - h} L ${cx - w / 2} ${topY}`;
  }
  if (cls === "backward") {
    // Route below the nodes so it doesn't overlap with self-loop arrows above
    return `M ${s.cx} ${s.bottom} L ${s.cx} ${L.belowApexY} L ${t.cx} ${L.belowApexY} L ${t.cx} ${t.bottom}`;
  }
  return `M ${s.right} ${s.cy} L ${t.left} ${s.cy}`;
}

function buildSvg(graph) {
  const mainNodes = graph.nodes || [];
  const edges = graph.edges || [];
  const nodeOrder = mainNodes.map((n) => n.id);
  const L = layout(mainNodes);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${L.W} ${L.H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
  svg.style.maxHeight = `${L.H}px`;

  const defs = document.createElementNS(SVG_NS, "defs");
  for (const [cls, color] of Object.entries(COLORS)) {
    const m = document.createElementNS(SVG_NS, "marker");
    m.setAttribute("id", `ke-arrow-${cls}`);
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
    path.setAttribute("marker-end", `url(#ke-arrow-${cls})`);
    path.setAttribute("fill", "none");
    path.setAttribute("d", elbowPath(cls, e.from, e.to, nodeOrder, L));
    svg.appendChild(path);
  }

  for (const n of mainNodes) {
    const cx = L.mainX[n.id], cy = L.mainY;
    const isActive = n.id === graph.current;
    const g = document.createElementNS(SVG_NS, "g");
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("class", ["node-rect", isActive ? "active" : ""].filter(Boolean).join(" "));
    rect.setAttribute("x", cx - GEO.nodeW / 2);
    rect.setAttribute("y", cy - GEO.nodeH / 2);
    rect.setAttribute("width", GEO.nodeW);
    rect.setAttribute("height", GEO.nodeH);
    rect.setAttribute("rx", "8");
    g.appendChild(rect);
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("class", "node-label");
    t.setAttribute("x", cx);
    t.setAttribute("y", cy + 5);
    t.textContent = n.label || n.id;
    g.appendChild(t);
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

  return html`<div class="state-graph" ref=${ref}></div>`;
}
