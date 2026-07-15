// Perf-per-iteration mini-chart tiles. Lazily creates one Chart.js line
// chart per metric (and one for iteration duration), reusing them across
// updates; drops tiles for metrics that disappear from the payload.

import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";
import { Chart, registerables } from "chart.js";

// The tree-shakeable ESM build of Chart.js v4 doesn't auto-register
// scales / controllers / elements — calling `new Chart(...)` without
// this throws "linear is not a registered scale". Register everything.
Chart.register(...registerables);

const PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f778ba", "#a371f7", "#79c0ff"];

function linearAxisOptions() {
  return {
    type: "linear",
    ticks: {
      color: "#8b949e",
      precision: 0,
      stepSize: 1,
      callback: (v) => (Number.isInteger(v) ? `iter ${v}` : ""),
    },
    grid: { color: "#21262d" },
    min: 1,
  };
}

function miniChartOptions(color, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: linearAxisOptions(),
      y: {
        ticks: { color: "#8b949e" },
        grid: { color: "#21262d" },
        beginAtZero: true,
        title: {
          display: !!yLabel,
          text: yLabel || "",
          color: "#8b949e",
        },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { title: (items) => `iter ${items[0].parsed.x}` } },
    },
  };
}

export function Charts({ payload }) {
  const gridRef = useRef(null);
  // key -> { chart, wrap }
  const chartsRef = useRef({});

  // Drop everything on unmount.
  useEffect(() => {
    const charts = chartsRef.current;
    return () => {
      for (const c of Object.values(charts)) c.chart.destroy();
      chartsRef.current = {};
    };
  }, []);

  useEffect(() => {
    if (!payload || !gridRef.current) return;
    const charts = chartsRef.current;
    const grid = gridRef.current;

    // Defensive: drop any cached chart whose wrap <div> is no longer a
    // child of the current grid. This happens when Preact unmounts and
    // remounts the component (e.g. on task switch), leaving us with
    // dangling references to detached DOM nodes that won't render.
    for (const k of Object.keys(charts)) {
      const wrap = charts[k].wrap;
      if (!wrap || !grid.contains(wrap)) {
        try { charts[k].chart.destroy(); } catch (_) { /* noop */ }
        delete charts[k];
      }
    }

    // Also drop any orphan .mini-chart nodes in the grid that we lost
    // track of (e.g. a previous render's leftovers after a hot reload).
    for (const orphan of Array.from(grid.querySelectorAll(".mini-chart"))) {
      const key = orphan.dataset.chartKey;
      if (!key || !charts[key]) orphan.remove();
    }

    const wanted = new Set(["duration"]);
    (payload.perf_series || []).forEach((s) => wanted.add(s.metric));

    // Drop unwanted tiles (metric disappeared from payload).
    for (const k of Object.keys(charts)) {
      if (!wanted.has(k)) {
        charts[k].chart.destroy();
        charts[k].wrap.remove();
        delete charts[k];
      }
    }

    function ensure(key, title, color, yLabel) {
      if (charts[key]) return charts[key].chart;
      const wrap = document.createElement("div");
      wrap.className = "mini-chart";
      wrap.dataset.chartKey = key;
      wrap.innerHTML = `<h3></h3><div class="chart-box-small"><canvas></canvas></div>`;
      wrap.querySelector("h3").textContent = title;
      grid.appendChild(wrap);
      const ctx = wrap.querySelector("canvas").getContext("2d");
      const chart = new Chart(ctx, {
        type: "line",
        data: {
          datasets: [{
            label: title,
            data: [],
            borderColor: color,
            backgroundColor: color + "33",
            tension: 0.2,
            fill: false,
            pointRadius: 3,
          }],
        },
        options: miniChartOptions(color, yLabel),
      });
      charts[key] = { chart, wrap };
      return chart;
    }

    const dur = ensure("duration", "Iteration duration (s)", "#d29922", "seconds");
    dur.data.datasets[0].data = (payload.durations || []).map(
      (d) => ({ x: d.x, y: d.y }),
    );
    dur.update();

    (payload.perf_series || []).forEach((s, i) => {
      const color = PALETTE[i % PALETTE.length];
      const ch = ensure(s.metric, s.metric, color);
      ch.data.datasets[0].data = (s.points || []).map(
        (p) => ({ x: p.x, y: p.y }),
      );
      ch.update();
    });
  }, [payload]);

  const empty = !payload
    || (!payload.durations?.length && !payload.perf_series?.length);

  return html`
    <div class="charts-panel">
      ${empty
        ? html`<p class="muted">No chart data yet. Tiles appear as iterations
            complete and report metrics.</p>`
        : null}
      <div class="charts-grid" ref=${gridRef}></div>
    </div>
  `;
}
