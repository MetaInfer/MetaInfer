// Per-iteration mini-chart tiles for knowledge-evolution tasks.
// Renders Chart.js line charts for oracle perf metrics
// (e.g. oracle_cases_passed) using vendored Chart.js v4.4.1.

import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";
import { Chart, registerables } from "chart.js";

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

export function Charts({ payload, iterations }) {
  const gridRef = useRef(null);
  const chartsRef = useRef({});

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

    // Drop detached charts (e.g. after hot reload / task switch).
    for (const k of Object.keys(charts)) {
      const wrap = charts[k].wrap;
      if (!wrap || !grid.contains(wrap)) {
        try { charts[k].chart.destroy(); } catch (_) { /* noop */ }
        delete charts[k];
      }
    }
    // Clean orphan DOM nodes.
    for (const orphan of Array.from(grid.querySelectorAll(".mini-chart"))) {
      const key = orphan.dataset.chartKey;
      if (!key || !charts[key]) orphan.remove();
    }

    const wanted = new Set();
    (payload.perf_series || []).forEach((s) => wanted.add(s.metric));

    // Drop tiles for disappeared metrics.
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

    (payload.perf_series || []).forEach((s, i) => {
      const color = PALETTE[i % PALETTE.length];
      const ch = ensure(s.metric, s.metric, color);
      ch.data.datasets[0].data = (s.points || []).map(
        (p) => ({ x: p.x, y: p.y }),
      );
      ch.update();
    });
  }, [payload]);

  const hasCharts = payload && payload.perf_series && payload.perf_series.length > 0;

  // Count oracle evaluations from iterations
  const totalIterations = (iterations || []).length;
  const oracleEvals = (iterations || []).filter((it) => {
    const p = it.phases || {};
    return ["A_attempt_pure", "B_enrich", "D_verify_final"].some((k) => p[k]);
  }).length;

  return html`
    <div class="charts-panel">
      ${!hasCharts ? html`
        <div class="ke-stats-bar">
          <span class="ke-stat"><strong>${totalIterations}</strong> iteration${totalIterations !== 1 ? "s" : ""}</span>
          <span class="ke-stat"><strong>${oracleEvals}</strong> oracle eval${oracleEvals !== 1 ? "s" : ""}</span>
          ${totalIterations > 0 && oracleEvals === 0 ? html`
            <span class="muted" style="font-size:11px">— charts appear after first passing oracle run</span>
          ` : oracleEvals > 0 ? html`
            <span class="muted" style="font-size:11px">— no metrics collected (oracle didn't pass)</span>
          ` : null}
        </div>
      ` : null}
      <div class="charts-grid" ref=${gridRef}></div>
    </div>
  `;
}
