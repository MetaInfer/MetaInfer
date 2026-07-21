import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

const DEFINITIONS = [
  ["latency_ms", "Weighted latency", "ms", "#58a6ff"],
  ["weighted_speedup", "Speedup vs baseline", "×", "#3fb950"],
  ["tflops", "Compute throughput", "TFLOPS", "#d29922"],
  ["bandwidth_gbps", "Modelled bandwidth", "GB/s", "#f778ba"],
  ["measured_bandwidth_gbps", "Profiler memory bandwidth", "GB/s", "#ff9b71"],
  ["l2_hit_pct", "L2 hit rate", "%pts", "#2dd4bf"],
  ["compute_busy_pct", "Compute busy", "%pts", "#f59e0b"],
  ["vgpr_count", "VGPR per work-item", "registers", "#c084fc"],
  ["lds_bytes", "LDS per workgroup", "bytes", "#22c55e"],
  ["critical_regression", "Critical regression", "%", "#a371f7"],
  ["duration_s", "Iteration duration", "s", "#79c0ff"],
];

function options(unit) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    parsing: false,
    scales: {
      x: {
        type: "linear",
        min: 0,
        ticks: {
          color: "#8b949e",
          precision: 0,
          callback: (v) => Number.isInteger(v) ? (v === 0 ? "base" : `iter ${v}`) : "",
        },
        grid: { color: "#21262d" },
      },
      y: {
        beginAtZero: true,
        ticks: {
          color: "#8b949e",
          callback: (v) => unit === "%" ? `${(Number(v) * 100).toFixed(1)}%`
            : unit === "%pts" ? `${Number(v).toFixed(1)}%` : v,
        },
        grid: { color: "#21262d" },
        title: { display: true, text: unit, color: "#8b949e" },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (items) => items[0].parsed.x === 0
            ? "certified baseline"
            : `iteration ${items[0].parsed.x}`,
          label: (item) => {
            const value = item.parsed.y;
            const shown = unit === "%"
              ? `${(value * 100).toFixed(2)}%`
              : unit === "%pts" ? `${Number(value).toFixed(2)}%`
              : `${Number(value).toFixed(4)} ${unit}`;
            return `${shown}${item.raw.promoted ? " · promoted" : ""}`;
          },
        },
      },
    },
  };
}

export function ProfilerCharts({ payload }) {
  const root = useRef(null);
  const charts = useRef({});

  useEffect(() => () => {
    Object.values(charts.current).forEach(({ chart }) => chart.destroy());
    charts.current = {};
  }, []);

  useEffect(() => {
    if (!root.current || !payload) return;
    const series = payload.series || {};
    const wanted = new Set(
      DEFINITIONS.filter(([key]) => (series[key] || []).length).map(([key]) => key),
    );
    for (const [key, value] of Object.entries(charts.current)) {
      if (!wanted.has(key) || !root.current.contains(value.wrap)) {
        value.chart.destroy();
        value.wrap.remove();
        delete charts.current[key];
      }
    }
    for (const [key, title, unit, color] of DEFINITIONS) {
      const points = series[key] || [];
      if (!points.length) continue;
      if (!charts.current[key]) {
        const wrap = document.createElement("div");
        wrap.className = "gemm-mini-chart";
        wrap.innerHTML = `<h3></h3><div class="gemm-chart-box"><canvas></canvas></div>`;
        wrap.querySelector("h3").textContent = title;
        root.current.appendChild(wrap);
        const chart = new Chart(wrap.querySelector("canvas").getContext("2d"), {
          type: "line",
          data: {
            datasets: [{
              data: [],
              borderColor: color,
              backgroundColor: `${color}33`,
              pointBackgroundColor: [],
              pointRadius: [],
              borderWidth: 2,
              tension: 0.2,
            }],
          },
          options: options(unit),
        });
        charts.current[key] = { chart, wrap };
      }
      const chart = charts.current[key].chart;
      chart.data.datasets[0].data = points;
      chart.data.datasets[0].pointBackgroundColor = points.map(
        (point) => point.promoted ? "#3fb950" : color,
      );
      chart.data.datasets[0].pointRadius = points.map((point) => point.promoted ? 5 : 3);
      chart.update();
    }
  }, [payload]);

  const empty = !Object.values(payload?.series || {}).some((points) => points?.length);
  return html`
    ${empty ? html`<p class="muted">Profiler data appears after baseline certification.</p>` : null}
    <div class="gemm-profiler-grid" ref=${root}></div>
  `;
}
