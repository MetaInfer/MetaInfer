// evolve-kernel task detail body.
//
// Renders the 8-phase kernel optimization flow:
//   Bootstrap: A(BGen Correctness Harness)→B(Review)→C(Gen Perf Harness)→D(Review)
//   Loop:      E(Select)→F(Optimize)→G(Verify)→H(Measure)⟲
//
// Composed of: state graph, kernel library, harness status, iteration metrics.

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { StateGraph } from "app/ok-evolve-state-graph";
import { KernelLibrary, KernelCodeView } from "app/ok-evolve-kernel-library";
import { AgentsPanel } from "app/agents-panel";
import { MultiGpuDashboard } from "app/ok-evolve-multi-gpu";
import {
  getIterations, getCharts, getStateGraph,
  getKernelLibrary, getCorrectnessHarness, getPerfHarness,
  getReferenceKernel, getKernelLineage, getFailures,
  getShapeBenchmark, refreshShapeBenchmark,
} from "app/ok-evolve-runtime-api";

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

function useRuntimeData(taskId) {
  const [data, setData] = useState({
    iterations: [], charts: null, graph: null,
    library: null, correctnessHarness: null, perfHarness: null,
    refKernel: null, failures: null,
  });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    if (run?.multi_gpu === false) { setIsMultiGpu(false); return; }
    try {
      const [it, ch, g, lib, chs, phs, ref, fails] = await Promise.all([
        withTimeout(getIterations(taskId)).catch(() => []),
        withTimeout(getCharts(taskId)).catch(() => null),
        withTimeout(getStateGraph(taskId)).catch(() => null),
        withTimeout(getKernelLibrary(taskId)).catch(() => null),
        withTimeout(getCorrectnessHarness(taskId)).catch(() => null),
        withTimeout(getPerfHarness(taskId)).catch(() => null),
        withTimeout(getReferenceKernel(taskId)).catch(() => null),
        withTimeout(getFailures(taskId)).catch(() => null),
      ]);
      setData({
        iterations: it || [], charts: ch, graph: g,
        library: lib, correctnessHarness: chs, perfHarness: phs,
        refKernel: ref, failures: fails,
      });
    } catch (e) {
      console.warn("evolve-kernel runtime fetch error:", e);
    }
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    if (run?.multi_gpu === false) { setIsMultiGpu(false); return; }
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);
  return { ...data, refresh };
}

// ---- Helper: flow indicator showing the active phase ----

function FlowIndicator({ graph }) {
  if (!graph) return null;
  const cur = graph.current;
  if (cur === "idle" || cur === "finished") {
    return html`<div class="ok-flow-indicator">
      <span class="muted">Status:</span>
      <span class="ok-phase-label">${cur === "finished" ? "Finished" : "Idle (waiting)"}</span>
    </div>`;
  }

  const phases = [
    "A_gen_correctness_harness", "B_review_correctness_harness",
    "C_gen_perf_harness", "D_review_perf_harness",
    "E_select_kernel", "F_optimize", "G_verify_correctness", "H_measure_perf",
  ];
  const labels = {
    "A_gen_correctness_harness": "A: Gen Correctness Harness",
    "B_review_correctness_harness": "B: Review Correctness Harness",
    "C_gen_perf_harness": "C: Gen Perf Harness",
    "D_review_perf_harness": "D: Review Perf Harness",
    "E_select_kernel": "E: Select Kernel",
    "F_optimize": "F: Optimize",
    "G_verify_correctness": "G: Verify Correctness",
    "H_measure_perf": "H: Measure Perf",
  };
  const curIdx = phases.indexOf(cur);

  return html`<div class="ok-flow-indicator">
    <span class="muted">Active:</span>
    ${phases.map((p, i) => html`
      <span key=${p} class=${"ok-phase-label" + (p === cur ? " active" : "")}
        style=${{ opacity: i > curIdx ? 0.3 : 1.0 }}>
        ${labels[p]}
      </span>
      ${i < phases.length - 1 ? html`<span class="ok-phase-divider" key=${'d' + i}>→</span>` : null}
    `)}
  </div>`;
}

// ---- Metric cards ----

function MetricCards({ library, iterations }) {
  const last = iterations && iterations.length > 0 ? iterations[iterations.length - 1] : null;
  const perf = last && last.perf ? last.perf : {};
  const bestKernel = library && library.kernels && library.kernels.length > 0 ? library.kernels[0] : null;

  return html`<div class="ok-meta-grid">
    <div class="ok-meta-card">
      <div class="ok-meta-value">${library ? library.size : 0}</div>
      <div class="ok-meta-label">Kernels in Library</div>
    </div>
    <div class="ok-meta-card">
      <div class="ok-meta-value">${bestKernel && bestKernel.exec_time_ms ? bestKernel.exec_time_ms.toFixed(3) + " ms" : "—"}</div>
      <div class="ok-meta-label">Best Exec Time</div>
    </div>
    <div class="ok-meta-card">
      <div class="ok-meta-value">${perf.speedup ? perf.speedup.toFixed(2) + "×" : "—"}</div>
      <div class="ok-meta-label">Last Speedup</div>
    </div>
    <div class="ok-meta-card">
      <div class="ok-meta-value">${iterations ? iterations.length : 0}</div>
      <div class="ok-meta-label">Optimization Iterations</div>
    </div>
  </div>`;
}

// ---- Harness status badges ----

function HarnessStatus({ harness, label }) {
  if (!harness) {
    return html`<span class="ok-harness-status pending">${label}: pending</span>`;
  }
  if (harness.exists) {
    return html`<span class="ok-harness-status pass">${label}: ready (${harness.lines} lines)</span>`;
  }
  return html`<span class="ok-harness-status fail">${label}: not generated</span>`;
}

// ---- Headroom Analysis Card ----

function bottleneckClass(bottleneck) {
  if (!bottleneck) return "";
  if (bottleneck === "near_optimal") return "ok-score-good";
  if (bottleneck === "memory_bound" || bottleneck === "compute_bound") return "ok-score-ok";
  return "ok-score-bad";
}

function bottleneckLabel(bottleneck) {
  if (!bottleneck) return "Unknown";
  const labels = {
    memory_bound: "Memory-Bound",
    compute_bound: "Compute-Bound",
    near_optimal: "Near-Optimal",
    inefficient: "Inefficient",
  };
  return labels[bottleneck] || bottleneck;
}

function HeadroomCard({ headroom }) {
  if (!headroom || !headroom.has_data) return null;

  const bwW = Math.min(100, Math.max(0, headroom.bw_util_pct || 0));
  const compW = Math.min(100, Math.max(0, headroom.compute_util_pct || 0));

  return html`
    <div class="ok-headroom-card">
      <div class="ok-headroom-header">
        <span class="ok-headroom-title">Roofline Analysis</span>
        <span class=${"ok-headroom-bottleneck " + bottleneckClass(headroom.bottleneck)}>
          ${bottleneckLabel(headroom.bottleneck)}
        </span>
      </div>

      <div class="ok-headroom-metrics">
        <div class="ok-headroom-bar-group">
          <div class="ok-headroom-bar-label">
            <span>HBM Bandwidth</span>
            <span class="muted" style="font-size:0.7rem;">${headroom.achieved_bw_gbps != null ? headroom.achieved_bw_gbps.toFixed(0) : "—"} / ${headroom.peak_bw_gbps != null ? headroom.peak_bw_gbps.toFixed(0) : "—"} GB/s</span>
          </div>
          <div class="ok-headroom-bar-track">
            <div class="ok-headroom-bar-fill ok-headroom-bar-bw" style=${{ width: bwW + "%" }}></div>
          </div>
          <span class="muted" style="font-size:0.7rem;">${headroom.bw_util_pct != null ? headroom.bw_util_pct.toFixed(0) : "—"}%</span>
        </div>

        <div class="ok-headroom-bar-group">
          <div class="ok-headroom-bar-label">
            <span>Compute (TFLOPS)</span>
            <span class="muted" style="font-size:0.7rem;">${headroom.achieved_tflops != null ? headroom.achieved_tflops.toFixed(2) : "—"} / ${headroom.peak_tflops != null ? headroom.peak_tflops.toFixed(0) : "—"} TFLOPS</span>
          </div>
          <div class="ok-headroom-bar-track">
            <div class="ok-headroom-bar-fill ok-headroom-bar-comp" style=${{ width: compW + "%" }}></div>
          </div>
          <span class="muted" style="font-size:0.7rem;">${headroom.compute_util_pct != null ? headroom.compute_util_pct.toFixed(0) : "—"}%</span>
        </div>
      </div>

      <div class="ok-headroom-summary-line">
        <span><strong>Estimated Headroom:</strong> ${headroom.headroom_pct != null ? headroom.headroom_pct.toFixed(0) + "%" : "—"}</span>
        <span class="muted" style="font-size:0.7rem;">Arithmetic Intensity: ${headroom.arithmetic_intensity != null ? headroom.arithmetic_intensity.toFixed(1) : "—"} FLOP/byte</span>
      </div>

      ${headroom.advice ? html`
        <div class="ok-headroom-advice">${headroom.advice}</div>
      ` : null}

      ${headroom.suggestions && headroom.suggestions.length > 0 ? html`
        <details class="ok-headroom-details">
          <summary>Optimization Suggestions (${headroom.suggestions.length})</summary>
          <ul class="ok-headroom-suggestions">
            ${headroom.suggestions.map(s => html`<li>${s}</li>`)}
          </ul>
        </details>
      ` : null}
    </div>
  `;
}

// ---- Kernel Lineage Panel (shown below kernel code when a kernel is selected) ----

function KernelLineagePanel({ kernelId, library, onSelectKernel, taskId }) {
  const [lineage, setLineage] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!taskId || !kernelId) return;
    let cancelled = false;
    setLoading(true);
    getKernelLineage(taskId, kernelId)
      .then((d) => { if (!cancelled) setLineage(d); })
      .catch((e) => { if (!cancelled) console.warn("lineage fetch error:", e); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [taskId, kernelId]);

  if (loading && !lineage) {
    return html`<div class="ok-lineage-panel"><p class="muted">Loading lineage…</p></div>`;
  }
  if (!lineage || lineage.error) {
    return null;
  }

  const { parent, improvement, ancestor_chain, retrospective, headroom } = lineage;

  return html`
    <div class="ok-lineage-panel">
      <div class="ok-lineage-header">
        <span class="ok-lineage-title">Optimization Lineage</span>
        ${ancestor_chain && ancestor_chain.length > 0 ? html`
          <span class="muted" style="font-size:0.75rem;">
            Ancestors: ${ancestor_chain.map(id => html`<code class="ok-lineage-id">${id.slice(0, 8)}</code>`)}
          </span>
        ` : html`<span class="muted" style="font-size:0.75rem;">Seed kernel (no ancestors)</span>`}
      </div>

      <div class="ok-lineage-body">
        <div class="ok-lineage-metrics">
          <div class="ok-lineage-metric">
            <span class="ok-lineage-metric-label">Parent</span>
            <span class="ok-lineage-metric-value">
              ${parent
                ? html`<a class="ok-lineage-link" onClick=${() => {
                    const pk = (library && library.kernels || []).find(k => k.id === parent.id);
                    if (pk && onSelectKernel) onSelectKernel(pk);
                  }}>${parent.id.slice(0, 8)}…</a>`
                : html`<span class="muted">seed</span>`
              }
            </span>
          </div>
          <div class="ok-lineage-metric">
            <span class="ok-lineage-metric-label">Parent Time</span>
            <span class="ok-lineage-metric-value">${parent && parent.exec_time_ms ? parent.exec_time_ms.toFixed(4) + " ms" : "—"}</span>
          </div>
          <div class="ok-lineage-metric">
            <span class="ok-lineage-metric-label">Speedup vs Parent</span>
            <span class=${"ok-lineage-metric-value " + (improvement.speedup_vs_parent && improvement.speedup_vs_parent > 1 ? "ok-score-good" : improvement.speedup_vs_parent && improvement.speedup_vs_parent < 1 ? "ok-score-bad" : "")}>
              ${improvement.speedup_vs_parent ? improvement.speedup_vs_parent.toFixed(2) + "×" : "—"}
            </span>
          </div>
          <div class="ok-lineage-metric">
            <span class="ok-lineage-metric-label">Time Delta</span>
            <span class=${"ok-lineage-metric-value " + (improvement.exec_time_delta_ms != null && improvement.exec_time_delta_ms < 0 ? "ok-score-good" : "")}>
              ${improvement.exec_time_delta_ms != null ? (improvement.exec_time_delta_ms > 0 ? "+" : "") + improvement.exec_time_delta_ms.toFixed(4) + " ms" : "—"}
            </span>
          </div>
        </div>

        ${improvement.summary_items && improvement.summary_items.length > 0 ? html`
          <div class="ok-lineage-summary">
            <div class="ok-lineage-summary-label">Improvement Summary</div>
            <ul class="ok-lineage-summary-items">
              ${improvement.summary_items.map(item => html`<li>${item}</li>`)}
            </ul>
          </div>
        ` : improvement.summary ? html`
          <div class="ok-lineage-summary">
            <div class="ok-lineage-summary-label">Improvement Summary</div>
            <div class="ok-lineage-summary-text">${improvement.summary}</div>
          </div>
        ` : null}

        ${headroom && headroom.has_data ? html`
          <${HeadroomCard} headroom=${headroom} />
        ` : null}

        ${retrospective && retrospective.has_retrospective ? html`
          <details class="ok-lineage-retrospective">
            <summary>Full Retrospective (iteration ${retrospective.iteration})</summary>
            <div class="ok-code-preview" style="margin-top:0.5rem;max-height:400px;">${retrospective.markdown}</div>
          </details>
        ` : null}
      </div>
    </div>
  `;
}

// ---- Shape Benchmark Panel ----

function speedupClass(val) {
  if (val == null || val <= 0) return "";
  if (val >= 1.2) return "ok-score-good";
  if (val >= 1.0) return "ok-score-ok";
  return "ok-score-bad";
}

function formatMs2(ms) {
  if (ms == null || ms === 0) return "—";
  if (ms < 1) return (ms * 1000).toFixed(1) + " μs";
  if (ms < 1000) return ms.toFixed(3) + " ms";
  return (ms / 1000).toFixed(2) + " s";
}

function ShapeBenchmarkPanel({ taskId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (forceRefresh) => {
    if (!taskId) return;
    if (run?.multi_gpu === false) { setIsMultiGpu(false); return; }
    setLoading(true);
    setError(null);
    try {
      const d = forceRefresh
        ? await refreshShapeBenchmark(taskId)
        : await getShapeBenchmark(taskId);
      setData(d);
      if (d.error) setError(d.error);
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [taskId]);

  useEffect(() => { load(false); }, [load]);

  if (loading && !data) {
    return html`<section class="panel ok-panel-full">
      <h2>Shape Benchmark</h2>
      <p class="muted">Loading benchmarks…</p>
    </section>`;
  }

  if (error && (!data || !data.results || data.results.length === 0)) {
    return html`<section class="panel ok-panel-full">
      <h2>Shape Benchmark</h2>
      <p class="muted">${error}</p>
    </section>`;
  }

  if (!data || !data.results || data.results.length === 0) {
    return null;
  }

  const results = data.results;
  const bestId = data.best_kernel_id ? data.best_kernel_id.slice(0, 8) : "?";

  // Group results by shape
  const shapeMap = {};
  for (const r of results) {
    const key = r.shape_label;
    if (!shapeMap[key]) shapeMap[key] = [];
    shapeMap[key].push(r);
  }

  return html`
    <section class="panel ok-panel-full">
      <h2>Shape Benchmark — Best Kernel vs Baseline
        <span class="muted" style="font-size:0.75rem;margin-left:0.5rem;">
          (kernel ${bestId}, ${data.cached ? "cached" : "fresh"})
        </span>
        <button class="btn btn-sm" style="float:right;"
          onClick=${() => { setRefreshing(true); load(true); }}
          disabled=${refreshing}>
          ${refreshing ? "Running…" : "Refresh"}
        </button>
      </h2>

      <div class="ok-shape-bench">
        ${Object.entries(shapeMap).map(([label, rows]) => html`
          <div class="ok-shape-group" key=${label}>
            <div class="ok-shape-group-label">${label}</div>
            <table class="ok-kernel-table ok-shape-table">
              <thead>
                <tr>
                  <th>M</th>
                  <th>(M×K×N)</th>
                  <th>Baseline</th>
                  <th>Best Kernel</th>
                  <th>Speedup</th>
                </tr>
              </thead>
              <tbody>
                ${rows.map(r => html`
                  <tr key=${label + "-" + r.M}>
                    <td><strong>${r.M}</strong></td>
                    <td class="muted" style="font-size:0.75rem;">${r.M}×${r.K}×${r.N}</td>
                    <td>${formatMs2(r.ref_ms)}</td>
                    <td class="ok-score-good">${formatMs2(r.best_ms)}</td>
                    <td class=${speedupClass(r.speedup)}>
                      ${r.error ? html`<span class="muted" title=${r.error}>err</span>`
                        : r.speedup.toFixed(2) + "×"}
                    </td>
                  </tr>
                `)}
              </tbody>
            </table>
          </div>
        `)}
      </div>
    </section>
  `;
}


// ---- Failure Log Panel ----

function FailureLog({ failures }) {
  const [expandedId, setExpandedId] = useState(null);

  if (!failures || !failures.failures || failures.failures.length === 0) {
    return null;
  }

  const { failures: items, total, errors, warnings } = failures;

  return html`
    <section class="panel ok-panel-full">
      <h2>Failure Log
        <span style="font-size:0.8rem;margin-left:0.5rem;">
          <span class="ok-badge" style="background:rgba(248,81,73,0.15);color:#f85149;">${errors} errors</span>
          ${warnings > 0 ? html`<span class="ok-badge" style="background:rgba(210,153,34,0.15);color:#d29922;margin-left:0.25rem;">${warnings} warnings</span>` : null}
        </span>
      </h2>

      <div class="ok-failure-log">
        ${items.map((f, i) => {
          const isExpanded = expandedId === i;
          const entryId = `fail-${f.iteration}-${f.phase || "iter"}-${i}`;
          return html`
            <div key=${entryId} class="ok-failure-entry">
              <div
                class=${"ok-failure-header " + (f.severity === "error" ? "severity-error" : "severity-warning")}
                onClick=${() => setExpandedId(isExpanded ? null : i)}
              >
                <span class="ok-failure-severity">${f.severity === "error" ? "✗" : "⚠"}</span>
                <span class="ok-failure-iteration">Iter ${f.iteration}</span>
                <span class="ok-failure-phase">${f.phase_label}</span>
                <span class="ok-failure-summary">${f.summary}</span>
                ${f.attempts > 1 ? html`<span class="ok-failure-attempts">${f.attempts} attempts</span>` : null}
                <span class="ok-failure-toggle">${isExpanded ? "▲" : "▼"}</span>
              </div>
              ${isExpanded ? html`
                <div class="ok-failure-detail">
                  <div class="ok-failure-meta">
                    <span><strong>Outcome:</strong> ${f.outcome}</span>
                    ${f.attempts > 1 ? html`<span><strong>Attempts:</strong> ${f.attempts}</span>` : null}
                  </div>
                  <div class="ok-code-preview">${f.detail}</div>
                </div>
              ` : null}
            </div>
          `;
        })}
      </div>
    </section>
  `;
}

// ---- Main view ----

export default function OptKernelDetailView({
  taskId,
  run,
  status,
  data,
}) {
  const [selectedKernel, setSelectedKernel] = useState(null);
  const [showHarness, setShowHarness] = useState(null); // "correctness" | "perf" | null
  const { agents, loadState, lastErr } = data;
  const rt = useRuntimeData(taskId);
  const [isMultiGpu, setIsMultiGpu] = useState(run?.multi_gpu ? true : null);

  // Check if this task is a multi-GPU task (initialized from run prop)
  useEffect(() => {
    if (!taskId) return;
    if (run?.multi_gpu === false) { setIsMultiGpu(false); return; }
    fetch(`/api/evolve-kernel/${taskId}/gpu-status`)
      .then(r => r.json())
      .then(d => setIsMultiGpu(d.is_multi_gpu ? true : false))
      .catch(() => setIsMultiGpu(false));
  }, [taskId]);

  if (loadState === "error" && lastErr) {
    return html`
      <div class="task-banner task-banner-err">
        <strong>Refresh failed:</strong> ${lastErr}
        <span class="muted">(auto-retry)</span>
      </div>
    `;
  }

  // If this is a multi-GPU task, show the unified dashboard
  if (isMultiGpu) {
    return html`
      <${MultiGpuDashboard} taskId=${taskId} />
    `;
  }

  return html`
    <${FlowIndicator} graph=${rt.graph} />

    <${MetricCards} library=${rt.library} iterations=${rt.iterations} />

    <div class="ok-grid">
      <section class="panel">
        <h2>State Machine</h2>
        <${StateGraph} graph=${rt.graph} />
      </section>

      <section class="panel">
        <h2>Kernel Library
          <span class="muted" style="font-size:0.75rem;margin-left:0.5rem;">(click for details)</span>
        </h2>
        <${KernelLibrary}
          library=${rt.library}
          selectedKernelId=${selectedKernel && selectedKernel.id}
          onSelectKernel=${(k) => setSelectedKernel(k)} />
      </section>
    </div>

    ${selectedKernel ? html`
      <${KernelLineagePanel}
        kernelId=${selectedKernel.id}
        library=${rt.library}
        onSelectKernel=${(k) => setSelectedKernel(k)}
        taskId=${taskId}
      />
	      <section class="panel ok-panel-full">
	        <h2>Kernel Source: ${selectedKernel.id.slice(0, 8)}…
	          <button class="btn btn-sm" style="float:right;" onClick=${() => setSelectedKernel(null)}>× close</button>
	        </h2>
	        <${KernelCodeView} kernel=${selectedKernel} optimizerMode=${(rt.library && rt.library.optimizer_mode) || "Triton (standard)"} />
	      </section>
    ` : null}

    <div class="ok-grid">
      <section class="panel">
        <h2>Harnesses</h2>
        <div style="display:flex;gap:0.75rem;margin-bottom:0.75rem;">
          <${HarnessStatus} harness=${rt.correctnessHarness} label="Correctness" />
          <${HarnessStatus} harness=${rt.perfHarness} label="Performance" />
        </div>
        <div style="display:flex;gap:0.5rem;">
          <button class="btn btn-sm" onClick=${() => setShowHarness(showHarness === "correctness" ? null : "correctness")}
            disabled=${!rt.correctnessHarness || !rt.correctnessHarness.exists}>
            ${showHarness === "correctness" ? "Hide" : "Show"} Correctness Harness
          </button>
          <button class="btn btn-sm" onClick=${() => setShowHarness(showHarness === "perf" ? null : "perf")}
            disabled=${!rt.perfHarness || !rt.perfHarness.exists}>
            ${showHarness === "perf" ? "Hide" : "Show"} Perf Harness
          </button>
        </div>

        ${showHarness === "correctness" && rt.correctnessHarness && rt.correctnessHarness.exists ? html`
          <div class="ok-code-preview" style="margin-top:0.75rem;">
            ${rt.correctnessHarness.code.split("\n").map(line => html`<div class="ok-code-line">${line}</div>`)}
          </div>
        ` : null}
        ${showHarness === "perf" && rt.perfHarness && rt.perfHarness.exists ? html`
          <div class="ok-code-preview" style="margin-top:0.75rem;">
            ${rt.perfHarness.code.split("\n").map(line => html`<div class="ok-code-line">${line}</div>`)}
          </div>
        ` : null}
      </section>

      <section class="panel">
        <h2>Live Sub-agents</h2>
        <${AgentsPanel} agents=${agents} />
      </section>
    </div>

    <${ShapeBenchmarkPanel} taskId=${taskId} />

    <${FailureLog} failures=${rt.failures} />
  `;
}
