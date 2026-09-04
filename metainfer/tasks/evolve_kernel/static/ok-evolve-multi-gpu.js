// Multi-GPU unified dashboard for evolve-kernel.

import { html } from "htm/preact";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { StateGraph } from "app/ok-evolve-state-graph";
import { KernelLibrary, KernelCodeView } from "app/ok-evolve-kernel-library";

const BASE = (taskId) => `/api/evolve-kernel/${taskId}`;

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

// ---- Helpers ----

function formatMs(ms) {
  if (ms == null || ms <= 0) return "—";
  if (ms < 1) return (ms * 1000).toFixed(1) + " μs";
  if (ms < 1000) return ms.toFixed(3) + " ms";
  return (ms / 1000).toFixed(2) + " s";
}

function formatAge(s) {
  if (s == null || s <= 0) return "";
  if (s < 60) return Math.round(s) + "s";
  if (s < 3600) return Math.round(s / 60) + "m";
  return (s / 3600).toFixed(1) + "h";
}

function speedupClass(val) {
  if (val == null || val <= 0) return "";
  if (val >= 1.2) return "ok-score-good";
  if (val >= 1.0) return "ok-score-ok";
  return "ok-score-bad";
}

function phaseLabel(p) {
  const labels = {
    "A_gen_correctness_harness": "A:Gen Harness", "B_review_correctness_harness": "B:Review Harness",
    "C_gen_perf_harness": "C:Gen Perf", "D_review_perf_harness": "D:Review Perf",
    "E_select_kernel": "E:Select", "F_optimize": "F:Optimize",
    "G_verify_correctness": "G:Verify", "H_measure_perf": "H:Measure",
    "finished": "Done", "starting": "Starting", "idle": "Idle", "running": "Running",
  };
  return labels[p] || (p || "?").slice(0, 20);
}

function phaseDotClass(p) {
  if (!p) return "idle";
  if (p.startsWith("A_") || p.startsWith("B_") || p.startsWith("C_") || p.startsWith("D_")) return "bootstrap";
  if (p.startsWith("E_") || p.startsWith("F_") || p.startsWith("G_") || p.startsWith("H_")) return "optimize";
  if (p === "finished") return "finished";
  return "idle";
}

function phaseCardClass(p) {
  if (!p) return "";
  if (p.startsWith("A") || p.startsWith("B") || p.startsWith("C") || p.startsWith("D")) return "bootstrap";
  if (p.startsWith("E") || p.startsWith("F") || p.startsWith("G") || p.startsWith("H")) return "optimize";
  if (p === "finished") return "done";
  return "";
}

function roleLabel(r) {
  const labels = {
    "correctness_harness_generator": "Gen Harness", "correctness_harness_reviewer": "Review Harness",
    "perf_harness_generator": "Gen Perf", "perf_harness_reviewer": "Review Perf",
    "kernel_optimizer": "Optimizer", "correctness_verifier": "Verify",
    "perf_measurer": "Measure", "headroom_analyzer": "Headroom", "seed_generator": "Seed",
  };
  return labels[r] || (r || "?").replace(/_/g, " ").slice(0, 16);
}

// ---- Parse shapes from extra_notes ----

function parseShapes(notes) {
  if (!notes) return [];
  const shapes = [];
  // Match: name (TP=N): M=a,b,c,... (M, K) @ (K, N)
  // Lines are indented within the multi-line string, so don't use ^.
  const re = /([\w_]+)\s*\(TP=(\d+)\):\s*M=([\d,]+)\s*\(\s*(?:M|m)\s*,\s*(\d+)\s*\)\s*@\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)/g;
  let m;
  while ((m = re.exec(notes)) !== null) {
    const name = m[1];
    const tp = parseInt(m[2]);
    const mValues = m[3].split(',').map(s => parseInt(s.trim())).filter(Boolean);
    const K = parseInt(m[4]);
    const K2 = parseInt(m[5]);
    const N = parseInt(m[6]);
    // Skip if K values don't match (malformed line or different format)
    if (K !== K2) continue;
    shapes.push({ name, tp, mValues, K, N, label: `${name} (TP=${tp}) ${K}×${N}` });
  }
  return shapes;
}

// ---- GPU Status Card ----

function GpuCard({ worker, onClick, isSelected }) {
  const phase = worker.phase || "starting";
  const running = worker.running;
  const gpuLabel = worker.label || `GPU ${worker.gpu_idx}`;
  const agents = worker.agents || [];
  const activeAgent = agents.find(a => a.phase === "running");

  return html`
    <div class=${"ok-mgpu-card " + phaseCardClass(phase) + (isSelected ? " selected" : "")} onClick=${onClick} style="cursor:pointer;">
      <div class="ok-mgpu-card-header">
        <span class="ok-mgpu-card-title">${gpuLabel}</span>
        <span class=${"ok-mgpu-card-status " + (running ? "running" : "stopped")}>
          ${running ? "● Running" : "○ Idle"}
        </span>
      </div>
      <div class="ok-mgpu-card-body">
        <div class="ok-mgpu-card-metrics">
          <div class="ok-mgpu-card-metric">
            <span class="ok-mgpu-card-label">Phase</span>
            <span class="ok-mgpu-card-value" style="font-size:0.78rem;display:flex;align-items:center;gap:2px;">
              <span class=${"ok-phase-dot " + phaseDotClass(phase)}></span>${phaseLabel(phase)}
            </span>
          </div>
          <div class="ok-mgpu-card-metric">
            <span class="ok-mgpu-card-label">Iter</span>
            <span class="ok-mgpu-card-value">${worker.iteration || 0}</span>
          </div>
        </div>
        <div class="ok-mgpu-card-metrics">
          <div class="ok-mgpu-card-metric">
            <span class="ok-mgpu-card-label">Best</span>
            <span class="ok-mgpu-card-value ok-score-good">${formatMs(worker.exec_time_ms)}</span>
          </div>
          <div class="ok-mgpu-card-metric">
            <span class="ok-mgpu-card-label">Speedup</span>
            <span class=${"ok-mgpu-card-value " + speedupClass(worker.speedup)}>
              ${worker.speedup > 0 ? worker.speedup.toFixed(2) + "×" : "—"}
            </span>
          </div>
        </div>
        <div class="ok-mgpu-card-metrics" style="margin-top:0.15rem;">
          <div class="ok-mgpu-card-metric" style="min-width:0;flex-direction:row;align-items:center;gap:0.25rem;">
            <span class="ok-mgpu-card-label" style="white-space:nowrap;">Kernels</span>
            <span class="ok-mgpu-card-value" style="font-size:0.8rem;">${worker.kernel_count || 0}</span>
          </div>
        </div>
      </div>
      ${worker.iteration > 0 ? html`
        <div class="ok-mgpu-card-iter-bar"><div class="ok-mgpu-card-iter-fill" style=${{ width: Math.min(100, (worker.iteration / 10) * 100) + "%" }}></div></div>
      ` : null}
    </div>
  `;
}

function GpuCards({ workers, selectedGpu, onSelectGpu }) {
  if (!workers || workers.length === 0) {
    return html`<p class="muted">No GPU workers found.</p>`;
  }
  return html`
    <div class="ok-mgpu-cards">
      ${workers.map(w => html`<${GpuCard} key=${w.gpu_idx} worker=${w} isSelected=${selectedGpu === w.gpu_idx} onClick=${() => onSelectGpu(selectedGpu === w.gpu_idx ? null : w.gpu_idx)} />`)}
    </div>
  `;
}

// ---- Collapsible Agent List ----

function PerGpuAgents({ workers }) {
  const hasAnyAgents = workers.some(w => (w.agents || []).length > 0);
  if (!hasAnyAgents) return null;

  const [collapsedSet, setCollapsedSet] = useState(() => {
    const s = new Set();
    workers.forEach((w, i) => { if (i > 0) s.add(w.gpu_idx); });
    return s;
  });

  return html`
    <section class="panel ok-panel-full">
      <h2>Live Sub-agents
        <button class="btn btn-sm" style="float:right;font-size:0.7rem;" onClick=${() => {
          const allOpen = workers.every(w => !collapsedSet.has(w.gpu_idx));
          setCollapsedSet(allOpen ? new Set(workers.map(w => w.gpu_idx)) : new Set());
        }}>
          ${workers.every(w => !collapsedSet.has(w.gpu_idx)) ? "Collapse All" : "Expand All"}
        </button>
      </h2>
      <div class="ok-mgpu-agents-grid">
        ${workers.map(w => {
          const agents = w.agents || [];
          if (agents.length === 0) return null;
          const isCollapsed = collapsedSet.has(w.gpu_idx);
          return html`
            <div key=${"a-" + w.gpu_idx} class=${"ok-mgpu-agent-group" + (isCollapsed ? " collapsed" : "")}>
              <div class="ok-mgpu-agent-group-header" onClick=${() => {
                setCollapsedSet(prev => { const n = new Set(prev); if (n.has(w.gpu_idx)) n.delete(w.gpu_idx); else n.add(w.gpu_idx); return n; });
              }}>
                <span>${w.label}</span>
                <span class=${"collapse-icon" + (isCollapsed ? "" : " open")}>▶</span>
              </div>
              ${agents.map(a => html`
                <div key=${a.name} class=${"ok-mgpu-agent-row " + (a.phase === "running" ? "agent-running" : a.success === false ? "agent-failed" : "agent-done")}>
                  <span class="ok-mgpu-agent-role">${roleLabel(a.role)}</span>
                  <span class="ok-mgpu-agent-status">
                    ${a.phase === "running" ? html`<span style="color:#3fb950;">● ${formatAge(a.elapsed_s)}</span>`
                      : a.success === false ? html`<span style="color:#f85149;">✗</span>`
                      : a.success === true ? html`<span style="color:#3fb950;">✓</span>`
                      : html`<span class="muted">${a.phase}</span>`}
                  </span>
                  ${a.error ? html`<span class="ok-mgpu-agent-error" title=${a.error}>${a.error.slice(0, 30)}</span>` : null}
                </div>
              `)}
            </div>
          `;
        })}
      </div>
    </section>
  `;
}

// ---- Stats Bar ----

function StatsBar({ detail }) {
  if (!detail) return null;
  return html`
    <div class="ok-stats-bar">
      <div class="ok-stats-item"><span class="ok-stats-label">Phase</span> <span class="ok-stats-value" style="color:#58a6ff;">${phaseLabel(detail.phase)}</span></div>
      <span class="ok-stats-sep">|</span>
      <div class="ok-stats-item"><span class="ok-stats-label">Iter</span> <span class="ok-stats-value">${detail.iteration}</span></div>
      <span class="ok-stats-sep">|</span>
      <div class="ok-stats-item"><span class="ok-stats-label">Best</span> <span class="ok-stats-value ok-score-good">${formatMs(detail.exec_time_ms)}</span></div>
      <span class="ok-stats-sep">|</span>
      <div class="ok-stats-item"><span class="ok-stats-label">Speedup</span> <span class="ok-stats-value ${speedupClass(detail.speedup)}">${detail.speedup > 0 ? detail.speedup.toFixed(2) + "×" : "—"}</span></div>
      <span class="ok-stats-sep">|</span>
      <div class="ok-stats-item"><span class="ok-stats-label">Library</span> <span class="ok-stats-value">${detail.kernel_count}/10</span></div>
    </div>
  `;
}

// ---- Roofline callout (clean, no plot) ----

const BOTTLENECK_META = {
  memory_bound:  { color: "#58a6ff", icon: "▦", label: "Memory-Bound",   desc: "HBM bandwidth limits performance. The data movement per FLOP (AI) is below the ridge point." },
  compute_bound: { color: "#d29922", icon: "◉", label: "Compute-Bound", desc: "Instruction throughput limits performance. The kernel has enough data reuse to saturate compute." },
  near_optimal:  { color: "#3fb950", icon: "✓", label: "Near-Optimal", desc: "The kernel is close to the hardware roofline. Further tuning yields diminishing returns." },
  inefficient:   { color: "#f85149", icon: "⚠", label: "Inefficient", desc: "Neither bandwidth nor compute are near peak. Check occupancy, register pressure, or tile alignment." },
};

function RooflineCallout({ headroom }) {
  if (!headroom || !headroom.has_data) return null;

  const b = headroom.bottleneck || "inefficient";
  const meta = BOTTLENECK_META[b] || BOTTLENECK_META.inefficient;
  const eff = headroom.roofline_efficiency_pct || 0;
  const ai = headroom.arithmetic_intensity || 0;
  const ridge = headroom.ai_ridge || 0;
  const pAchieved = headroom.achieved_tflops || 0;
  const pMax = headroom.p_max_tflops || 0;
  const peak = headroom.peak_tflops || 220;
  const bwPeak = headroom.peak_bw_gbps || 700;
  const pBwRoof = headroom.p_bw_roof_tflops;

  return html`
    <div class="ok-roofline-callout" style=${{ borderLeftColor: meta.color }}>
      <div class="ok-roofline-callout-header">
        <span class="ok-roofline-icon" style=${{ color: meta.color }}>${meta.icon}</span>
        <strong style="color:${meta.color};font-size:1rem;">${meta.label}</strong>
        ${ridge > 0 ? html`
          <span class="muted" style="font-size:0.75rem;margin-left:0.5rem;">
            AI = ${ai.toFixed(1)} ${ai < ridge ? "<" : "≥"} Ridge = ${ridge.toFixed(1)} FLOP/byte
          </span>
        ` : html`
          <span class="muted" style="font-size:0.75rem;margin-left:0.5rem;">AI = ${ai.toFixed(1)} FLOP/byte</span>
        `}
      </div>
      <p class="ok-roofline-callout-desc">${meta.desc}</p>

      <div class="ok-roofline-eq" style="margin-top:0.5rem;">
        <div class="ok-roofline-eq-row">
          <span class="muted" style="min-width:120px;">P_achieved</span>
          <strong>${pAchieved.toFixed(2)}</strong> TFLOPS
          <span class="muted" style="margin-left:0.5rem;font-size:0.7rem;">= FLOPs / exec_time</span>
        </div>
        ${pBwRoof ? html`
          <div class="ok-roofline-eq-row">
            <span class="muted" style="min-width:120px;">P_bw_roof</span>
            <strong>${pBwRoof.toFixed(2)}</strong> TFLOPS
            <span class="muted" style="margin-left:0.5rem;font-size:0.7rem;">= BW_peak × AI = ${bwPeak.toFixed(0)} × ${ai.toFixed(0)} / 1000</span>
          </div>
        ` : null}
        <div class="ok-roofline-eq-row">
          <span class="muted" style="min-width:120px;">P_max</span>
          <strong>${pMax > 0 ? pMax.toFixed(2) : `min(${peak.toFixed(0)}T compute, BW×AI)`}</strong> TFLOPS
          <span class="muted" style="margin-left:0.5rem;font-size:0.7rem;">= roofline ceiling</span>
        </div>
      </div>

      <div class="ok-roofline-eff-bar" style="margin-top:0.6rem;">
        <span class="muted" style="font-size:0.7rem;min-width:100px;">Roofline Eff.</span>
        <div class="ok-headroom-bar-track" style="flex:1;">
          <div class="ok-headroom-bar-fill" style=${{ width: Math.min(100, Math.max(0, eff)) + "%", background: meta.color }}></div>
        </div>
        <strong style="min-width:40px;text-align:right;color:${meta.color};">${eff.toFixed(0)}%</strong>
      </div>

      ${headroom.advice ? html`<p class="muted" style="font-size:0.75rem;margin-top:0.4rem;line-height:1.4;">${headroom.advice}</p>` : null}
    </div>
  `;
}

// ---- Shape Summary (what this GPU optimizes) ----

function ShapeSummary({ detail }) {
  // Parse shapes from the agents data (the detail doesn't have extra_notes itself)
  // We try to guess from the shapes field or leave it for the caller
  return null; // Will be rendered inline with detail.shapes data
}

// ---- Code with line numbers ----

function CodeBlock({ code }) {
  if (!code) return html`<p class="muted">Code not available</p>`;
  const lines = code.split("\n");
  return html`
    <div class="ok-code-preview large">
      ${lines.map(line => html`<div class="ok-code-line">${line}</div>`)}
    </div>
  `;
}

// ---- Per-GPU Detail Panel ----

function GpuDetailPanel({ taskId, gpuIdx, onClose }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedKernel, setSelectedKernel] = useState(null);
  const [kernelLineage, setKernelLineage] = useState(null);
  const [lineageLoading, setLineageLoading] = useState(false);
  const [showHarness, setShowHarness] = useState(null);
  const [harnessData, setHarnessData] = useState({});
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    setLoading(true);
    async function fetchDetail() {
      if (!taskId || gpuIdx == null) return;
      try {
        const res = await fetch(`${BASE(taskId)}/gpu/${gpuIdx}/detail`);
        if (!res.ok) throw new Error(`${res.status}`);
        const d = await res.json();
        if (!mountedRef.current) return;
        setDetail(d);
        setError(null);
      } catch (e) {
        if (!mountedRef.current) return;
        setError(e.message);
      } finally {
        if (mountedRef.current) setLoading(false);
      }
    }
    fetchDetail();
    const pollId = setInterval(fetchDetail, 3000);
    return () => { mountedRef.current = false; clearInterval(pollId); };
  }, [taskId, gpuIdx]);

  useEffect(() => {
    if (!showHarness || !taskId || gpuIdx == null) return;
    const htype = showHarness;
    if (harnessData[htype]) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${BASE(taskId)}/gpu/${gpuIdx}/harnesses/${htype}`);
        if (!res.ok) throw new Error(`${res.status}`);
        const d = await res.json();
        if (!cancelled) setHarnessData(prev => ({ ...prev, [htype]: d }));
      } catch (e) { /* ignore */ }
    })();
    return () => { cancelled = true; };
  }, [showHarness, taskId, gpuIdx]);

  useEffect(() => {
    if (!selectedKernel || !taskId || gpuIdx == null) {
      setKernelLineage(null);
      return;
    }
    let cancelled = false;
    setLineageLoading(true);
    (async () => {
      try {
        const res = await fetch(`${BASE(taskId)}/gpu/${gpuIdx}/kernel-library/${selectedKernel.id}/lineage`);
        if (!res.ok) throw new Error(`${res.status}`);
        const d = await res.json();
        if (!cancelled) setKernelLineage(d);
      } catch (e) { /* ignore */ }
      finally { if (!cancelled) setLineageLoading(false); }
    })();
    return () => { cancelled = true; };
  }, [selectedKernel, taskId, gpuIdx]);

  if (loading && !detail && !error) {
    return html`<section class="panel ok-panel-full"><p class="muted">Loading GPU ${gpuIdx} detail…</p></section>`;
  }

  if (error && !detail) {
    return html`<section class="panel ok-panel-full"><p style="color:#f85149;">Error: ${error}</p><button class="btn btn-sm" onClick=${onClose}>× close</button></section>`;
  }

  if (!detail) return null;

  const headroom = kernelLineage && kernelLineage.headroom && kernelLineage.headroom.has_data
    ? kernelLineage.headroom : null;
  const improvement = kernelLineage ? kernelLineage.improvement : null;
  const parent = kernelLineage ? kernelLineage.parent : null;
  const profiling = kernelLineage ? kernelLineage.kernel : null;
  const correctnessReady = harnessData["correctness"] ? harnessData["correctness"].exists : null;
  const perfReady = harnessData["perf"] ? harnessData["perf"].exists : null;

  // Parse target shapes from detail
  const shapes = parseShapes(detail.shapes || "");

  // Categorize M values
  const decodeMs = (() => {
    const s = new Set();
    shapes.forEach(sh => sh.mValues.forEach(v => { if (v <= 16) s.add(v); }));
    return [...s].sort((a, b) => a - b);
  })();
  const largeMs = (() => {
    const s = new Set();
    shapes.forEach(sh => sh.mValues.forEach(v => { if (v > 16) s.add(v); }));
    return [...s].sort((a, b) => a - b);
  })();

  return html`
    <section class="panel ok-panel-full">
      <h2>GPU ${gpuIdx} Detail
        <span class="muted" style="font-size:0.7rem;margin-left:0.75rem;">${shapes.map(s => `TP=${s.tp}`).filter((v,i,a) => a.indexOf(v)===i).join(', ')}</span>
        <button class="btn btn-sm" style="float:right;" onClick=${onClose}>× close</button>
      </h2>

      <${StatsBar} detail=${detail} />

      <!-- Shapes being optimized -->
      ${shapes.length > 0 ? html`
        <div class="ok-shape-targets">
          ${shapes.map(s => {
            const hasDecode = s.mValues.some(v => v <= 16);
            const hasLarge = s.mValues.some(v => v > 16);
            return html`
              <div class="ok-shape-target" key=${s.label}>
                <div class="ok-shape-target-header">
                  <span class="ok-shape-target-name">${s.name}</span>
                  <span class="ok-badge" style="font-size:0.6rem;">TP=${s.tp}</span>
                  <span class="ok-badge muted-badge" style="margin-left:0.25rem;font-size:0.6rem;">${s.K}×${s.N}</span>
                </div>
                <div class="ok-shape-target-m">
                  ${hasDecode ? html`<span class="ok-shape-m-tag decode" title="Latency-critical">⚡ M≤16: ${s.mValues.filter(v => v <= 16).join(', ')}</span>` : null}
                  ${hasLarge ? html`<span class="ok-shape-m-tag large" title="Throughput-critical">📦 M=${s.mValues.filter(v => v > 16).join(', ')}</span>` : null}
                </div>
              </div>
            `;
          })}
        </div>
      ` : null}

      <!-- Main grid: State Machine | Kernel Library -->
      <div class="ok-grid">
        <section class="panel">
          <h2>State Machine</h2>
          <${StateGraph} graph=${detail.graph} />
        </section>
        <section class="panel">
          <h2>Kernel Library
            <span class="muted" style="font-size:0.75rem;margin-left:0.5rem;">(${detail.kernel_count} kernels)</span>
          </h2>
          <${KernelLibrary}
            library=${detail.library}
            selectedKernelId=${selectedKernel && selectedKernel.id}
            onSelectKernel=${(k) => setSelectedKernel(selectedKernel && selectedKernel.id === k.id ? null : k)}
          />
        </section>
      </div>

      <!-- Kernel Detail (when selected) -->
      ${selectedKernel ? html`
        <div class="ok-grid" style="margin-top:1rem;">
          <section class="panel ok-panel-full">
            <h2>Kernel: ${selectedKernel.id.slice(0, 8)}…
              <span class="muted" style="font-size:0.75rem;margin-left:0.5rem;">
                ${formatMs(selectedKernel.exec_time_ms)} | Cmplx ${((selectedKernel.complexity_score || 0) * 100).toFixed(0)}% | Iter ${selectedKernel.iteration_added || 0}
              </span>
              <button class="btn btn-sm" style="float:right;" onClick=${() => setSelectedKernel(null)}>× close</button>
            </h2>
            <${KernelCodeView} kernel=${selectedKernel} optimizerMode=${detail && detail.optimizer_mode} />
          </section>
        </div>

        ${lineageLoading ? html`<section class="panel ok-panel-full"><p class="muted">Loading analysis…</p></section>`
          : kernelLineage && !kernelLineage.error ? html`
          <div class="ok-grid">
            <!-- Roofline (left) -->
            <section class="panel">
              <h2>Roofline
                ${headroom && headroom.shape_label ? html`<span class="muted" style="font-size:0.7rem;margin-left:0.5rem;">(on ${headroom.shape_label})</span>` : null}
              </h2>
              <${RooflineCallout} headroom=${headroom} />

              ${profiling && profiling.profiled ? html`
                <div class="ok-profiling-badges" style="margin-top:0.5rem;">
                  <div class="ok-profiling-badge"><span class="label">hipprof</span> <span class="value">${profiling.profiling_kernel_duration_us ? profiling.profiling_kernel_duration_us.toFixed(0) + " µs" : "—"}</span></div>
                  <div class="ok-profiling-badge"><span class="label">BW</span> <span class="value">${profiling.profiling_achieved_bw_gbps ? profiling.profiling_achieved_bw_gbps.toFixed(1) + " GB/s" : "—"}</span></div>
                  <div class="ok-profiling-badge"><span class="label">Occ</span> <span class="value">${profiling.profiling_occupancy_pct ? profiling.profiling_occupancy_pct.toFixed(0) + "%" : "—"}</span></div>
                  <div class="ok-profiling-badge"><span class="label">L2$</span> <span class="value">${profiling.profiling_l2_cache_hit_pct ? profiling.profiling_l2_cache_hit_pct.toFixed(0) + "%" : "—"}</span></div>
                </div>
              ` : null}
            </section>

            <!-- Lineage (right) -->
            ${improvement ? html`
              <section class="panel">
                <h2>Optimization
                  <span class="muted" style="font-size:0.7rem;margin-left:0.5rem;">vs ${parent ? parent.id.slice(0,8) + '…' : 'seed'}</span>
                </h2>
                <div class="ok-lineage-metrics-compact">
                  <div class="ok-lm-item"><span class="ok-lm-val">${parent && parent.exec_time_ms ? parent.exec_time_ms.toFixed(4) + " ms" : "—"}</span><span class="ok-lm-lbl">Parent Time</span></div>
                  <div class="ok-lm-item"><span class=${"ok-lm-val " + (improvement.speedup_vs_parent && improvement.speedup_vs_parent > 1 ? "ok-score-good" : improvement.speedup_vs_parent && improvement.speedup_vs_parent < 1 ? "ok-score-bad" : "")}>${improvement.speedup_vs_parent ? improvement.speedup_vs_parent.toFixed(2) + "×" : "—"}</span><span class="ok-lm-lbl">Speedup</span></div>
                  <div class="ok-lm-item"><span class=${"ok-lm-val " + (improvement.exec_time_delta_ms != null && improvement.exec_time_delta_ms < 0 ? "ok-score-good" : "")}>${improvement.exec_time_delta_ms != null ? (improvement.exec_time_delta_ms > 0 ? "+" : "") + improvement.exec_time_delta_ms.toFixed(4) + " ms" : "—"}</span><span class="ok-lm-lbl">Delta</span></div>
                </div>

                ${improvement.summary_items && improvement.summary_items.length > 0 ? html`
                  <div class="ok-lineage-summary" style="margin-top:0.5rem;">
                    <ul class="ok-lineage-summary-items">
                      ${improvement.summary_items.map(item => html`<li>${item}</li>`)}
                    </ul>
                  </div>
                ` : improvement.summary ? html`
                  <div class="ok-lineage-summary" style="margin-top:0.5rem;">
                    <div class="ok-lineage-summary-text">${improvement.summary}</div>
                  </div>
                ` : null}
              </section>
            ` : html`<section class="panel"><h2>Optimization</h2><p class="muted">Seed kernel — no parent to compare.</p></section>`}
          </div>
        ` : kernelLineage && kernelLineage.error ? html`
          <section class="panel ok-panel-full"><p class="muted">Analysis not available for this kernel.</p></section>
        ` : null}
      ` : null}

      <!-- Harnesses + Agents -->
      <div class="ok-grid" style="margin-top:1rem;">
        <section class="panel">
          <h2>Harnesses</h2>
          <div style="display:flex;gap:0.5rem;margin-bottom:0.5rem;flex-wrap:wrap;align-items:center;">
            <span class=${"ok-harness-status " + (correctnessReady ? "pass" : correctnessReady === false ? "fail" : "pending")}>${correctnessReady ? "✓ Correctness" : correctnessReady === false ? "✗ Correctness" : "… Correctness"}</span>
            <span class=${"ok-harness-status " + (perfReady ? "pass" : perfReady === false ? "fail" : "pending")}>${perfReady ? "✓ Perf" : perfReady === false ? "✗ Perf" : "… Perf"}</span>
          </div>
          <div style="display:flex;gap:0.5rem;">
            <button class="btn btn-sm" onClick=${() => setShowHarness(showHarness === "correctness" ? null : "correctness")}>${showHarness === "correctness" ? "Hide" : "Show"} Correctness</button>
            <button class="btn btn-sm" onClick=${() => setShowHarness(showHarness === "perf" ? null : "perf")}>${showHarness === "perf" ? "Hide" : "Show"} Perf</button>
          </div>
          ${showHarness && harnessData[showHarness] && harnessData[showHarness].exists ? html`
            <div class="ok-code-preview" style="margin-top:0.5rem;max-height:300px;">
              ${(harnessData[showHarness].code || "").split("\n").map(line => html`<div class="ok-code-line">${line}</div>`)}
            </div>
          ` : showHarness && harnessData[showHarness] && !harnessData[showHarness].exists ? html`
            <p class="muted" style="margin-top:0.5rem;">Harness not yet generated.</p>
          ` : showHarness ? html`<p class="muted" style="margin-top:0.5rem;">Loading…</p>` : null}
        </section>

        <section class="panel">
          <h2>Agents</h2>
          <div class="ok-agent-list-scroll" style="max-height:200px;overflow-y:auto;">
            ${(detail.agents || []).length === 0 ? html`<p class="muted">No agents yet.</p>` : null}
            ${(detail.agents || []).map(a => html`
              <div key=${a.name} class=${"ok-mgpu-agent-row " + (a.phase === "running" ? "agent-running" : a.success === false ? "agent-failed" : "agent-done")} style="padding:3px 4px;">
                <span class="ok-mgpu-agent-role">${roleLabel(a.role)}</span>
                <span class="ok-mgpu-agent-status">
                  ${a.phase === "running" ? html`<span style="color:#3fb950;">● ${formatAge(a.elapsed_s)}</span>`
                    : a.success === false ? html`<span style="color:#f85149;">✗</span>`
                    : a.success === true ? html`<span style="color:#3fb950;">✓</span>`
                    : html`<span class="muted">${a.phase}</span>`}
                </span>
              </div>
            `)}
          </div>
        </section>
      </div>
    </section>
  `;
}

// ---- Aggregated Shape Benchmark ----

function AggregatedBenchTable({ benchData }) {
  if (!benchData || !benchData.results || benchData.results.length === 0) {
    return html`<p class="muted">No benchmark data yet. Run shape benchmarks from the task page.</p>`;
  }
  const results = benchData.results;
  const groups = {};
  for (const r of results) {
    const key = r.shape_label || r.gpu_source || "?";
    if (!groups[key]) groups[key] = [];
    groups[key].push(r);
  }
  return html`
    <div class="ok-shape-bench">
      ${Object.entries(groups).map(([label, rows]) => html`
        <div class="ok-shape-group" key=${label}>
          <div class="ok-shape-group-label">${label}</div>
          <table class="ok-kernel-table ok-shape-table">
            <thead><tr><th>M</th><th>Baseline</th><th>Best</th><th>Speedup</th></tr></thead>
            <tbody>
              ${rows.map(r => html`
                <tr key=${label + "-" + r.M} class=${r.M <= 16 ? "row-decode" : r.M >= 4096 ? "row-large" : ""}>
                  <td><strong>${r.M}</strong></td>
                  <td>${formatMs(r.ref_ms)}</td>
                  <td class="ok-score-good">${formatMs(r.best_ms)}</td>
                  <td class=${speedupClass(r.speedup)}>${r.error ? html`<span class="muted" title=${r.error}>err</span>` : ((r.speedup || 0) > 0 ? r.speedup.toFixed(2) + "×" : "—")}</td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `)}
    </div>
  `;
}

// ---- Best Kernels Summary ----

function BestKernelsSummary({ benchData }) {
  if (!benchData || !benchData.kernels || benchData.kernels.length === 0) return null;
  return html`
    <section class="panel ok-panel-full">
      <h2>Best Kernels by GPU</h2>
      <div class="ok-kernel-table-wrap" style="max-height:200px;">
        <table class="ok-kernel-table">
          <thead><tr><th>Source</th><th>ID</th><th>Exec Time</th><th>Cmplx</th><th>Iter</th></tr></thead>
          <tbody>
            ${benchData.kernels.map(k => html`
              <tr key=${k.id}>
                <td class="muted">${k.source_workspace || k.gpu_label || "?"}</td>
                <td class="muted" style="font-size:0.72rem;font-family:monospace;">${(k.id || "").slice(0, 8)}</td>
                <td class="ok-score-good">${formatMs(k.exec_time_ms)}</td>
                <td>${((k.complexity_score || 0) * 100).toFixed(0)}%</td>
                <td class="muted">${k.iteration_added || 0}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

// ---- Main Multi-GPU Dashboard ----

export function MultiGpuDashboard({ taskId }) {
  const [gpuData, setGpuData] = useState(null);
  const [benchData, setBenchData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedGpu, setSelectedGpu] = useState(null);

  const refresh = useCallback(async () => {
    if (!taskId) return;
    try {
      const [gpu, bench] = await Promise.all([
        fetchJSON(`${BASE(taskId)}/gpu-status`).catch(() => null),
        fetchJSON(`${BASE(taskId)}/aggregate-bench`).catch(() => null),
      ]);
      if (gpu) setGpuData(gpu);
      if (bench) setBenchData(bench);
      setError(null);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, [taskId]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [taskId, refresh]);

  if (loading && !gpuData) {
    return html`<section class="panel ok-panel-full"><h2>Multi-GPU Optimization</h2><p class="muted">Loading…</p></section>`;
  }
  if (error && !gpuData) {
    return html`<section class="panel ok-panel-full"><h2>Multi-GPU Optimization</h2><p style="color:#f85149;">${error}</p></section>`;
  }

  const workers = gpuData ? gpuData.workers : [];
  const runningCount = workers.filter(w => w.running).length;

  return html`
    <section class="panel ok-panel-full">
      <h2>Multi-GPU Optimization
        <span class="muted" style="font-size:0.75rem;margin-left:0.5rem;">(${runningCount}/${workers.length} GPUs active)</span>
      </h2>
      <${GpuCards} workers=${workers} selectedGpu=${selectedGpu} onSelectGpu=${setSelectedGpu} />
    </section>

    ${selectedGpu != null ? html`
      <${GpuDetailPanel} taskId=${taskId} gpuIdx=${selectedGpu} onClose=${() => setSelectedGpu(null)} />
    ` : null}

    <${PerGpuAgents} workers=${workers} />

    <section class="panel ok-panel-full">
      <h2>Aggregated Shape Benchmarks</h2>
      <${AggregatedBenchTable} benchData=${benchData} />
    </section>

    <${BestKernelsSummary} benchData=${benchData} />
  `;
}
