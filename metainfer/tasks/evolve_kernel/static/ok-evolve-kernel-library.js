// Kernel library table component for evolve-kernel.
// Shows ranked kernels with exec_time, complexity, combined_score.
// Color-coded rows by speedup, headroom badge, combined score bar.

import { html } from "htm/preact";
import { useState } from "preact/hooks";

function scoreClass(val) {
  if (val == null || val === 0) return "";
  if (val > 0.8) return "ok-score-good";
  if (val > 0.4) return "ok-score-ok";
  return "ok-score-bad";
}

function formatMs(ms) {
  if (ms == null || ms === 0) return "—";
  if (ms < 1) return (ms * 1000).toFixed(1) + " μs";
  if (ms < 1000) return ms.toFixed(3) + " ms";
  return (ms / 1000).toFixed(2) + " s";
}

function speedupRowClass(k) {
  const su = k.speedup_vs_parent;
  if (su == null || su <= 0) return "";
  if (su >= 1.2) return "speedup-good";
  if (su < 1.0) return "speedup-bad";
  return "";
}

function headroomBadge(headroom) {
  if (!headroom || !headroom.bottleneck) return null;
  const b = headroom.bottleneck;
  const labels = {
    near_optimal: "Near-Opt",
    memory_bound: "Mem-Bound",
    compute_bound: "Comp-Bound",
    inefficient: "Ineff",
  };
  return html`<span class=${"ok-hr-badge " + b} title=${"BW " + (headroom.bw_util_pct || 0).toFixed(0) + "% / TFLOP " + (headroom.compute_util_pct || 0).toFixed(0) + "%"}>${labels[b] || b}</span>`;
}

function combinedBar(score, maxScore) {
  const pct = maxScore > 0 ? Math.min(100, (score / maxScore) * 100) : 0;
  return html`
    <span class="ok-combined-bar"><span class="ok-combined-bar-fill" style=${{ width: pct + "%" }}></span></span>
    <span style="font-size:0.75rem;">${score.toFixed(2)}</span>
  `;
}

export function KernelLibrary({ library, selectedKernelId, onSelectKernel }) {
  if (!library || !library.kernels || library.kernels.length === 0) {
    return html`<p class="muted">Kernel library is empty — waiting for the first optimization result.</p>`;
  }

  const kernels = library.kernels;
  const maxScore = Math.max(...kernels.map(k => k.combined_score || 0), 1);

  return html`
    <div class="ok-kernel-table-wrap">
      <table class="ok-kernel-table">
        <thead>
          <tr>
            <th>#</th>
            <th>ID</th>
            <th>Exec Time</th>
            <th>Δ Parent</th>
            <th>Score</th>
            <th>Cmplx</th>
            <th>Iter</th>
            <th>Lines</th>
            <th>Headroom</th>
          </tr>
        </thead>
        <tbody>
          ${kernels.map((k, i) => {
            const headroom = k.headroom;
            const hasProfile = k.profiled;
            return html`
              <tr
                key=${k.id}
                class=${(k.id === selectedKernelId ? "selected" : "") + " " + speedupRowClass(k)}
                onClick=${() => onSelectKernel && onSelectKernel(k)}
                style=${{ cursor: onSelectKernel ? "pointer" : "default" }}
              >
                <td><strong>${i + 1}</strong></td>
                <td class="muted" style="font-size:0.72rem;font-family:monospace;">${k.id.slice(0, 8)}</td>
                <td class="ok-score-good" style="font-variant-numeric:tabular-nums;">${formatMs(k.exec_time_ms)}</td>
                <td>
                  ${k.speedup_vs_parent
                    ? html`<span class=${k.speedup_vs_parent >= 1.2 ? "ok-score-good" : k.speedup_vs_parent >= 1.0 ? "ok-score-ok" : "ok-score-bad"}>${k.speedup_vs_parent.toFixed(2)}×</span>`
                    : html`<span class="muted">—</span>`}
                </td>
                <td style="white-space:nowrap;">${combinedBar(k.combined_score, maxScore)}</td>
                <td>${((k.complexity_score || 0) * 100).toFixed(0)}%</td>
                <td class="muted">${k.iteration_added || 0}</td>
                <td class="muted" style="font-size:0.72rem;">${k.code_lines || 0}</td>
                <td>${headroomBadge(headroom)}${hasProfile ? html`<span title="hipprof profiled" style="margin-left:2px;font-size:0.6rem;">📊</span>` : null}</td>
              </tr>
            `;
          })}
        </tbody>
      </table>
    </div>
    <p class="muted" style="margin-top:0.4rem;font-size:0.7rem;">${library.size} kernel(s) ranked by combined score —
      <span style="color:#3fb950;">▲</span> ≥1.2× speedup,
      <span style="color:#f85149;">▼</span> regression
    </p>
  `;
}

// Kernel code viewer:
// - HIP mode: defaults to .cpp source, toggle to .py wrapper
// - Triton mode: shows .py source directly
export function KernelCodeView({ kernel, optimizerMode }) {
  const isHip = optimizerMode && (
    optimizerMode === "HIP C++ (from scratch)" ||
    optimizerMode === "hip_cpp" ||
    optimizerMode === "hip"
  );
  const cppCode = kernel.cpp_code || null;
  const pyCode = kernel.code || kernel.code_preview || null;
  const hasBoth = cppCode && pyCode;

  // Default: .cpp for HIP mode, .py for Triton mode
  const [showCpp, setShowCpp] = useState(isHip);

  const activeCode = showCpp && cppCode ? cppCode : pyCode;
  const cppLines = cppCode ? cppCode.split("\n").length : 0;
  const pyLines = pyCode ? pyCode.split("\n").length : 0;

  if (!activeCode) return html`<p class="muted">Code not available</p>`;

  const lines = activeCode.split("\n");

  return html`
    <div>
      ${hasBoth ? html`
        <div style="margin-bottom:0.5rem;display:flex;gap:0.3rem;">
          <button class="btn btn-sm ${showCpp ? "btn-primary" : ""}"
                  onClick=${() => setShowCpp(true)}>
            HIP C++ (.cpp) — ${cppLines} lines
          </button>
          <button class="btn btn-sm ${!showCpp ? "btn-primary" : ""}"
                  onClick=${() => setShowCpp(false)}>
            Python wrapper (.py) — ${pyLines} lines
          </button>
        </div>
      ` : cppCode ? html`
        <p class="muted" style="font-size:0.75rem;margin-bottom:0.3rem;">
          HIP C++ kernel source (${cppLines} lines)
        </p>
      ` : null}
      <div class="ok-code-preview large">
        ${lines.map(line => html`<div class="ok-code-line">${line}</div>`)}
      </div>
    </div>
  `;
}
