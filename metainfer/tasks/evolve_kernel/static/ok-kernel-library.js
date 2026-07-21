// Kernel library table component for evolve-kernel.
// Shows ranked kernels with exec_time, complexity, combined_score.

import { html } from "htm/preact";

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

export function KernelLibrary({ library, selectedKernelId, onSelectKernel }) {
  if (!library || !library.kernels || library.kernels.length === 0) {
    return html`<p class="muted">Kernel library is empty — waiting for the first optimization result.</p>`;
  }

  const kernels = library.kernels;

  return html`
    <table class="ok-kernel-table">
      <thead>
        <tr>
          <th>#</th>
          <th>ID</th>
          <th>Exec Time</th>
          <th>Complexity</th>
          <th>Combined</th>
          <th>Iter</th>
          <th>Lines</th>
        </tr>
      </thead>
      <tbody>
        ${kernels.map((k, i) => html`
          <tr
            key=${k.id}
            class=${k.id === selectedKernelId ? "selected" : ""}
            onClick=${() => onSelectKernel && onSelectKernel(k)}
            style=${{ cursor: onSelectKernel ? "pointer" : "default" }}
          >
            <td><strong>${i + 1}</strong></td>
            <td class="muted" style="font-size:0.75rem;">${k.id.slice(0, 8)}</td>
            <td class="ok-score-good">${formatMs(k.exec_time_ms)}</td>
            <td>${(k.complexity_score * 100).toFixed(0)}%</td>
            <td class=${scoreClass(k.combined_score)}>${k.combined_score.toFixed(4)}</td>
            <td class="muted">${k.iteration_added || 0}</td>
            <td class="muted">${k.code_lines || 0}</td>
          </tr>
        `)}
      </tbody>
    </table>
    <p class="muted" style="margin-top:0.5rem;font-size:0.75rem;">${library.size} kernel(s) in library (max 10) — ranked by combined score</p>
  `;
}
