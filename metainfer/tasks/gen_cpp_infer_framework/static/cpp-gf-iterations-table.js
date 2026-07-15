// Iterations table with clickable rows. Clicking opens the retrospective
// modal (handled by the parent, so this component just calls onSelect).

import { html } from "htm/preact";
import { labelFor, fmtDur, escapeHtml } from "app/utils";

function perfCell(rec) {
  const perf = rec.perf || {};
  const parts = Object.entries(perf).map(
    ([k, v]) => `${k}=${(+v).toFixed(2)}`,
  );
  const has = !!rec.retrospective_path;
  const dot = has
    ? html`<span style="color:var(--accent); margin-left:6px">●</span>`
    : null;
  return html`<span class="perf-cell">${parts.join(", ") || "—"}${dot}</span>`;
}

export function IterationsTable({ iterations, selectedN, onSelect }) {
  const rows = (iterations || []).map((r) => {
    const statusLabel = r.interrupted ? "interrupted" : r.status;
    const statusCls = r.interrupted ? "interrupted" : r.status;
    const cls = r.iteration === selectedN ? "selected" : "";
    return html`
      <tr key=${r.iteration} class=${cls}
          onClick=${() => onSelect && onSelect(r.iteration)}>
        <td><strong>${r.iteration}</strong></td>
        <td><span class="pill ${r.start_phase}">${labelFor(r.start_phase)}</span></td>
        <td>${r.goal || "(no goal recorded)"}</td>
        <td><span class="pill ${statusCls}">${statusLabel}</span></td>
        <td>${perfCell(r)}</td>
        <td>${fmtDur(r.duration_s)}</td>
        <td class="fail-reason">${r.failure_reason || ""}</td>
      </tr>
    `;
  });

  if (rows.length === 0) {
    return html`
      <div class="iter-table empty">
        <p class="muted">No iterations yet. The orchestrator writes iteration
          records as the pipeline loops.</p>
      </div>
    `;
  }

  return html`
    <div class="iter-table">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Start</th>
            <th>Goal</th>
            <th>Status</th>
            <th>Perf</th>
            <th>Dur</th>
            <th>Failure reason</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}
