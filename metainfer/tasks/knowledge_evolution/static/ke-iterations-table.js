// Knowledge-evolution iterations table.
// Each row shows iteration #, phase, goal, status, outcome, duration.
// Clicking a row expands failure reason details inline.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { labelFor } from "app/utils";

const NOW = () => Date.now() / 1000;

function fmtDur(s, startedAt) {
  // For running iterations without duration_s, show elapsed time
  if (s == null || s <= 0) {
    if (startedAt > 0) {
      const elapsed = NOW() - startedAt;
      if (elapsed < 5) return "…";
      if (elapsed < 60) return `${elapsed.toFixed(0)}s`;
      const m = Math.floor(elapsed / 60);
      const sec = Math.floor(elapsed % 60);
      return `${m}m ${sec}s`;
    }
    return "—";
  }
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}m ${sec}s`;
}

function phasePill(phase) {
  if (!phase) return html`<span class="muted">—</span>`;
  return html`<span class="pill ${phase}">${labelFor(phase)}</span>`;
}

export function IterationsTable({ iterations, selectedN, onSelect }) {
  const [_, force] = useState(0);
  // Re-render periodically so elapsed times update for running iterations
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);

  const rows = (iterations || []).map((r) => {
    const phase = r.current_phase || r.start_phase || "";
    const status = r.interrupted ? "interrupted" : (r.status || "");
    const outcome = r.last_outcome || r.outcome || "";
    const n = r.n || r.iteration;
    const isSel = n === selectedN;
    return html`
      <tr key=${n} class=${isSel ? "selected" : ""}
          onClick=${() => onSelect && onSelect(n)}>
        <td><strong>${n}</strong></td>
        <td>${phasePill(phase)}</td>
        <td><span class="pill ${status}">${status || "running"}</span></td>
        <td>${(r.agent_status || "").replace("running: ", "") || "—"}</td>
        <td>${outcome || "—"}</td>
        <td>${fmtDur(r.duration_s, r.started_at)}</td>
        <td class="fail-reason">${r.failure_reason || ""}</td>
      </tr>
    `;
  });

  if (rows.length === 0) {
    return html`<p class="muted">No iterations yet. The orchestrator writes iteration records as the pipeline loops.</p>`;
  }

  return html`
    <div class="iter-table">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Phase</th>
            <th>Status</th>
            <th>Agent</th>
            <th>Outcome</th>
            <th>Dur</th>
            <th>Failure</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}
