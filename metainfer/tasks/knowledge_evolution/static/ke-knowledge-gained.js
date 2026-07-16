// Knowledge gained panel — shows what consolidator (phase C) wrote into notebooks/.

import { html } from "htm/preact";

export function KnowledgeGainedPanel({ payload, taskId, onViewDiff }) {
  if (!payload || !payload.iterations || payload.iterations.length === 0) {
    return html`<p class="muted">No knowledge has been distilled yet. The
      consolidator (Phase C) writes new knowledge to notebooks/ after each
      successful enrichment cycle.</p>`;
  }
  return html`
    <div class="knowledge-list">
      ${payload.iterations.map((entry) => html`
        <div class="knowledge-entry" key=${entry.iteration}>
          <div class="knowledge-entry-header">
            <span class="knowledge-iter">iter ${entry.iteration}</span>
            <span class="pill ${entry.status || "success"}">${entry.status || "completed"}</span>
          </div>
          ${entry.summary ? html`
            <p class="knowledge-summary">${entry.summary}</p>
          ` : null}
          ${entry.files && entry.files.length > 0 ? html`
            <ul class="knowledge-files">
              ${entry.files.map((f) => html`
                <li key=${f}>
                  <a class="knowledge-file-link"
                    onClick=${(e) => { e.preventDefault(); onViewDiff(entry.iteration, f); }}>
                    <code>${f}</code>
                  </a>
                </li>`)}
            </ul>
          ` : null}
        </div>
      `)}
    </div>
  `;
}
