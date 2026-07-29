// Retrospective modal — shows the retrospective.md written by the retro_writer agent.
// Opened when clicking an iteration row in the iterations table.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { getRetrospective } from "app/ke-runtime-api";
import { marked } from "marked";

export function RetrospectiveModal({ taskId, iteration, onClose }) {
  const [md, setMd] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!taskId || iteration == null) return;
    setLoading(true);
    setErr(null);
    getRetrospective(taskId, iteration)
      .then((d) => { setMd(d); setLoading(false); })
      .catch((e) => { setErr(String(e)); setLoading(false); });
  }, [taskId, iteration]);

  useEffect(() => {
    const h = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, [onClose]);

  return html`
    <div class="modal-overlay" onClick=${(e) => e.target === e.currentTarget && onClose()}>
      <div class="modal">
        <div class="modal-header">
          <h3>Retrospective · iter ${iteration}</h3>
          <button class="close" onClick=${onClose}>×</button>
        </div>
        <div class="modal-body">
          ${loading
            ? html`<p class="muted">Loading retrospective…</p>`
            : err
              ? html`<p class="text-err">Failed to load: ${err}</p>`
              : md && md.found
                ? html`<div class="retro-md"
                    dangerouslySetInnerHTML=${{ __html: marked.parse(md.markdown) }} />`
                : html`<p class="muted">No retrospective written for this iteration yet.</p>`}
        </div>
      </div>
    </div>
  `;
}
