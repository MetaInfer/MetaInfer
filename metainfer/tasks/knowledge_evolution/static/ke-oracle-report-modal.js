// Oracle report modal — fetches and renders the oracle's correctness report markdown.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { getOracleReport } from "app/ke-runtime-api";
import { marked } from "marked";

export function OracleReportModal({ taskId, iteration, onClose }) {
  const [md, setMd] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!taskId || iteration == null) return;
    setLoading(true);
    setErr(null);
    getOracleReport(taskId, iteration)
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
          <h3>Oracle report · iter ${iteration}</h3>
          <button class="close" onClick=${onClose}>×</button>
        </div>
        <div class="modal-body">
          ${loading
            ? html`<p class="muted">Loading oracle report…</p>`
            : err
              ? html`<p class="text-err">Failed to load: ${err}</p>`
              : md && md.found
                ? html`<div class="retro-md"
                    dangerouslySetInnerHTML=${{ __html: marked.parse(md.markdown) }} />`
                : html`<p class="muted">No oracle report available for this iteration.</p>`}
        </div>
      </div>
    </div>
  `;
}
