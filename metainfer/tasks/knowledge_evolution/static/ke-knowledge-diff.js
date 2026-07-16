// Knowledge diff modal — shows file content from notebooks/ distilled by the consolidator.
// Renders with diff-like color coding: green for +, red for -, blue for @@ hunks.
// Falls back to plain code rendering when the content isn't a valid diff.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { getKnowledgeDiff } from "app/ke-runtime-api";

function renderDiffLines(text) {
  if (!text) return html`<p class="muted">Empty file.</p>`;
  const lines = text.split("\n");
  // Detect if this looks like a unified diff (has @@ hunk headers)
  const isDiff = lines.some((l) => l.startsWith("@@"));
  if (isDiff) {
    return html`
      <pre class="diff-content"><code>${lines.map((line, i) => {
        let cls = "diff-line";
        if (line.startsWith("+") && !line.startsWith("+++")) cls += " diff-add";
        else if (line.startsWith("-") && !line.startsWith("---")) cls += " diff-rem";
        else if (line.startsWith("@@")) cls += " diff-hunk";
        return html`<div key=${i} class=${cls}>${line}</div>`;
      })}</code></pre>
    `;
  }
  // Plain text fallback
  return html`<pre class="diff-content"><code>${text}</code></pre>`;
}

export function KnowledgeDiffModal({ taskId, iteration, file, onClose }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!taskId || iteration == null || !file) return;
    setLoading(true);
    setErr(null);
    getKnowledgeDiff(taskId, iteration, file)
      .then((d) => { setContent(d); setLoading(false); })
      .catch((e) => { setErr(String(e)); setLoading(false); });
  }, [taskId, iteration, file]);

  useEffect(() => {
    const h = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, [onClose]);

  return html`
    <div class="modal-overlay" onClick=${(e) => e.target === e.currentTarget && onClose()}>
      <div class="modal">
        <div class="modal-header">
          <h3>Knowledge diff · iter ${iteration}</h3>
          <button class="close" onClick=${onClose}>×</button>
        </div>
        <div class="modal-body">
          ${loading
            ? html`<p class="muted">Loading file…</p>`
            : err
              ? html`<p class="text-err">Failed to load: ${err}</p>`
              : content && content.found
                ? html`
                  <p class="muted" style="margin:0 0 var(--sp-3)"><code>${content.file || file}</code></p>
                  ${renderDiffLines(content.diff || content.text || "(empty)")}
                `
                : html`<p class="muted">File not found.</p>`}
        </div>
      </div>
    </div>
  `;
}
