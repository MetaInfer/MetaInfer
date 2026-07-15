// Retrospective modal. Renders the markdown for an iteration. The
// backend returns a sensible placeholder when no retro file exists, so
// the modal is always informative.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { marked } from "marked";
import { getRetrospective } from "app/cpp-gf-runtime-api";

export function RetrospectiveModal({ taskId, iteration, onClose }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (taskId == null || iteration == null) return;
    let alive = true;
    setData(null);
    setErr(null);
    getRetrospective(taskId, iteration)
      .then((d) => { if (alive) setData(d); })
      .catch((e) => { if (alive) setErr(e); });
    return () => { alive = false; };
  }, [taskId, iteration]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose && onClose(); };
    if (iteration != null) document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [iteration, onClose]);

  if (iteration == null) return null;

  let body;
  if (err) {
    body = html`<p style="color:var(--err)">failed to load retrospective: ${String(err)}</p>`;
  } else if (!data) {
    body = html`<p class="muted">loading…</p>`;
  } else {
    const html_ = marked.parse(data.markdown || "");
    body = html`<div class="retro-md" dangerouslySetInnerHTML=${{ __html: html_ }}></div>`;
  }

  return html`
    <div class="modal-overlay open" onClick=${(e) => {
      if (e.target === e.currentTarget) onClose && onClose();
    }}>
      <div class="modal">
        <div class="modal-header">
          <h3>
            Iteration ${iteration} retrospective
            ${data && !data.has_retrospective
              ? html`<span class="retro-flag missing">no retro file</span>`
              : null}
          </h3>
          <button class="close" onClick=${onClose} title="close (Esc)">×</button>
        </div>
        <div class="modal-body">${body}</div>
      </div>
    </div>
  `;
}
