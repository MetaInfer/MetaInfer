// Knowledge gained panel — shows what consolidator (phase C) wrote into notebooks/
// plus model architecture knowledge from plan.md and existing notebooks.

import { html } from "htm/preact";
import { useState } from "preact/hooks";

// ---- Detail modal for viewing full architecture / notebook content ---------

function TextDetailModal({ title, text, onClose }) {
  return html`
    <div class="modal-overlay" onClick=${onClose}>
      <div class="modal-content modal-code" onClick=${(e) => e.stopPropagation()}>
        <div class="modal-header">
          <span class="modal-title">${title}</span>
          <button class="modal-close" onClick=${onClose}>&times;</button>
        </div>
        <div class="modal-body">
          <pre class="code-block"><code>${text}</code></pre>
        </div>
      </div>
    </div>
  `;
}

// ---- Architecture card (condensed) -----------------------------------------

function ArchitectureCard({ entry }) {
  const [expanded, setExpanded] = useState(false);

  return html`
    <div class="knowledge-entry" key=${"arch-" + entry.iteration}>
      <div class="knowledge-entry-header">
        <span class="knowledge-iter">iter ${entry.iteration}</span>
        <span class="pill phase-B_enrich">plan</span>
      </div>
      ${entry.key_points && entry.key_points.length > 0 ? html`
        <ul class="arch-key-points">
          ${entry.key_points.map((pt) => html`<li key=${pt}>${pt}</li>`)}
        </ul>
      ` : html`<p class="muted">No architecture summary found in plan.md.</p>`}
      <button class="btn-detail"
        onClick=${() => setExpanded(!expanded)}>
        ${expanded ? "Collapse" : "View Detail"}
      </button>
      ${expanded ? html`
        <div class="arch-detail">
          <pre class="arch-text">${entry.architecture}</pre>
        </div>
      ` : null}
    </div>
  `;
}

// ---- Notebook entry card ---------------------------------------------------

function NotebookCard({ entry }) {
  const [expanded, setExpanded] = useState(false);
  const preview = entry.text
    .replace(/^#.*\n/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
    .slice(0, 300);

  return html`
    <div class="knowledge-entry" key=${entry.path}>
      <div class="knowledge-entry-header">
        <span class="knowledge-iter">${entry.title}</span>
        <span class="pill phase-C_consolidate">notebook</span>
      </div>
      <p class="knowledge-summary">${preview}${entry.text.length > 300 ? "…" : ""}</p>
      <button class="btn-detail"
        onClick=${() => setExpanded(!expanded)}>
        ${expanded ? "Collapse" : "View Detail"}
      </button>
      ${expanded ? html`
        <div class="arch-detail">
          <pre class="arch-text">${entry.text}</pre>
        </div>
      ` : null}
    </div>
  `;
}

// ---- Section wrapper with expand/collapse ----------------------------------

function CollapsibleSection({ title, count, defaultOpen, children }) {
  const [open, setOpen] = useState(!!defaultOpen);

  return html`
    <div class="knowledge-section">
      <div class="knowledge-section-header" onClick=${() => setOpen(!open)} style="cursor:pointer;display:flex;align-items:center;gap:8px">
        <span class="ke-expand-arrow">${open ? "▾" : "▸"}</span>
        <h3 style="margin:0">${title}</h3>
        <span class="muted" style="font-size:11px">(${count} ${count === 1 ? "entry" : "entries"})</span>
      </div>
      ${open ? html`
        <div class="knowledge-list" style="margin-top:8px">${children}</div>
      ` : null}
    </div>
  `;
}

// ---- Main panel -------------------------------------------------------------

export function KnowledgeGainedPanel({ payload, modelCode, taskId, onViewDiff }) {
  const hasKnowledge = payload && payload.iterations && payload.iterations.length > 0;
  const modelData = modelCode || {};
  const hasArchitecture = modelData.iterations && modelData.iterations.length > 0;
  const hasNotebooks = modelData.notebook_entries && modelData.notebook_entries.length > 0;

  if (!hasKnowledge && !hasArchitecture && !hasNotebooks) {
    return html`<p class="muted">No knowledge has been distilled yet. The
      consolidator (Phase C) writes new knowledge to notebooks/ after each
      successful enrichment cycle.</p>`;
  }

  return html`
    <div class="knowledge-layout">

      ${hasKnowledge ? html`
        <${CollapsibleSection} title="New Knowledge" count=${payload.iterations.length} defaultOpen=${true}>
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
        </${CollapsibleSection}>
      ` : null}

      ${hasArchitecture ? html`
        <${CollapsibleSection} title="Model Architecture" count=${modelData.iterations.length} defaultOpen=${!hasKnowledge}>
          ${modelData.iterations.map((entry) => html`
            <${ArchitectureCard} entry=${entry} />
          `)}
        </${CollapsibleSection}>
      ` : null}

      ${hasNotebooks ? html`
        <${CollapsibleSection} title="Existing Knowledge Base" count=${modelData.notebook_entries.length}>
          ${modelData.notebook_entries.map((entry) => html`
            <${NotebookCard} entry=${entry} />
          `)}
        </${CollapsibleSection}>
      ` : null}

    </div>
  `;
}
