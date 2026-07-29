// Inline iteration detail — shows sub-agent progress, oracle per-case results,
// oracle report markdown, retrospective, and c-repair-attempt content
// within the Oracle correctness panel.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { getIterationDetail } from "app/ke-runtime-api";
import { marked } from "marked";

const ROLE_LABELS = {
  planner: "Planner",
  implementer: "Implementer",
  c_debugger: "C-Debugger",
};

const ROLE_PILLS = {
  planner: "success",
  implementer: "phase-A_attempt_pure",
  c_debugger: "phase-C_consolidate",
};

// ---- Expandable markdown section --------------------------------------------

function MarkdownSection({ title, md, maxPreview = 300 }) {
  const [expanded, setExpanded] = useState(false);
  if (!md) return null;
  const clean = md.trim();
  const preview = clean.replace(/#{1,4}\s/g, "").slice(0, maxPreview);

  return html`
    <div class="iter-detail-section">
      <h4>${title}</h4>
      ${expanded
        ? html`<div class="arch-detail" style="max-height:600px"
              dangerouslySetInnerHTML=${{ __html: marked.parse(clean) }} />`
        : html`<p class="muted" style="font-size:12px;margin:0">${preview}${clean.length > maxPreview ? "…" : ""}</p>`}
      ${clean.length > maxPreview ? html`
        <button class="btn-detail" style="margin-top:6px" onClick=${() => setExpanded(!expanded)}>
          ${expanded ? "Collapse" : "Read full report"}
        </button>
      ` : null}
    </div>
  `;
}

// ---- Sub-agent card ---------------------------------------------------------

function SubAgentCard({ role, data }) {
  const [expanded, setExpanded] = useState(false);
  if (!data) return null;
  const label = ROLE_LABELS[role] || role;
  const pillCls = ROLE_PILLS[role] || "";
  const ok = data.success;
  const text = data.final_text_head || "";

  return html`
    <div class="sa-card">
      <div class="sa-card-header">
        <span class="pill ${pillCls}">${label}</span>
        <span class="pill ${ok ? "success" : "err"}">${ok ? "OK" : "FAIL"}</span>
        <span class="muted" style="font-size:11px">
          ${data.attempts != null ? html`${data.attempts} attempt${data.attempts > 1 ? "s" : ""}` : null}
          ${data.duration_s != null ? html` · ${data.duration_s.toFixed(0)}s` : null}
        </span>
      </div>
      ${text ? html`
        <div class="sa-summary">
          ${expanded
            ? html`<pre class="arch-text">${text}</pre>`
            : html`<p class="muted" style="font-size:12px;margin:4px 0">${text.slice(0, 200)}${text.length > 200 ? "…" : ""}</p>`}
          ${text.length > 200 ? html`
            <button class="btn-detail" onClick=${() => setExpanded(!expanded)}>
              ${expanded ? "Collapse" : "Read more"}
            </button>
          ` : null}
        </div>
      ` : null}
    </div>
  `;
}

// ---- Oracle cases table -----------------------------------------------------

function OracleCasesTable({ oracle }) {
  if (!oracle || !oracle.cases || !oracle.cases.length) return null;

  return html`
    <div class="oracle-cases">
      <h4 style="margin:0 0 4px;font-size:12px;font-weight:600;color:var(--muted)">Per-case Results</h4>
      <table class="ke-oracle-table">
        <thead>
          <tr>
            <th>Case</th>
            <th>Prompt</th>
            <th>Verdict</th>
            <th>Gate</th>
            <th>Latency</th>
          </tr>
        </thead>
        <tbody>
          ${oracle.cases.map((c) => html`
            <tr key=${c.case_id}>
              <td style="font-size:11px;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title=${c.case_id}>${c.case_id}</td>
              <td style="font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title=${c.prompt}>${(c.prompt || "").slice(0, 60)}</td>
              <td><span class="pill ${c.judge_verdict === "pass" ? "success" : "err"}">${c.judge_verdict || "?"}</span></td>
              <td><span class="pill ${c.gating === "hard" ? "err" : ""}">${c.gating || ""}</span></td>
              <td class="muted" style="font-size:11px">${c.elapsed_s != null ? c.elapsed_s.toFixed(1) + "s" : "—"}</td>
            </tr>
          `)}
        </tbody>
      </table>
      ${oracle.judge_mode ? html`
        <p class="muted" style="font-size:10px;margin-top:4px">Judge: ${oracle.judge_mode} · ${oracle.cases_passed || 0}/${oracle.cases_total || 0} passed · ${oracle.hard_passed || 0}/${oracle.hard_total || 0} hard</p>
      ` : null}
    </div>
  `;
}

// ---- C-repair card ----------------------------------------------------------

function CRepairCard({ repair }) {
  const [expanded, setExpanded] = useState(false);
  const md = repair.markdown || "";

  return html`
    <div class="sa-card" style="border-color:var(--accent)">
      <div class="sa-card-header">
        <span class="pill phase-C_consolidate">${repair.file}</span>
      </div>
      <div class="sa-summary">
        ${expanded
          ? html`<div class="arch-detail" style="max-height:500px"
                dangerouslySetInnerHTML=${{ __html: marked.parse(md) }} />`
          : html`<p class="muted" style="font-size:12px;margin:4px 0">${md.slice(0, 200).replace(/#/g, "")}${md.length > 200 ? "…" : ""}</p>`}
        <button class="btn-detail" onClick=${() => setExpanded(!expanded)}>
          ${expanded ? "Collapse" : "Read full repair"}
        </button>
      </div>
    </div>
  `;
}

function RepairEntryRow({ entry }) {
  const [expanded, setExpanded] = useState(false);
  const dbg = entry.debugger || {};
  const test = entry.test || {};

  return html`
    <div class="sa-card" style="border-color:var(--accent)">
      <div class="sa-card-header">
        <span class="pill phase-C_consolidate">Repair #${entry.attempt}</span>
        ${test.outcome ? html`<span class="pill ${test.outcome === "ok" ? "success" : "err"}">${test.outcome}</span>` : null}
        <span class="muted" style="font-size:11px">${dbg.duration_s != null ? dbg.duration_s.toFixed(0) + "s" : ""}</span>
      </div>
      ${entry.input_failure ? html`
        <p class="muted" style="font-size:11px;margin:4px 0"><strong>Input failure:</strong> ${entry.input_failure.slice(0, 200)}</p>
      ` : null}
      ${dbg.final_text_head ? html`
        <div class="sa-summary">
          ${expanded
            ? html`<pre class="arch-text">${dbg.final_text_head}</pre>`
            : html`<p class="muted" style="font-size:12px;margin:4px 0">${dbg.final_text_head.slice(0, 200)}…</p>`}
          <button class="btn-detail" onClick=${() => setExpanded(!expanded)}>
            ${expanded ? "Collapse" : "Read more"}
          </button>
        </div>
      ` : null}
      ${entry.note ? html`<p class="muted" style="font-size:11px;margin:4px 0">${entry.note}</p>` : null}
    </div>
  `;
}

// ---- Phase-level summary (when sub-agent detail is missing) -----------------

function PhaseSummary({ detail }) {
  // Show B_enrich / A_attempt_pure / D_verify_final phase outcomes when
  // sub-agent detail (planner/implementer/c_debugger) is not available.
  const phases = detail.phase_outcomes || {};
  const phaseKeys = ["A_attempt_pure", "B_enrich", "C_consolidate", "D_verify_final"];
  const entries = phaseKeys.filter((k) => phases[k]);

  if (!entries.length) return null;

  return html`
    <div class="iter-detail-section">
      <h4>Phase Outcomes</h4>
      <div class="sa-list">
        ${entries.map((k) => {
          const p = phases[k];
          const ok = p.outcome === "OK" || p.outcome === "ok";
          return html`
            <div class="sa-card" key=${k}>
              <div class="sa-card-header">
                <span class="pill ${k === "B_enrich" ? "warn" : k === "C_consolidate" ? "phase-C_consolidate" : k === "D_verify_final" ? "phase-D_verify_final" : "phase-A_attempt_pure"}">${k}</span>
                <span class="pill ${ok ? "success" : "err"}">${p.outcome || "?"}</span>
                ${p.attempts != null ? html`<span class="muted" style="font-size:11px">${p.attempts} attempt${p.attempts > 1 ? "s" : ""}</span>` : null}
              </div>
              ${p.failure ? html`
                <p class="muted" style="font-size:11px;margin:4px 0">${String(p.failure).slice(0, 300)}</p>
              ` : null}
              ${p.perf ? html`
                <p class="muted" style="font-size:10px;margin:2px 0">
                  ${Object.entries(p.perf).map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(1) : v}`).join(" · ")}
                </p>
              ` : null}
            </div>
          `;
        })}
      </div>
    </div>
  `;
}

// ---- Main component ---------------------------------------------------------

export function IterationDetail({ taskId, iteration }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!taskId || iteration == null) return;
    setLoading(true);
    setErr(null);
    getIterationDetail(taskId, iteration)
      .then((d) => { setDetail(d); setLoading(false); })
      .catch((e) => { setErr(String(e)); setLoading(false); });
  }, [taskId, iteration]);

  if (loading) return html`<p class="muted" style="padding:8px">Loading iteration detail…</p>`;
  if (err) return html`<p class="text-err" style="padding:8px">Failed to load: ${err}</p>`;
  if (!detail || !detail.found) return html`<p class="muted" style="padding:8px">No detail available for this iteration.</p>`;

  const sa = detail.sub_agents || {};
  const hasSubAgents = Object.keys(sa).length > 0;
  const phaseOut = detail.phase_outcomes || {};
  const hasPhaseOutcomes = Object.keys(phaseOut).length > 0;
  const hasOracle = detail.oracle && detail.oracle.cases && detail.oracle.cases.length > 0;
  const hasOracleMd = !!detail.oracle_md;
  const hasRetroMd = !!detail.retro_md;
  const hasRepairs = (detail.repairs || []).length > 0;
  const hasRepairEntries = (detail.repair_entries || []).length > 0;
  const hasContent = hasSubAgents || hasPhaseOutcomes || hasOracle || hasOracleMd || hasRetroMd || hasRepairs || hasRepairEntries;

  return html`
    <div class="iter-detail-inline">
      <div class="iter-detail-header">
        <span style="font-weight:600;font-size:13px">Iteration ${iteration}</span>
        ${detail.status ? html`<span class="pill ${detail.status === "success" ? "success" : detail.status === "failed" ? "err" : "warn"}">${detail.status}</span>` : null}
        ${detail.start_phase ? html`<span class="pill phase-${detail.start_phase}">${detail.start_phase}</span>` : null}
        ${detail.outcome ? html`<span class="pill ${detail.outcome === "ok" ? "success" : "err"}">${detail.outcome}</span>` : null}
        ${detail.failure_reason ? html`<span class="fail-reason" style="max-width:300px" title=${detail.failure_reason}>${detail.failure_reason}</span>` : null}
      </div>

      ${hasSubAgents ? html`
        <div class="iter-detail-section">
          <h4>Sub-agent Progress</h4>
          <div class="sa-list">
            ${Object.entries(sa).map(([role, data]) => html`
              <${SubAgentCard} key=${role} role=${role} data=${data} />
            `)}
          </div>
        </div>
      ` : hasPhaseOutcomes ? html`
        <${PhaseSummary} detail=${detail} />
      ` : null}

      ${hasOracle ? html`
        <div class="iter-detail-section">
          <${OracleCasesTable} oracle=${detail.oracle} />
        </div>
      ` : null}

      ${hasOracleMd ? html`
        <${MarkdownSection} title="Oracle Report" md=${detail.oracle_md} maxPreview=${300} />
      ` : null}

      ${hasRetroMd ? html`
        <${MarkdownSection} title="Retrospective" md=${detail.retro_md} maxPreview=${300} />
      ` : null}

      ${hasRepairs ? html`
        <div class="iter-detail-section">
          <h4>C-Repair Attempts</h4>
          ${detail.repairs.map((rp) => html`
            <${CRepairCard} key=${rp.file} repair=${rp} />
          `)}
        </div>
      ` : null}

      ${hasRepairEntries ? html`
        <div class="iter-detail-section">
          <h4>Repair Audit Log</h4>
          ${detail.repair_entries.map((entry, i) => html`
            <${RepairEntryRow} key=${i} entry=${entry} />
          `)}
        </div>
      ` : null}

      ${!hasContent ? html`
        <p class="muted" style="padding:8px">No detailed data for this iteration. The iteration may still be in progress, or no oracle/repair artifacts were produced.</p>
      ` : null}
    </div>
  `;
}
