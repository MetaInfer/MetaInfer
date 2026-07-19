/**
 * port-model detail view — Preact component mounted inside the sys-shell
 * task-detail chrome. Shows phase graph, memory docs, diff, and test results.
 *
 * Importmap key: app/pm-detail
 * Shell passes: { run, timeline, agents, loadState, lastErr, refreshShell }
 */
import { h, render, useState, useEffect, useCallback } from "/static/vendor/preact.js";
import htm from "/static/vendor/htm.js";
const html = htm.bind(h);

/* ---- helpers ----------------------------------------------------------- */

const API = (taskId) => `/api/port-model/${taskId}`;

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.text();
}

/* ---- tab bar ----------------------------------------------------------- */

const TABS = ["overview", "memory", "diff", "results"];

function TabBar({ active, onSelect }) {
  return html`
    <div class="pm-tabs">
      ${TABS.map((t) => html`
        <button class=${`pm-tab${active === t ? " active" : ""}`}
                onClick=${() => onSelect(t)}>${t}</button>
      `)}
    </div>
  `;
}

/* ---- memory tab -------------------------------------------------------- */

function MemoryTab({ taskId }) {
  const [doc, setDoc] = useState(null);
  const [step, setStep] = useState("p1_model_analysis");
  const [err, setErr] = useState(null);
  const STEPS = [
    { key: "p1_model_analysis", label: "1. Model" },
    { key: "p2_source_analysis", label: "2. Source FW" },
    { key: "p3_target_analysis", label: "3. Target FW" },
  ];

  const load = useCallback(async (s) => {
    setErr(null); setDoc(null);
    try {
      const txt = await fetchText(`${API(taskId)}/memory/${s}`);
      setDoc(txt || "(not yet generated)");
    } catch (e) { setErr(e.message); }
  }, [taskId]);

  useEffect(() => { load(step); }, [step]);

  return html`
    <div style="display:flex;gap:0.5rem;margin-bottom:1rem;">
      ${STEPS.map((s) => html`
        <button class=${`btn ${step === s.key ? "" : "ghost"}`}
                onClick=${() => setStep(s.key)}>${s.label}</button>
      `)}
    </div>
    ${err && html`<div class="control-error">${err}</div>`}
    <div class="pm-memory" dangerouslySetInnerHTML=${{ __html: marked.parse(doc || "") }}></div>
  `;
}

/* ---- diff tab ---------------------------------------------------------- */

function DiffTab({ taskId }) {
  const [diff, setDiff] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const txt = await fetchText(`${API(taskId)}/diff`);
        setDiff(txt || "(no patch yet)");
      } catch (e) { /* 404 = not generated */ setDiff("(no patch yet)"); }
    })();
  }, [taskId]);

  return html`
    <h3>model_port.patch</h3>
    <pre class="pm-diff">${diff || "loading..."}</pre>
  `;
}

/* ---- results tab ------------------------------------------------------- */

function ResultsTab({ taskId }) {
  const [results, setResults] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetchJSON(`${API(taskId)}/test-results`);
        if (Object.keys(r).length > 0) setResults(r);
      } catch (e) { /* ignore */ }
    })();
  }, [taskId]);

  if (!results || !results.results) return html`<p class="muted">No test results yet.</p>`;

  return html`
    <div class="pm-results">
      <p>
        <span class=${results.passed ? "pill finished" : "pill error"}>
          ${results.passed ? "PASSED" : "FAILED"}
        </span>
        <span class="muted" style="margin-left:0.5rem;">
          ${results.passed_cases ?? 0}/${results.total_cases ?? 0} cases passed
        </span>
      </p>
      ${results.error && html`<div class="control-error">${results.error}</div>`}
      <table>
        <thead><tr><th>Prompt</th><th>Match</th><th>Reason</th></tr></thead>
        <tbody>
          ${(results.results || []).map((r) => html`
            <tr>
              <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.prompt}</td>
              <td class=${r.match ? "pass" : "fail"}>${r.match ? "yes" : "no"}</td>
              <td style="max-width:300px;font-size:0.75rem;">${r.reason || ""}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

/* ---- main component ---------------------------------------------------- */

export default function PortModelDetail({ taskId, run }) {
  const [tab, setTab] = useState("overview");
  const phase = run?.current_phase || "idle";

  return html`
    <${TabBar} active=${tab} onSelect=${setTab} />
    ${tab === "overview" && html`
      <p><span class="pill ${phase}">${phase}</span></p>
      <p class="muted">Select a tab above to inspect analysis, diff, or test results.</p>
    `}
    ${tab === "memory" && html`<${MemoryTab} taskId=${taskId} />`}
    ${tab === "diff" && html`<${DiffTab} taskId=${taskId} />`}
    ${tab === "results" && html`<${ResultsTab} taskId=${taskId} />`}
  `;
}
