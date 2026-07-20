// find-low-hanging-kernel task detail body.
//
// Tabs:
//   - overview  : phase state graph + iterations table + phase description
//   - memory    : step1_code_analysis / step2_tracing_analysis markdown
//   - graph     : iframe pointing at /visualization (self-contained ELK + SVG)
//   - trace     : summary of trace_parsed.json
//   - runtime   : agents panel + timeline (shell-shared)
//
// Shell passes {run, timeline, agents, loadState, lastErr} via the `data` prop.

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";

const PLUGIN_TYPE = "find-low-hanging-kernel";

const TABS = [
  { id: "overview", label: "概览" },
  { id: "memory",   label: "记忆文件" },
  { id: "graph",    label: "流程图" },
  { id: "trace",    label: "Trace 解析" },
  { id: "runtime",  label: "运行状态" },
];

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

async function getJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function useRuntimeData(taskId) {
  const [data, setData] = useState({
    iterations: [], stateGraph: null, flowGraph: null, trace: null,
  });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    const base = `/api/${PLUGIN_TYPE}/${encodeURIComponent(taskId)}`;
    const [it, sg, fg, tp] = await Promise.all([
      withTimeout(getJson(`${base}/iterations`).catch((e) => { console.warn("iterations:", e); return []; })),
      withTimeout(getJson(`${base}/state-graph`).catch(() => null)),
      withTimeout(getJson(`${base}/flow-graph`).catch(() => null)),
      withTimeout(getJson(`${base}/trace-parsed`).catch(() => null)),
    ]);
    setData({
      iterations: it || [],
      stateGraph: sg,
      flowGraph: fg,
      trace: tp,
    });
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);
  return { ...data, refresh };
}

function useMemoryFile(taskId, step) {
  const [md, setMd] = useState({ ready: false, markdown: "" });
  useEffect(() => {
    if (!taskId || !step) return;
    const base = `/api/${PLUGIN_TYPE}/${encodeURIComponent(taskId)}`;
    getJson(`${base}/memory/${step}`)
      .then(setMd)
      .catch(() => setMd({ ready: false, markdown: "" }));
  }, [taskId, step]);
  return md;
}

function PhaseGraph({ graph }) {
  if (!graph) return html`<div class="muted">等待 state-graph 数据…</div>`;
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const active = graph.active_edge || null;
  const current = graph.current || "idle";
  // Render as a simple list with edges (the heavy graph view is on the
  // "流程图" tab via ELK). Keep this tab lightweight.
  return html`
    <div class="phase-graph">
      <div class="phase-nodes">
        ${nodes.map((n) => html`
          <span class=${"phase-pill phase-" + n.id + (n.id === current ? " active" : "")}>
            ${n.label}
          </span>
        `)}
      </div>
      <ul class="phase-edges">
        ${edges.map((e) => html`
          <li class=${(active && active.from === e.from && active.to === e.to) ? "active" : ""}>
            <code>${e.from}</code> → <code>${e.to}</code> <span class="muted">${e.label}</span>
          </li>
        `)}
      </ul>
    </div>
  `;
}

function IterationsTable({ iterations }) {
  if (!iterations || !iterations.length) {
    return html`<div class="muted">尚无校验轮次记录。</div>`;
  }
  return html`
    <table class="iter-table">
      <thead>
        <tr><th>Round</th><th>Status</th><th>Outcome</th><th>Integrity fixes</th><th>Issues</th></tr>
      </thead>
      <tbody>
        ${iterations.map((it) => html`
          <tr key=${it.iteration}>
            <td>${it.round || it.iteration}</td>
            <td>${it.status}</td>
            <td>${it.outcome || "—"}</td>
            <td>${(it.integrity_fixes || []).length}</td>
            <td>${(it.semantic_issues || []).length}</td>
          </tr>
        `)}
      </tbody>
    </table>
  `;
}

function MemoryPanel({ taskId }) {
  const [step, setStep] = useState("step1_code_analysis");
  const md = useMemoryFile(taskId, step);
  return html`
    <div class="memory-panel">
      <div class="memory-tabs">
        ${["step1_code_analysis", "step2_tracing_analysis", "validation_warnings"].map((s) => html`
          <button class=${"memory-tab" + (s === step ? " active" : "")} onClick=${() => setStep(s)}>
            ${s}
          </button>
        `)}
      </div>
      <pre class="memory-md">${md.ready ? md.markdown : "(尚未生成)"}</pre>
    </div>
  `;
}

function GraphPanel({ taskId, flowGraph }) {
  if (!taskId) return null;
  const src = `/api/${PLUGIN_TYPE}/${encodeURIComponent(taskId)}/visualization`;
  const ready = flowGraph && flowGraph.ready;
  return html`
    <div class="graph-panel">
      <div class="graph-toolbar">
        <a href=${src} target="_blank" rel="noopener">在新窗口打开</a>
        <span class="muted">${ready ? "图已生成" : "等待 Step 4 完成…"}</span>
      </div>
      ${ready
        ? html`<iframe class="graph-iframe" src=${src}></iframe>`
        : html`<div class="muted">visualization 尚未就绪。</div>`}
    </div>
  `;
}

function TracePanel({ trace }) {
  if (!trace || !trace.ready) {
    return html`<div class="muted">等待 trace_parsed.json …</div>`;
  }
  const s = trace.summary || {};
  const rows = (s.by_name_cat || []).slice(0, 30);
  return html`
    <div class="trace-panel">
      <div class="trace-meta">
        <span>file: <code>${s.source || "?"}</code></span>
        <span>events: ${s.event_count || 0}</span>
        <span>CPU stack: ${s.has_cpu_stack ? "yes" : "no"}</span>
        <span>CUDA Graph: ${s.cuda_graph_detected ? "detected" : "n/a"}</span>
      </div>
      <table class="iter-table">
        <thead>
          <tr>
            <th>Kernel</th><th>Cat</th><th>Count</th>
            <th>μ (us)</th><th>σ (us)</th><th>total (us)</th>
            <th>p99 (us)</th><th>bimodal?</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((r, i) => html`
            <tr key=${i}>
              <td><code>${r.name}</code></td>
              <td>${r.cat}</td>
              <td>${r.count}</td>
              <td>${r.mean_us}</td>
              <td>${r.std_us}</td>
              <td>${r.total_us}</td>
              <td>${r.p99_us}</td>
              <td>${r.bimodal_suspect ? "⚠" : ""}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

export default function FlhkDetailView({ taskId, run, status, data }) {
  const [activeTab, setActiveTab] = useState("overview");
  const phase = run?.current_phase || "idle";
  const { timeline, agents } = data;
  const rt = useRuntimeData(taskId);

  return html`
    <div class="flhk-detail">
      <nav class="task-tabs">
        ${TABS.map((t) => html`
          <button
            class=${"task-tab" + (t.id === activeTab ? " active" : "")}
            onClick=${() => setActiveTab(t.id)}
          >${t.label}</button>
        `)}
      </nav>

      <section class="task-tab-body">
        ${activeTab === "overview" && html`
          <div class="panel-grid">
            <div class="panel">
              <h3>当前阶段</h3>
              <p><strong>${phase}</strong></p>
              <${PhaseGraph} graph=${rt.stateGraph} />
            </div>
            <div class="panel">
              <h3>校验轮次</h3>
              <${IterationsTable} iterations=${rt.iterations} />
            </div>
          </div>
        `}

        ${activeTab === "memory" && html`<${MemoryPanel} taskId=${taskId} />`}

        ${activeTab === "graph" && html`<${GraphPanel} taskId=${taskId} flowGraph=${rt.flowGraph} />`}

        ${activeTab === "trace" && html`<${TracePanel} trace=${rt.trace} />`}

        ${activeTab === "runtime" && html`
          <div class="panel-grid">
            <div class="panel">
              <h3>Agents</h3>
              <${AgentsPanel} agents=${agents} />
            </div>
            <div class="panel">
              <h3>Timeline</h3>
              <${Timeline} events=${timeline} />
            </div>
          </div>
        `}
      </section>
    </div>
  `;
}
