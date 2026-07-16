// Knowledge-evolution task detail body.
//
// Rendered by the task-detail shell when detail_view_module === "app/ke-detail".
// Shell passes { taskId, run, status, data } where data = { timeline, agents, loadState, lastErr }.
//
// Knowledge-evolution-specific panels:
//   - State machine (SVG, 4-phase A→B→C→D)
//   - Iterations table
//   - Live sub-agents (shared)
//   - Oracle correctness (charts)
//   - Knowledge gained (consolidated notebooks)
//   - Event timeline (shared)
//   - Modals: oracle report, knowledge diff, log viewer

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { StateGraph } from "app/ke-state-graph";
import { IterationsTable } from "app/ke-iterations-table";
import { KnowledgeGainedPanel } from "app/ke-knowledge-gained";
import { OracleReportModal } from "app/ke-oracle-report-modal";
import { KnowledgeDiffModal } from "app/ke-knowledge-diff";
import { RetrospectiveModal } from "app/ke-retrospective-modal";
import { LogViewer } from "app/ke-log-viewer";
import { Charts } from "app/ke-charts";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import {
  getIterations, getStateGraph, getKnowledgeGained, getCharts, getAgentStatus,
} from "app/ke-runtime-api";

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

function useRuntimeData(taskId) {
  const [iterations, setIterations] = useState([]);
  const [graph, setGraph] = useState(null);
  const [knowledgeGained, setKnowledgeGained] = useState(null);
  const [chartsData, setChartsData] = useState(null);
  const [loadErr, setLoadErr] = useState(null);

  const refresh = useCallback(async () => {
    if (!taskId) return;
    try {
      const [it, g, kg, ch] = await Promise.all([
        withTimeout(getIterations(taskId)).catch((e) => { console.warn("iterations:", e); return []; }),
        withTimeout(getStateGraph(taskId)).catch((e) => { console.warn("state-graph:", e); return null; }),
        withTimeout(getKnowledgeGained(taskId)).catch((e) => { console.warn("knowledge-gained:", e); return null; }),
        withTimeout(getCharts(taskId)).catch((e) => { console.warn("charts:", e); return null; }),
      ]);
      setIterations(it || []);
      setGraph(g);
      setKnowledgeGained(kg);
      setChartsData(ch);
      setLoadErr(null);
    } catch (e) {
      setLoadErr(e.message);
    }
  }, [taskId]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);

  return { iterations, graph, knowledgeGained, chartsData, loadErr, refresh };
}

export default function KEDetailView({ taskId, run, status, data }) {
  const { timeline, agents, loadState, lastErr } = data;
  const {
    iterations, graph, knowledgeGained, chartsData, loadErr,
  } = useRuntimeData(taskId);
  const [selectedIter, setSelectedIter] = useState(null);
  const [oracleIter, setOracleIter] = useState(null);
  const [retroIter, setRetroIter] = useState(null);
  const [diffTarget, setDiffTarget] = useState(null);
  const [showLog, setShowLog] = useState(false);

  const isKE = (run?.task_type || "") === "knowledge-evolution";
  const phase = run?.current_phase || "idle";

  // Fetch agent_status separately (KE-private file, avoids shared state.py change)
  const [agentStatus, setAgentStatus] = useState("");
  useEffect(() => {
    if (!taskId) return;
    const poll = () => {
      getAgentStatus(taskId).then((s) => setAgentStatus(s || "")).catch(() => {});
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [taskId]);

  // Mode indicator: Pure KB (A/D) vs OS enabled (B)
  const modeLabel = (() => {
    if (phase === "idle" || phase === "finished") return null;
    if (phase === "B_enrich") return { label: "OS enabled", cls: "warn" };
    return { label: "Pure KB", cls: "success" };
  })();

  if ((loadState === "error" && lastErr) || loadErr) {
    return html`
      <div class="task-banner task-banner-err">
        <strong>Refresh failed:</strong> ${loadErr || lastErr}
        <span class="muted">(polling will retry automatically)</span>
      </div>
    `;
  }

  return html`
    ${isKE && (modeLabel || agentStatus) ? html`
      <div class="ke-mode-bar">
        ${modeLabel ? html`
          <span class="ke-mode-label">Mode:</span>
          <span class="pill ${modeLabel.cls}">${modeLabel.label}</span>
          <span class="muted"> — A/D = pure notebooks, B = open-source allowed</span>
        ` : null}
        ${agentStatus ? html`
          <span style="margin-left:auto" class="pill ${agentStatus.includes("fail") ? "err" : agentStatus.includes("running") ? "warn" : "success"}">${agentStatus}</span>
        ` : null}
      </div>` : null}

    <div class="task-grid">

      <section class="panel">
        <h2>State machine</h2>
        <${StateGraph} graph=${graph} />
      </section>

      <section class="panel">
        <h2>Iterations <span class="muted">(click row for detail)</span></h2>
        <${IterationsTable}
          iterations=${iterations}
          selectedN=${selectedIter}
          onSelect=${(n) => { setSelectedIter(n); setRetroIter(n); }} />
      </section>

      <section class="panel">
        <h2>Live sub-agents</h2>
        <${AgentsPanel} agents=${agents} />
      </section>

      <section class="panel">
        <h2>Oracle correctness
          ${iterations.length > 0 ? html`
            <button class="btn ghost" style="margin-left:auto;font-size:10px"
              onClick=${() => {
                const latest = iterations[iterations.length - 1];
                setOracleIter(latest.n || latest.iteration);
              }}>
              View latest report
            </button>` : null}
        </h2>
        <${Charts} payload=${chartsData} />
        ${iterations.length > 0
          ? html`
            <table class="ke-oracle-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Phase</th>
                  <th>Oracle</th>
                  <th>Perf</th>
                </tr>
              </thead>
              <tbody>
                ${iterations.map((it) => {
                  const n = it.n || it.iteration;
                  const phases = it.phases || {};
                  const oraclePhase = ["A_attempt_pure", "B_enrich", "D_verify_final"]
                    .map((p) => ({ name: p, data: phases[p] }))
                    .find((p) => p.data);
                  if (!oraclePhase) return null;
                  const { name, data } = oraclePhase;
                  const shortName = name === "A_attempt_pure" ? "A" : name === "B_enrich" ? "B" : "D";
                  const ok = data.outcome === "OK";
                  const perf = data.perf;
                  return html`
                    <tr key=${n} class=${ok ? "oracle-pass" : "oracle-fail"}>
                      <td>
                        <a href="#" class="ke-link"
                          onClick=${(e) => { e.preventDefault(); setOracleIter(n); setRetroIter(n); }}>
                          ${n}
                        </a>
                      </td>
                      <td><span class="pill ${name === "B_enrich" ? "warn" : "success"}">${shortName}</span></td>
                      <td><span class="pill ${ok ? "success" : "err"}">${ok ? "pass" : "fail"}</span></td>
                      <td class="muted" style="font-size:11px">
                        ${perf
                          ? Object.entries(perf).map(([k, v]) => html`<span title=${k}>${k}: ${typeof v === "number" ? v.toFixed(2) : "" + v}</span><br/>`)
                          : html`&mdash;`}
                      </td>
                    </tr>
                  `;
                })}
              </tbody>
            </table>
          `
          : html`<p class="muted">No oracle data yet. The correctness oracle runs during A/B/D phases to validate generated code against a reference implementation.</p>`
        }
      </section>

      ${isKE ? html`
        <section class="panel knowledge-panel">
          <h2>
            Knowledge gained
            ${knowledgeGained?.total_files != null
              ? html`<span class="muted">(${knowledgeGained.total_files} files)</span>`
              : null}
          </h2>
          <${KnowledgeGainedPanel}
            payload=${knowledgeGained}
            taskId=${taskId}
            onViewDiff=${(iter, file) => setDiffTarget({ iteration: iter, file })} />
        </section>
      ` : null}

      <section class="panel timeline-panel">
        <h2>Event timeline</h2>
        <${Timeline} events=${timeline.events} />
      </section>
    </div>

    ${showLog ? html`
      <${LogViewer} taskId=${taskId} visible=${showLog} />
    ` : null}
    <div style="text-align:center;padding:8px">
      <button class="btn ghost"
        onClick=${() => setShowLog(!showLog)}>
        ${showLog ? "Hide orchestrator log" : "Show orchestrator log"}
      </button>
    </div>

    ${oracleIter != null ? html`
      <${OracleReportModal}
        taskId=${taskId}
        iteration=${oracleIter}
        onClose=${() => setOracleIter(null)} />
    ` : null}

    ${retroIter != null ? html`
      <${RetrospectiveModal}
        taskId=${taskId}
        iteration=${retroIter}
        onClose=${() => setRetroIter(null)} />
    ` : null}

    ${diffTarget != null ? html`
      <${KnowledgeDiffModal}
        taskId=${taskId}
        iteration=${diffTarget.iteration}
        file=${diffTarget.file}
        onClose=${() => setDiffTarget(null)} />
    ` : null}
  `;
}
