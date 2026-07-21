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
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { StateGraph } from "app/ke-state-graph";
import { IterationsTable } from "app/ke-iterations-table";
import { KnowledgeGainedPanel } from "app/ke-knowledge-gained";
import { OracleReportModal } from "app/ke-oracle-report-modal";
import { KnowledgeDiffModal } from "app/ke-knowledge-diff";
import { RetrospectiveModal } from "app/ke-retrospective-modal";
import { IterationDetail } from "app/ke-iteration-detail";
import { LogViewer } from "app/ke-log-viewer";
import { Charts } from "app/ke-charts";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import {
  getIterations, getStateGraph, getKnowledgeGained, getModelCode, getCharts, getAgentStatus,
} from "app/ke-runtime-api";

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

function useRuntimeData(taskId, phase) {
  const [iterations, setIterations] = useState([]);
  const [graph, setGraph] = useState(null);
  const [knowledgeGained, setKnowledgeGained] = useState(null);
  const [modelCode, setModelCode] = useState(null);
  const [chartsData, setChartsData] = useState(null);
  const [loadErr, setLoadErr] = useState(null);
  const [panelErrs, setPanelErrs] = useState({});
  const backoffRef = useRef(0);

  const refresh = useCallback(async () => {
    if (!taskId) return;
    try {
      const results = await Promise.allSettled([
        withTimeout(getIterations(taskId)),
        withTimeout(getStateGraph(taskId)),
        withTimeout(getKnowledgeGained(taskId)),
        withTimeout(getModelCode(taskId)),
        withTimeout(getCharts(taskId)),
      ]);
      const keys = ["iterations", "state-graph", "knowledge-gained", "model-code", "charts"];
      const newErrs = {};
      const [it, g, kg, mc, ch] = results.map((r, i) => {
        if (r.status === "fulfilled") return r.value;
        newErrs[keys[i]] = r.reason?.message || "fetch failed";
        return null;
      });
      setIterations(it || []);
      setGraph(g);
      setKnowledgeGained(kg);
      setModelCode(mc);
      setChartsData(ch);
      setPanelErrs(newErrs);
      setLoadErr(null);
      backoffRef.current = 0;
    } catch (e) {
      setLoadErr(e.message);
    }
  }, [taskId]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!taskId) return;
    // Stop polling when task is finished
    if (phase === "finished") return;

    const isIdle = phase === "idle";
    const baseInterval = isIdle ? 15000 : 5000;

    const schedule = () => {
      const delay = backoffRef.current > 0
        ? Math.min(baseInterval * Math.pow(2, backoffRef.current), 60000)
        : baseInterval;
      return setTimeout(() => {
        refresh().then(() => {
          // On success backoff is reset inside refresh
        }).catch(() => {
          backoffRef.current = Math.min(backoffRef.current + 1, 5);
        });
        timerRef.current = schedule();
      }, delay);
    };
    const timerRef = { current: schedule() };

    return () => clearTimeout(timerRef.current);
  }, [taskId, phase, refresh]);

  return { iterations, graph, knowledgeGained, modelCode, chartsData, loadErr, panelErrs, refresh };
}

export default function KEDetailView({ taskId, run, status, data }) {
  const { timeline, agents, loadState, lastErr } = data;
  const phase = run?.current_phase || "idle";
  const {
    iterations, graph, knowledgeGained, modelCode, chartsData, loadErr, panelErrs, refresh,
  } = useRuntimeData(taskId, phase);
  const [selectedIter, setSelectedIter] = useState(null);
  const [oracleIter, setOracleIter] = useState(null);
  const [detailIter, setDetailIter] = useState(null);
  const [retroIter, setRetroIter] = useState(null);
  const [diffTarget, setDiffTarget] = useState(null);
  const [showLog, setShowLog] = useState(false);

  const isKE = true;

  // Fetch agent_status — stop polling when finished
  const [agentStatus, setAgentStatus] = useState("");
  useEffect(() => {
    if (!taskId) return;
    const poll = () => {
      getAgentStatus(taskId).then((s) => setAgentStatus(s || "")).catch(() => {});
    };
    poll();
    if (phase === "finished") return;
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, [taskId, phase]);

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

  // Per-panel inline error badge with retry
  const panelErr = (key) => {
    const msg = panelErrs[key];
    if (!msg) return null;
    return html`<span class="pill err ke-panel-err" title=${msg} onClick=${refresh}>⚠ ${msg.slice(0, 30)}${msg.length > 30 ? "…" : ""} (retry)</span>`;
  };

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
        <h2>State machine ${panelErr("state-graph")}</h2>
        <${StateGraph} graph=${graph} />
      </section>

      <section class="panel">
        <h2>Iterations <span class="muted">(click row for detail)</span> ${panelErr("iterations")}</h2>
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
        <h2>Oracle correctness ${panelErr("charts")}
          ${iterations.length > 0 ? html`
            <button class="btn ghost" style="margin-left:auto;font-size:10px"
              onClick=${() => {
                const latest = iterations[iterations.length - 1];
                setOracleIter(latest.n || latest.iteration);
              }}>
              View latest report
            </button>` : null}
        </h2>
        <${Charts} payload=${chartsData} iterations=${iterations} />
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
                  // Also show running iterations even without oracle data yet
                  const isRunning = it.status === "running";
                  if (!oraclePhase && !isRunning) return null;

                  if (oraclePhase) {
                    const { name, data } = oraclePhase;
                    const shortName = name === "A_attempt_pure" ? "A" : name === "B_enrich" ? "B" : "D";
                    const ok = data.outcome === "OK";
                    const perf = data.perf;
                    return html`
                      <tr key=${n} class=${ok ? "oracle-pass" : "oracle-fail"}>
                        <td>
                          <a href="#" class="ke-link"
                            onClick=${(e) => { e.preventDefault(); setDetailIter(detailIter === n ? null : n); }}>
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
                  }

                  // Running iteration with no oracle data yet — show status
                  const subRoles = ["planner", "implementer", "c_debugger", "reviewer"]
                    .filter((r) => phases[r]);
                  const activeRole = subRoles.length > 0 ? subRoles[subRoles.length - 1] : null;
                  return html`
                    <tr key=${n} class="oracle-running">
                      <td>
                        <a href="#" class="ke-link"
                          onClick=${(e) => { e.preventDefault(); setDetailIter(detailIter === n ? null : n); }}>
                          ${n}
                        </a>
                      </td>
                      <td><span class="pill warn">running</span></td>
                      <td><span class="pill warn">${activeRole || "…"}</span></td>
                      <td class="muted" style="font-size:11px">&mdash;</td>
                    </tr>
                  `;
                })}
              </tbody>
            </table>
          `
          : html`<p class="muted">No oracle data yet. The correctness oracle runs during A/B/D phases to validate generated code against a reference implementation.</p>`
        }
        ${detailIter != null ? html`
          <${IterationDetail} taskId=${taskId} iteration=${detailIter} />
        ` : null}
      </section>

      <section class="panel knowledge-panel">
        <h2>
          Knowledge gained ${panelErr("knowledge-gained")}
          ${knowledgeGained?.total_files != null
            ? html`<span class="muted">(${knowledgeGained.total_files} files)</span>`
            : null}
        </h2>
        <${KnowledgeGainedPanel}
          payload=${knowledgeGained}
          modelCode=${modelCode}
          taskId=${taskId}
          onViewDiff=${(iter, file) => setDiffTarget({ iteration: iter, file })} />
      </section>

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
