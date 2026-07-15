// gen-infer-framework-cpp task detail body.
//
// Rendered when detail_view_module === "app/cpp-gf-detail".
// Fetches its own data from /api/gen-infer-framework-cpp/<id>/* and composes
// the panels using C++ task-owned widgets plus the shell-shared agents and
// timeline panels.
//
// Shell passes {run, timeline, agents, loadState, lastErr} via `data`.

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { StateGraph } from "app/cpp-gf-state-graph";
import { IterationsTable } from "app/cpp-gf-iterations-table";
import { Charts } from "app/cpp-gf-charts";
import { RetrospectiveModal } from "app/cpp-gf-retrospective-modal";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import {
  getIterations, getCharts, getStateGraph,
} from "app/cpp-gf-runtime-api";

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

function useRuntimeData(taskId) {
  const [data, setData] = useState({
    iterations: [], charts: null, graph: null,
  });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    const [it, ch, g] = await Promise.all([
      withTimeout(getIterations(taskId)).catch((e) => { console.warn("iterations:", e); return []; }),
      withTimeout(getCharts(taskId)).catch((e) => { console.warn("charts:", e); return null; }),
      withTimeout(getStateGraph(taskId)).catch((e) => { console.warn("state-graph:", e); return null; }),
    ]);
    setData({ iterations: it || [], charts: ch, graph: g });
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);
  return { ...data, refresh };
}

export default function GenInferDetailView({
  taskId,
  run,
  status,
  data,
}) {
  const [selectedIter, setSelectedIter] = useState(null);
  const { timeline, agents, loadState, lastErr } = data;
  const rt = useRuntimeData(taskId);

  if (loadState === "error" && lastErr) {
    return html`
      <div class="task-banner task-banner-err">
        <strong>刷新失败：</strong> ${lastErr}
        <span class="muted">（轮询会自动重试）</span>
      </div>
    `;
  }

  return html`
    <div class="task-grid">
      <section class="panel">
        <h2>State machine</h2>
        <${StateGraph} graph=${rt.graph} />
      </section>

      <section class="panel">
        <h2>Iterations <span class="muted">(click for retrospective)</span></h2>
        <${IterationsTable}
          iterations=${rt.iterations}
          selectedN=${selectedIter}
          onSelect=${(n) => setSelectedIter(n)} />
      </section>

      <section class="panel">
        <h2>Live sub-agents</h2>
        <${AgentsPanel} agents=${agents} />
      </section>

      <section class="panel">
        <h2>Performance &amp; duration</h2>
        <${Charts} payload=${rt.charts} />
      </section>

      <section class="panel timeline-panel">
        <h2>Event timeline</h2>
        <${Timeline} events=${timeline.events} />
      </section>
    </div>

    ${selectedIter != null ? html`
      <${RetrospectiveModal}
        taskId=${taskId}
        iteration=${selectedIter}
        onClose=${() => setSelectedIter(null)} />
    ` : null}
  `;
}
