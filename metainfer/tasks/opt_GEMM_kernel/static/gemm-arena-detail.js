import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import { ProfilerCharts } from "app/gemm-profiler-charts";

async function getJson(taskId, suffix) {
  const response = await fetch(`/api/opt_GEMM_kernel/${encodeURIComponent(taskId)}${suffix}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function postJson(taskId, suffix, body) {
  const response = await fetch(`/api/opt_GEMM_kernel/${encodeURIComponent(taskId)}${suffix}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* noop */ }
    throw new Error(detail);
  }
  return response.json();
}

function useArena(taskId) {
  const [value, setValue] = useState({
    iterations: [], champion: {}, baseline: {}, charts: null, graph: null,
    guidance: { pending_count: 0, items: [] },
  });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    const [iterations, champion, baseline, charts, graph, guidance] = await Promise.all([
      getJson(taskId, "/iterations"),
      getJson(taskId, "/champion"),
      getJson(taskId, "/baseline"),
      getJson(taskId, "/charts"),
      getJson(taskId, "/state-graph"),
      getJson(taskId, "/guidance"),
    ]);
    setValue({ iterations, champion, baseline, charts, graph, guidance });
  }, [taskId]);
  useEffect(() => { refresh().catch(console.warn); }, [refresh]);
  useEffect(() => {
    const id = setInterval(() => refresh().catch(console.warn), 5000);
    return () => clearInterval(id);
  }, [refresh]);
  return { ...value, refresh };
}

function pct(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(2)}%`;
}

function number(value, digits = 3, suffix = "") {
  return value == null || !Number.isFinite(Number(value))
    ? "—"
    : `${Number(value).toFixed(digits)}${suffix}`;
}

function shape(value) {
  if (!value) return "—";
  return `${value.m}×${value.n}×${value.k}${Number(value.batch || 1) > 1 ? ` × b${value.batch}` : ""}`;
}

function compact(value) {
  if (!value || typeof value !== "object") return value || "—";
  return Object.entries(value).map(([key, item]) => {
    const shown = item && typeof item === "object" ? JSON.stringify(item) : item;
    return `${key}=${shown}`;
  }).join(", ");
}

function contractShapes(value) {
  const shapes = value?.benchmark_shapes || [];
  if (!shapes.length) return "—";
  return shapes.map((item) => {
    const dims = shape(item.shape);
    return `${item.id}: ${dims}`;
  }).join("\n");
}

function GuidancePanel({ taskId, guidance, onSubmitted }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const submit = async () => {
    if (!text.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      await postJson(taskId, "/guidance", { text });
      setText("");
      await onSubmitted();
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
    }
  };
  const items = [...(guidance?.items || [])].reverse().slice(0, 8);
  return html`
    <div class="gemm-guidance-compose">
      <textarea class="input" rows="3" value=${text}
        placeholder="例如：下一轮尝试 128×128 tile；优先检查 K 非对齐 shape；不要使用 fast-math。"
        onInput=${(event) => setText(event.target.value)}></textarea>
      <button class="btn primary" disabled=${busy || !text.trim()} onClick=${submit}>
        ${busy ? "Submitting…" : "Guide next agent"}
      </button>
    </div>
    <p class="gemm-guidance-note">
      ${guidance?.pending_count || 0} pending · 指导会在下一次 Planner 或 Implementer 启动时生效；不会修改 correctness、benchmark 或 champion 门槛。
    </p>
    ${error ? html`<p class="form-err">${error}</p>` : null}
    <div class="gemm-guidance-list">
      ${items.map((item) => html`
        <div class="gemm-guidance-item">
          <span class=${item.status === "pending" ? "pending" : "applied"}>${item.status}</span>
          <p>${item.text}</p>
          <small>${item.status === "applied"
            ? `iter ${item.applied_iteration} · ${item.applied_role}`
            : "waiting for next agent boundary"}</small>
        </div>
      `)}
    </div>
  `;
}

export default function GemmArenaDetail({ taskId, data }) {
  const arena = useArena(taskId);
  const [selectedCase, setSelectedCase] = useState("");
  const nodes = arena.graph?.nodes || [];
  const cases = arena.charts?.profile_cases || [];
  const caseIds = arena.charts?.case_ids || [];
  useEffect(() => {
    if (!caseIds.length) {
      setSelectedCase("");
    } else if (!caseIds.includes(selectedCase)) {
      setSelectedCase(caseIds[0]);
    }
  }, [caseIds.join("\u0000"), selectedCase]);
  const selectedProfile = cases.find((item) => item.id === selectedCase) || {};
  const selectedSeries = arena.charts?.case_series?.[selectedCase]?.series || {};
  const championIteration = arena.charts?.champion_summary?.iteration ?? 0;
  const chartPayload = { series: selectedSeries };
  const selectedIterationCase = (item) => (
    item.score?.cases || []
  ).find((caseItem) => caseItem.id === selectedCase) || {};
  return html`
    <div class="task-grid gemm-arena">
      <section class="panel gemm-wide">
        <h2>Kernel optimization pipeline</h2>
        <div class="gemm-phases">
          ${nodes.map((node) => html`
            <span class=${node.active ? "gemm-phase active" : "gemm-phase"}>${node.label}</span>
          `)}
        </div>
      </section>

      <section class="panel gemm-wide">
        <h2>Optimization guidance</h2>
        <${GuidancePanel}
          taskId=${taskId}
          guidance=${arena.guidance}
          onSubmitted=${arena.refresh} />
      </section>

      <section class="panel gemm-wide">
        <div class="gemm-section-heading">
          <div>
            <h2>Champion profiler</h2>
            <p class="muted">Every shape must beat a same-round current Champion hipprof trace. Values are raw GPU DurationNs operator means; no shape weighting is used.</p>
          </div>
          <label class="gemm-case-picker">
            <span>Case</span>
            <select class="input" value=${selectedCase}
              onChange=${(event) => setSelectedCase(event.target.value)}>
              ${caseIds.map((caseId) => html`<option value=${caseId}>${caseId}</option>`)}
            </select>
          </label>
        </div>
        <div class="gemm-kpis">
          <div class="gemm-kpi"><span>Same-round incumbent</span><strong>${number(selectedProfile.baseline_ms, 4, " ms")}</strong></div>
          <div class="gemm-kpi"><span>Champion latency</span><strong>${number(selectedProfile.candidate_ms, 4, " ms")}</strong></div>
          <div class="gemm-kpi"><span>Stddev</span><strong>${number(selectedProfile.latency_stddev_ms, 5, " ms")}</strong></div>
          <div class="gemm-kpi"><span>CV</span><strong>${pct(selectedProfile.latency_cv)}</strong></div>
          <div class="gemm-kpi"><span>Observed range</span><strong>${number(selectedProfile.latency_min_ms, 4)}–${number(selectedProfile.latency_max_ms, 4, " ms")}</strong></div>
          <div class="gemm-kpi"><span>Samples / batches</span><strong>${number(selectedProfile.sample_count, 0)} / ${number(selectedProfile.measurement_batches || 1, 0)}</strong></div>
          <div class="gemm-kpi"><span>Case speedup</span><strong>${number(selectedProfile.speedup, 4, "×")}</strong></div>
          <div class="gemm-kpi"><span>Regression</span><strong>${pct(selectedProfile.regression)}</strong></div>
          <div class="gemm-kpi"><span>GPU dispatches/call</span><strong>${number(selectedProfile.dispatch_count, 0)}</strong></div>
          <div class="gemm-kpi"><span>Compute</span><strong>${number(selectedProfile.candidate_tflops, 2, " TFLOPS")}</strong></div>
          <div class="gemm-kpi"><span>HBM total</span><strong>${number(selectedProfile.measured_bandwidth_gbps, 2, " GB/s")}</strong></div>
          <div class="gemm-kpi"><span>HBM read</span><strong>${number(selectedProfile.hbm_read_gbps, 2, " GB/s")}</strong></div>
          <div class="gemm-kpi"><span>HBM write</span><strong>${number(selectedProfile.hbm_write_gbps, 2, " GB/s")}</strong></div>
          <div class="gemm-kpi"><span>L2 hit</span><strong>${number(selectedProfile.l2_hit_pct, 2, "%")}</strong></div>
          <div class="gemm-kpi"><span>Occupancy</span><strong>${number(selectedProfile.occupancy_pct, 2, "%")}</strong></div>
          <div class="gemm-kpi"><span>Champion</span><strong>iter ${championIteration}</strong></div>
        </div>
        <p class="muted">Promotion: ${arena.charts?.champion_summary?.reason || "certified Triton baseline"}</p>
        <p class="muted">Kernel contribution per operator call: ${compact(selectedProfile.kernel_breakdown_us)}</p>
      </section>

      <section class="panel gemm-wide">
        <h2>Certified baseline</h2>
        <div class="gemm-baseline-grid">
          <span>Status: ${arena.baseline.certified ? "frozen" : "pending"}</span>
          <span>Stack: ${arena.baseline.kernel_language || arena.baseline.backend || "—"}</span>
          <span>Hardware: ${arena.baseline.detected_hardware || arena.baseline.target_hardware || "—"}</span>
          <span>Arch: ${arena.baseline.gpu_arch || "—"}</span>
          <span>Cases: ${arena.baseline.benchmark?.case_count ?? "—"}</span>
          <span>Timer: ${arena.baseline.benchmark?.methodology?.timer || "—"}</span>
          <span>Samples: ${arena.baseline.benchmark?.methodology?.samples ?? "—"}</span>
          <span>Build: ${(arena.baseline.build_fingerprint || "—").slice(0, 16)}</span>
          <span>Profiler: ${arena.baseline.profiler?.tool || "—"}</span>
          <span>Profile: ${arena.baseline.profiler?.profile_id || "—"}</span>
          <span>Profiler fingerprint: ${(arena.baseline.profiler?.fingerprint || "—").slice(0, 16)}</span>
        </div>
      </section>

      <section class="panel gemm-wide">
        <h2>Performance by iteration · ${selectedCase || "no case"}</h2>
        <${ProfilerCharts} payload=${chartPayload} />
      </section>

      <section class="panel gemm-wide">
        <h2>Champion workload profile</h2>
        <div class="gemm-table-wrap"><table>
          <thead><tr><th>Case</th><th>M×N×K</th><th>Same-round incumbent</th><th>Champion mean</th><th>Stddev/CV</th><th>Speedup</th><th>Dispatches</th><th>HBM R/W/total</th><th>L2 hit</th><th>VGPR/AGPR/SGPR</th><th>LDS/scratch</th></tr></thead>
          <tbody>${cases.map((item) => html`<tr>
            <td>${item.id}</td>
            <td>${shape(item.shape)}</td>
            <td>${number(item.baseline_ms, 4, " ms")}</td>
            <td>${number(item.candidate_ms, 4, " ms")}</td>
            <td>${number(item.latency_stddev_ms, 5, " ms")} / ${pct(item.latency_cv)}</td>
            <td>${number(item.speedup, 3, "×")}</td>
            <td>${number(item.dispatch_count, 0)}</td>
            <td>${number(item.hbm_read_gbps, 1)} / ${number(item.hbm_write_gbps, 1)} / ${number(item.measured_bandwidth_gbps, 1)} GB/s</td>
            <td>${number(item.l2_hit_pct, 2, "%")}</td>
            <td>${number(item.vgpr_count, 0)} / ${number(item.agpr_count, 0)} / ${number(item.sgpr_count, 0)}</td>
            <td>${number(item.lds_bytes, 0)} / ${number(item.scratch_bytes, 0)} B</td>
          </tr>`)}</tbody>
        </table></div>
        ${cases.length ? null : html`<p class="muted">Case metrics appear after baseline certification. TFLOPS requires shape/flops metadata; bandwidth requires bytes metadata in evaluator task.yaml.</p>`}
      </section>

      <section class="panel gemm-wide">
        <h2>Iterations · ${selectedCase || "no case"}</h2>
        <div class="gemm-table-wrap"><table>
          <thead><tr><th>#</th><th>Status</th><th>Outcome</th><th>Case latency</th><th>Case speedup</th><th>Case regression</th><th>All-shape gates</th><th>Promoted</th></tr></thead>
          <tbody>${arena.iterations.map((item) => {
            const caseItem = selectedIterationCase(item);
            const failed = item.score?.failed_case_ids || [];
            return html`<tr>
              <td>${item.iteration}</td>
              <td>${item.status}</td>
              <td>${item.outcome || "—"}</td>
              <td>${number(caseItem.candidate_ms, 4, " ms")}</td>
              <td>${number(caseItem.speedup, 4, "×")}</td>
              <td>${pct(caseItem.regression)}</td>
              <td>${item.score?.passed ? "passed" : (failed.length ? `failed: ${failed.join(", ")}` : "not run")}</td>
              <td>${item.promoted ? "yes" : "no"}</td>
            </tr>`;
          })}</tbody>
        </table></div>
      </section>

      <section class="panel"><h2>Live sub-agents</h2><${AgentsPanel} agents=${data.agents} /></section>
      <section class="panel timeline-panel"><h2>Event timeline</h2><${Timeline} events=${data.timeline.events} /></section>

      <section class="panel gemm-wide">
        <h2>Task contract <span class="gemm-readonly">read-only · frozen evaluator</span></h2>
        <div class="gemm-contract-grid">
          <div><span>Kernel path</span><code>${arena.baseline.task?.kernel_path || "—"}</code></div>
          <div><span>Data types</span><strong>${compact(arena.baseline.task?.public_contract?.dtype)}</strong></div>
          <div><span>Max iterations</span><strong>${arena.baseline.task?.max_iterations ?? "—"}</strong></div>
          <div class="gemm-contract-wide"><span>Operation</span><pre>${arena.baseline.task?.public_contract?.operation || "GEMM"}</pre></div>
          <div class="gemm-contract-wide"><span>Public shapes</span><pre>${contractShapes(arena.baseline.task?.public_contract)}</pre></div>
          <div class="gemm-contract-wide"><span>Layout</span><pre>${compact(arena.baseline.task?.public_contract?.layout)}</pre></div>
          <div class="gemm-contract-wide"><span>ABI</span><pre>${compact(arena.baseline.task?.public_contract?.abi)}</pre></div>
        </div>
      </section>
    </div>
  `;
}
