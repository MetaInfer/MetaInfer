import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `${url} returned HTTP ${response.status}`);
  }
  return payload;
}

function StateMachine({ graph }) {
  const nodes = graph?.nodes || [];
  if (!nodes.length) return html`<p class="muted">Waiting for the control plane.</p>`;
  return html`
    <div class="dkao-state-machine">
      ${nodes.map((node, index) => html`
        <div class="dkao-state-step ${graph.current === node.id ? "active" : ""} ${node.is_terminal ? "terminal" : ""}">
          <span class="dkao-state-index">${index + 1}</span>
          <span>${node.label}</span>
        </div>
        ${index < nodes.length - 1 ? html`<span class="dkao-state-arrow">→</span>` : null}
      `)}
    </div>
  `;
}

function metric(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function statusClass(status) {
  if (status === "passed" || status === "completed") return "success";
  if (status?.startsWith("bootstrap_")) {
    return ["bootstrap_failed", "bootstrap_orphaned"].includes(status)
      ? "failed"
      : status === "bootstrap_passed" ? "success" : "running";
  }
  if (
    status === "running"
    || status === "validating"
    || status === "retrying"
    || status === "agent_running"
    || status === "building"
    || status === "baseline"
    || status === "validating_candidate"
    || status === "recording_result"
    || status === "optimization_complete"
    || status === "skill_writing"
  ) {
    return "running";
  }
  if (
    status === "failed"
    || status === "orphaned"
    || status === "timed_out"
    || status === "skipped"
  ) return "failed";
  return "idle";
}

function bestRoundPerShape(lanes) {
  const best = new Map();
  for (const lane of lanes || []) {
    for (const experiment of lane.experiments || []) {
      if (experiment.correctness_passed === false) continue;
      const median = Number(experiment.metrics?.median_us);
      if (!Number.isFinite(median) || !experiment.shape_id) continue;
      const accepted = experiment.accepted === true;
      let current = best.get(experiment.shape_id);
      if (!current) {
        current = { accepted: false, iteration: 0, median_us: Infinity };
      }
      // Prefer accepted ("best") rounds; among those, the lowest median.
      // Before the first acceptance, fall back to the best tested round.
      const preferred = accepted && !current.accepted;
      const better = accepted === current.accepted && median < current.median_us;
      const laterWinner = accepted === current.accepted
        && median === current.median_us
        && Number(experiment.iteration) > current.iteration;
      if (preferred || better || laterWinner) {
        best.set(experiment.shape_id, {
          accepted,
          worker_id: lane.worker_id,
          iteration: experiment.iteration,
          median_us: median,
          metrics: experiment.metrics || {},
          speedup: experiment.speedup,
          baseline_us: experiment.baseline_us,
        });
      }
    }
  }
  return best;
}

function ExploreGpuGrid({ lanes }) {
  return html`
    <div class="dkao-explore-summary">
      <div class="dkao-explore-heading">
        <strong>Parallel explore · live GPU lanes</strong>
        <span>Agent → compile → correctness → benchmark → record → skill</span>
      </div>
      <div class="dkao-explore-grid">
        ${lanes.map((lane) => {
          const completed = Number(lane.completed_rounds || 0);
          const target = Number(lane.target_rounds || 0);
          const progress = target
            ? Math.min(100, Math.round(completed / target * 100))
            : 0;
          const lastOutput = lane.agent?.last_output_age_s;
          return html`
            <article class="dkao-explore-gpu ${lane.long_running ? "stale" : ""}">
              <header>
                <div>
                  <strong>GPU ${lane.gpu}</strong>
                  <small>${lane.worker_id}</small>
                </div>
                <span class="pill ${statusClass(lane.state)}">${lane.state}</span>
              </header>
              <p class="dkao-explore-step">${lane.step}</p>
              <dl>
                <dt>Shape</dt>
                <dd>${lane.current_shape || lane.assigned_shapes?.join(", ") || "—"}</dd>
                <dt>Round</dt>
                <dd>${completed} / ${target || "—"}</dd>
                <dt>Agent output</dt>
                <dd>${Number.isFinite(Number(lastOutput))
                  ? `${metric(lastOutput, 0)} s ago`
                  : "—"}</dd>
              </dl>
              <div class="dkao-explore-progress">
                <span style=${`width:${progress}%`}></span>
              </div>
              ${lane.long_running ? html`
                <small class="dkao-explore-warning">
                  No Agent output for over 3 minutes; the lane timeout guard is active.
                </small>
              ` : null}
              ${lane.agent?.error ? html`
                <small class="dkao-explore-warning">${lane.agent.error}</small>
              ` : null}
            </article>
          `;
        })}
      </div>
    </div>
  `;
}

function BootstrapResultCard({ attempt }) {
  const files = attempt.generated_files || [];
  const shapeEntries = Object.entries(attempt.metrics || {});
  const shapeMetrics = shapeEntries.map(([, item]) => item);
  const medians = shapeMetrics
    .map((item) => Number(item?.median_us))
    .filter((item) => Number.isFinite(item));
  const medianRange = medians.length
    ? `${Math.min(...medians).toFixed(3)}–${Math.max(...medians).toFixed(3)} µs`
    : "—";
  return html`
    <div class="dkao-iteration-card ${attempt.status === "passed" ? "accepted" : ""}">
      <div class="dkao-iteration-title">
        <strong>Bootstrap ${attempt.attempt}</strong>
        <span class="pill ${statusClass(attempt.status)}">${attempt.status}</span>
      </div>
      <dl>
        <dt>Elapsed</dt><dd>${metric(attempt.elapsed_s, 1)} s</dd>
        <dt>Last output</dt><dd>${metric(attempt.last_output_age_s, 1)} s ago</dd>
        <dt>Files</dt><dd>${files.length}</dd>
        <dt>Verified</dt><dd>${shapeMetrics.length} shapes</dd>
        <dt>Median</dt><dd>${medianRange}</dd>
        <dt>Code snapshot</dt>
        <dd>${attempt.artifact_dir || "—"}</dd>
      </dl>
      ${shapeEntries.map(([shapeId, metrics]) => html`
        <div class="dkao-bootstrap-shape-metrics">
          <strong>${shapeId}</strong>
          <dl>
            <dt>Correct</dt>
            <dd>
              <span class="pill ${metrics?.passed ? "success" : "failed"}">
                ${metrics?.passed ? "PASS" : "FAIL"}
              </span>
            </dd>
            <dt>Median</dt><dd>${metric(metrics?.median_us)} µs</dd>
            <dt>P90</dt><dd>${metric(metrics?.p90_us)} µs</dd>
            <dt>INT8 TOPS</dt>
            <dd>${metric(metrics?.logical_tops ?? metrics?.tflops, 3)}</dd>
            <dt>Algorithmic BW</dt>
            <dd>${metric(metrics?.algorithmic_bandwidth_gb_s ?? metrics?.bandwidth_gb_s, 1)} GB/s</dd>
          </dl>
        </div>
      `)}
      ${attempt.error ? html`<small>${attempt.error}</small>` : null}
    </div>
  `;
}

function IterationCard({ experiment }) {
  const metrics = experiment.metrics || {};
  return html`
    <div class="dkao-iteration-card ${experiment.accepted ? "accepted" : ""}">
      <div class="dkao-iteration-title">
        <strong>Round ${experiment.iteration}</strong>
        <span class="pill ${experiment.accepted ? "success" : "idle"}">
          ${experiment.accepted ? "best" : "tested"}
        </span>
      </div>
      <dl>
        <dt>Shape</dt><dd>${experiment.shape_id || "—"}</dd>
        <dt>Correct</dt>
        <dd>
          <span class="pill ${experiment.correctness_passed ? "success" : "failed"}">
            ${experiment.correctness_passed ? "PASS" : "FAIL"}
          </span>
        </dd>
        <dt>P90</dt><dd>${metric(metrics.p90_us)} µs</dd>
        <dt>INT8 TOPS</dt>
        <dd>${metric(metrics.logical_tops ?? metrics.tflops, 3)}</dd>
        <dt>Algorithmic BW</dt>
        <dd>${metric(metrics.algorithmic_bandwidth_gb_s ?? metrics.bandwidth_gb_s, 1)} GB/s</dd>
        <dt>Counter HBM BW</dt>
        <dd>${metric(experiment.pmc_evidence?.memory_traffic?.counter_derived_hbm_bandwidth_gb_s, 1)} GB/s</dd>
        <dt>Baseline</dt><dd>${metric(experiment.baseline_us)} µs</dd>
        <dt>Speedup</dt><dd>${metric(experiment.speedup, 3)}×</dd>
        <dt>Test Time</dt><dd>${metric(metrics.median_us)} µs</dd>
        <dt>Code snapshot</dt>
        <dd>${experiment.artifact_dir || "—"}</dd>
      </dl>
      ${experiment.failure_reason
        ? html`<small>${experiment.failure_reason}</small>` : null}
    </div>
  `;
}

function RunningIterationCard({ active }) {
  const activity = active.state === "profiling_current_best"
    ? "profiling"
    : active.state === "repairing_candidate"
      ? "repairing"
      : active.state === "validating_candidate"
        ? "validating"
        : "optimizing";
  return html`
    <div class="dkao-iteration-card running">
      <div class="dkao-iteration-title">
        <strong>Round ${active.iteration}</strong>
        <span class="pill running">${activity}</span>
      </div>
      <dl>
        <dt>Shape</dt><dd>${active.shape_id || "—"}</dd>
        <dt>Step</dt><dd>${active.step || "Agent optimizing kernel"}</dd>
        ${active.state === "repairing_candidate"
          ? html`<dt>Repair</dt><dd>${active.repair || 0}/${active.max_repairs || 4}</dd>`
          : null}
        <dt>Agent</dt><dd>${active.agent_status || "running"}</dd>
        <dt>Elapsed</dt><dd>${metric(active.elapsed_s, 1)} s</dd>
        <dt>Output</dt>
        <dd>${Number.isFinite(Number(active.last_output_age_s))
          ? `${metric(active.last_output_age_s, 1)} s ago`
          : "waiting"}</dd>
      </dl>
      <div class="dkao-running-pulse"><span></span></div>
    </div>
  `;
}

function PlanCard({ experiment }) {
  return html`
    <div class="dkao-plan-card ${experiment.manual_guidance ? "manual" : ""}">
      <div class="dkao-plan-title">
        <strong>Round ${experiment.iteration}</strong>
        ${experiment.manual_guidance
          ? html`<span class="pill active">manual</span>`
          : html`<span class="pill idle">agent</span>`}
      </div>
      <p>${experiment.hypothesis || "No optimization plan recorded."}</p>
      ${(experiment.changes || []).length
        ? html`<small>${experiment.changes.join(" · ")}</small>` : null}
    </div>
  `;
}

function BootstrapPlanCard({ attempt }) {
  const files = attempt.generated_files || [];
  return html`
    <div class="dkao-plan-card">
      <div class="dkao-plan-title">
        <strong>Bootstrap ${attempt.attempt}</strong>
        <span class="pill ${statusClass(attempt.status)}">${attempt.status}</span>
      </div>
      <p>${attempt.hypothesis}</p>
      ${files.length ? html`<small>${files.join(" · ")}</small>` : null}
      ${attempt.error ? html`<small>${attempt.error}</small>` : null}
    </div>
  `;
}

function GuidanceCard({ item }) {
  return html`
    <div class="dkao-plan-card guidance ${item.status}">
      <div class="dkao-plan-title">
        <strong>${item.status === "pending" ? "Next round" : `Round ${item.consumed_iteration}`}</strong>
        <span class="pill ${item.status === "pending" ? "running" : "success"}">
          ${item.status}
        </span>
      </div>
      <p>${item.text}</p>
      <small>Manual optimization guidance</small>
    </div>
  `;
}

function WorkerLane({ lane, taskId, onSaved }) {
  const [text, setText] = useState("");
  const [saving, setSaving] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [message, setMessage] = useState("");
  const submit = async (event) => {
    event.preventDefault();
    if (!text.trim() || saving || !lane.assigned) return;
    setSaving(true);
    setMessage("");
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      await postJson(
        `${base}/workers/${encodeURIComponent(lane.worker_id)}/guidance`,
        { text: text.trim() },
      );
      setText("");
      setMessage("Queued for this worker's next round.");
      await onSaved();
    } catch (err) {
      setMessage(String(err));
    } finally {
      setSaving(false);
    }
  };
  const restart = async () => {
    if (restarting || !["failed", "timed_out"].includes(lane.state)) return;
    if (!window.confirm(
      `Restart ${lane.worker_id} on GPU ${lane.gpu}? Other GPU workers will keep running.`,
    )) return;
    setRestarting(true);
    setMessage("");
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      await postJson(
        `${base}/workers/${encodeURIComponent(lane.worker_id)}/restart`, {},
      );
      setMessage(`Restarted on GPU ${lane.gpu}; sibling workers were not interrupted.`);
      await onSaved();
    } catch (err) {
      setMessage(String(err));
    } finally {
      setRestarting(false);
    }
  };
  const pending = (lane.guidance || []).filter((item) => item.status === "pending");
  const bootstrap = lane.bootstrap_attempts || [];
  const rounds = lane.experiments || [];
  const currentIteration = Number(lane.current_iteration || 0);
  const currentShape = lane.current_shape;
  const activeStates = new Set([
    "profiling_current_best",
    "agent_running",
    "validating_candidate",
    "repairing_candidate",
    "recording_result",
  ]);
  const currentRecorded = rounds.some((item) =>
    Number(item.iteration || 0) === currentIteration
      && item.shape_id === currentShape
  );
  const active = lane.active_iteration || (
    currentIteration > 0
      && activeStates.has(lane.state)
      && !currentRecorded
      ? {
          iteration: currentIteration,
          shape_id: currentShape,
          state: lane.state,
          step: lane.step,
          agent_name: lane.agent?.name,
          agent_status: lane.agent?.status,
          elapsed_s: lane.agent?.elapsed_s,
          last_output_age_s: lane.agent?.last_output_age_s,
          repair: lane.repair,
          max_repairs: lane.max_repairs,
        }
      : null
  );
  return html`
    <div class="dkao-worker-lane ${lane.assigned ? "" : "unassigned"}">
      <div class="dkao-worker-meta">
        <strong>${lane.worker_id}</strong>
        <span>GPU ${lane.gpu}</span>
        <span class="pill ${statusClass(lane.state)}">
          ${lane.state}
        </span>
        <small>${lane.assigned_shapes?.join(", ") || "No shapes assigned"}</small>
        ${["failed", "timed_out"].includes(lane.state) ? html`
          <button
            class="btn dkao-restart-worker"
            type="button"
            disabled=${restarting}
            onClick=${restart}
          >${restarting ? "Restarting…" : "Restart worker"}</button>
        ` : null}
      </div>
      <div class="dkao-lane-section">
        <div class="dkao-lane-heading">
          <strong>Iteration results</strong>
          <span>
            ${bootstrap.length} bootstrap · ${rounds.length} completed
            ${active ? " · 1 optimizing" : ""}
          </span>
        </div>
        <div class="dkao-iteration-strip">
          ${bootstrap.map((item) => html`<${BootstrapResultCard} attempt=${item} />`)}
          ${rounds.map((item) => html`<${IterationCard} experiment=${item} />`)}
          ${active ? html`<${RunningIterationCard} active=${active} />` : null}
          ${!bootstrap.length && !rounds.length && !active
            ? html`<p class="muted">${lane.assigned ? "Waiting for iteration 1." : "Worker not assigned."}</p>`
            : null}
        </div>
      </div>
      <div class="dkao-lane-section dkao-plan-section">
        <div class="dkao-lane-heading">
          <strong>Optimization plans</strong>
          <span>${bootstrap.length + rounds.length} recorded · ${pending.length} pending</span>
        </div>
        <div class="dkao-plan-strip">
          ${bootstrap.map((item) => html`<${BootstrapPlanCard} attempt=${item} />`)}
          ${rounds.map((item) => html`<${PlanCard} experiment=${item} />`)}
          ${(lane.guidance || [])
            .filter((item) => item.status === "pending")
            .map((item) => html`<${GuidanceCard} item=${item} />`)}
          ${!bootstrap.length && !rounds.length && !(lane.guidance || []).length
            ? html`<p class="muted">No plan recorded.</p>` : null}
        </div>
        <form class="dkao-guidance-form" onSubmit=${submit}>
          <input
            value=${text}
            onInput=${(event) => setText(event.currentTarget.value)}
            placeholder="Guide this GPU worker in its next round…"
            maxlength="4000"
            disabled=${!lane.assigned || saving}
          />
          <button class="btn btn-primary" type="submit" disabled=${!text.trim() || !lane.assigned || saving}>
            ${saving ? "Saving…" : "Use next round"}
          </button>
        </form>
        ${message ? html`<small class="dkao-guidance-message">${message}</small>` : null}
      </div>
    </div>
  `;
}

function SkillFile({ skill, canPublish, canFuse, onPublish, onFuse, busy }) {
  return html`
    <details class="dkao-skill-file">
      <summary>
        <span>
          <strong>${skill.name}</strong>
          <small>${skill.kind} · ${skill.source}</small>
        </span>
        ${canFuse ? html`
          <button
            class="btn"
            type="button"
            disabled=${busy === `fuse-${skill.name}`}
            onClick=${(event) => {
              event.preventDefault();
              event.stopPropagation();
              onFuse(skill);
            }}
          >${busy === `fuse-${skill.name}` ? "Fusing…" : "融合到 Skill 库"}</button>
        ` : null}
        ${canPublish ? html`
          <button
            class="btn btn-primary"
            type="button"
            disabled=${busy === `publish-${skill.name}`}
            onClick=${(event) => {
              event.preventDefault();
              event.stopPropagation();
              onPublish(skill);
            }}
          >${busy === `publish-${skill.name}` ? "Adding…" : "Add to existing"}</button>
        ` : html`<span class="pill success">existing</span>`}
      </summary>
      <pre>${skill.content}</pre>
    </details>
  `;
}

function SkillFuseStatus({ status, onRollback }) {
  if (!status || status.status === "idle") return null;
  if (status.status === "running") {
    return html`<div class="task-banner"><span class="spin"></span> 融合中…（主 agent 正在扫描 skill 库并决策）</div>`;
  }
  if (status.status === "error") {
    return html`<div class="task-banner task-banner-err">融合失败: ${status.error || "unknown error"}</div>`;
  }
  const synced = status.synced_to_ccb;
  return html`
    <div class="task-banner">
      融合完成：${status.action === "merge" ? `已融入 "${status.name}"` : `新增 skill "${status.name}"`}
      ${synced && (synced.added?.length || synced.updated?.length)
        ? ` · 已同步到 ccb 库${synced.added?.length ? `（新增 ${synced.added.length}）` : ""}${synced.updated?.length ? `（更新 ${synced.updated.length}）` : ""}`
        : ""}
      ${status.action === "merge" ? html`
        <button class="btn" type="button"
          onClick=${() => onRollback(status.name)}>还原</button>
      ` : null}
      ${status.diff ? html`
        <details class="dkao-skill-diff"><summary>查看 diff</summary><pre>${status.diff}</pre></details>
      ` : null}
    </div>
  `;
}

function SkillLibrary({ taskId, onClose }) {
  const [library, setLibrary] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const load = useCallback(async () => {
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      setLibrary(await fetchJson(`${base}/skills`));
      setError("");
    } catch (err) {
      setError(String(err));
    }
  }, [taskId]);
  useEffect(() => { load(); }, [load]);
  // While a fusion job is running, poll the library so the status flips to
  // done/error without a manual refresh.
  useEffect(() => {
    if (!library || library.fuse_status?.status !== "running") return;
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, [library, load]);
  const publish = async (skill) => {
    if (!window.confirm(`Add "${skill.name}" to the existing skill library (dsh, mirrored to Claude)?`)) return;
    setBusy(`publish-${skill.name}`);
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      await postJson(
        `${base}/skills/${encodeURIComponent(skill.name)}/publish`, {},
      );
      await load();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy("");
    }
  };
  const fuse = async (skill) => {
    if (!window.confirm(
      `Have the main agent fuse "${skill.name}" into the existing skill library?`
    )) return;
    setBusy(`fuse-${skill.name}`);
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      await postJson(`${base}/skills/fuse`, { skill_name: skill.name });
      await load();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy("");
    }
  };
  const syncNow = async () => {
    setBusy("sync");
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      await postJson(`${base}/skills/sync`, {});
      await load();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy("");
    }
  };
  const rollback = async (name) => {
    if (!window.confirm(`Restore the last backup of skill "${name}"?`)) return;
    setBusy(`rollback-${name}`);
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      await postJson(`${base}/skills/${encodeURIComponent(name)}/rollback`, {});
      await load();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy("");
    }
  };
  return html`
    <section class="panel dkao-skill-library">
      <div class="dkao-skill-library-title">
        <div>
          <h2>Skill library</h2>
          <p class="muted">dsh 库为权威，融合/发布后自动同步到 Claude (ccb) 库。</p>
        </div>
        <button class="btn" type="button" disabled=${busy === "sync"}
          onClick=${syncNow}>${busy === "sync" ? "Syncing…" : "同步 Skill 库"}</button>
        <button class="btn" type="button" onClick=${onClose}>Close</button>
      </div>
      ${error ? html`<div class="task-banner task-banner-err">${error}</div>` : null}
      ${library?.publish_disabled_reason ? html`
        <div class="task-banner task-banner-err">
          ${library.publish_disabled_reason}
          ${library.quarantined_count
            ? ` (${library.quarantined_count} pending files quarantined)` : ""}
        </div>
      ` : null}
      <${SkillFuseStatus} status=${library?.fuse_status} onRollback=${rollback} />
      ${!library ? html`<p class="muted">Loading skills…</p>` : html`
        <div class="dkao-skill-columns">
          <div class="dkao-skill-column">
            <h3>Existing skills <span>${library.existing?.length || 0}</span></h3>
            <p class="muted">${library.existing_root}</p>
            <div class="dkao-skill-list">
              ${(library.existing || []).map((skill) => html`
                <${SkillFile} skill=${skill} canPublish=${false} />
              `)}
            </div>
          </div>
          <div class="dkao-skill-column pending">
            <h3>Pending skills <span>${library.pending?.length || 0}</span></h3>
            <p class="muted">Worker skills and the main-agent merged skill.</p>
            <div class="dkao-skill-list">
              ${(library.pending || []).map((skill) => html`
                <${SkillFile}
                  skill=${skill}
                  canPublish=${true}
                  canFuse=${true}
                  busy=${busy}
                  onPublish=${publish}
                  onFuse=${fuse}
                />
              `)}
              ${!(library.pending || []).length
                ? html`<p class="muted">No pending skills.</p>` : null}
            </div>
          </div>
        </div>
      `}
    </section>
  `;
}

export default function DcuKernelAutoOptDetail({ taskId, data }) {
  const [runtime, setRuntime] = useState({ summary: null, graph: null, lanes: [] });
  const [error, setError] = useState(null);
  const [showSkills, setShowSkills] = useState(false);
  const refresh = useCallback(async () => {
    if (!taskId) return;
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      const [summary, graph, iterations] = await Promise.all([
        fetchJson(`${base}/summary`),
        fetchJson(`${base}/state-graph`),
        fetchJson(`${base}/iterations`),
      ]);
      setRuntime({ summary, graph, lanes: iterations.workers || [] });
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }, [taskId]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [refresh]);

  // ---- variant library (加入 variant 按钮) ------------------------------ //
  const [variantBusy, setVariantBusy] = useState("");
  const [variantMsg, setVariantMsg] = useState("");
  const [variantIndex, setVariantIndex] = useState(null); // null = loading
  const loadVariants = useCallback(async () => {
    if (!taskId) return;
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      const data = await fetchJson(`${base}/variants`);
      // Map shape -> variant record (incl. median_us/speedup/source) so the
      // update flow can compare the existing variant against this task.
      setVariantIndex(new Map((data.variants || []).map((v) => [v.shape, v])));
    } catch (err) { /* non-fatal */ }
  }, [taskId]);
  useEffect(() => { loadVariants(); }, [loadVariants]);
  const addVariant = async (shape) => {
    const existing = variantIndex ? variantIndex.get(shape) : null;
    if (existing) {
      const candidate = bestRoundPerShape(runtime.lanes).get(shape) || {};
      const newUs = candidate.median_us;
      const oldUs = existing.median_us;
      let verdict = "";
      if (newUs != null && oldUs != null) {
        verdict = newUs < oldUs
          ? "（本次更快）"
          : newUs > oldUs ? "（本次更慢）" : "（持平）";
      }
      const ok = window.confirm(
        `该算子已有 variant：median ${oldUs ?? "?"} µs`
        + `（来源 ${existing.source || "?"}）。\n`
        + `本次候选：median ${newUs ?? "?"} µs ${verdict}。\n`
        + `用本次候选替换？旧文件会备份（.bak-*）。`
      );
      if (!ok) return;
    }
    setVariantBusy(shape);
    setVariantMsg("");
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      const res = await postJson(`${base}/variants`, { shape_id: shape });
      const meta = res.meta || {};
      const tp = meta.tp != null ? `tp${meta.tp}` : "tp?";
      setVariantMsg(
        `已加入 variant: ${meta.type} | ${meta.model} | ${tp} | ${meta.operator}`
        + `（${res.action === "updated" ? "更新已有分区" : "新增分区"}）`
      );
      await loadVariants();
    } catch (err) {
      setVariantMsg(`加入 variant 失败: ${String(err)}`);
    } finally {
      setVariantBusy("");
    }
  };

  const report = runtime.summary?.report;
  const run = runtime.summary?.run;
  const stopped = run?.final_status === "stopped";
  const stopReason = (run?.notes || []).join(" · ") || "The optimizer stopped before completion.";
  const smokeOnly = report?.mode === "real-agent-dcu-smoke"
    || ["Real agents + DCU (smoke harness)", "Infrastructure smoke (not operator optimization)"]
      .includes(runtime.summary?.plan?.execution_mode);
  const repoPath = runtime.summary?.plan?.kernel_repo;
  const repoName = repoPath ? String(repoPath).split("/").pop() : null;
  // ---- kernel repo rename ------------------------------------------- //
  const [renameBusy, setRenameBusy] = useState(false);
  const [renameMsg, setRenameMsg] = useState("");
  const renameRepo = async () => {
    if (!repoName) return;
    const target = window.prompt(
      "新的仓库名（kernel-repos/ 下的目录名）:", repoName
    );
    if (!target || target.trim() === "" || target.trim() === repoName) return;
    setRenameBusy(true);
    setRenameMsg("");
    try {
      const base = `/api/dcu-kernel-auto-opt/${encodeURIComponent(taskId)}`;
      const res = await postJson(`${base}/rename-repo`, {
        new_name: target.trim(),
      });
      setRenameMsg(`已重命名: ${res.old_name} → ${res.new_name}`);
      await refresh();
    } catch (err) {
      setRenameMsg(`重命名失败: ${String(err)}`);
    } finally {
      setRenameBusy(false);
    }
  };
  const bestRounds = bestRoundPerShape(runtime.lanes);
  const shapeOrder = new Map(
    (runtime.summary?.plan?.shapes || []).map((item, index) => [item.id, index]),
  );
  const bestRoundEntries = Array.from(bestRounds.entries())
    .sort(([a], [b]) => (shapeOrder.get(a) ?? 1e9) - (shapeOrder.get(b) ?? 1e9));
  return html`
    <div class="dkao-detail">
      ${error ? html`<div class="task-banner task-banner-err">${error}</div>` : null}
      ${smokeOnly ? html`
        <div class="task-banner task-banner-err">
          Infrastructure smoke only: this task did not execute the selected
          operator. Its metrics and generated skills are not valid operator
          optimization results.
        </div>
      ` : null}
      ${stopped ? html`
        <div class="task-banner task-banner-err">
          <strong>Task stopped in ${run?.current_phase || "startup"}:</strong>
          ${stopReason}
        </div>
      ` : null}

      ${repoName ? html`
        <section class="panel dkao-repo-panel">
          <h2>
            Kernel repository
            <span class="muted"> — ${repoName}</span>
          </h2>
          <div class="dkao-repo-actions">
            <button
              class="btn"
              disabled=${renameBusy}
              onClick=${renameRepo}
            >${renameBusy ? "Renaming…" : "重命名仓库"}</button>
            ${renameMsg ? html`<span class="dkao-rename-msg">${renameMsg}</span>` : null}
          </div>
        </section>
      ` : null}

      <section class="panel dkao-state-panel">
        <h2>
          State machine
          <span class="muted"> — currently: </span>
          <span class="dkao-current-phase">
            ${(() => {
              const nodes = runtime.graph?.nodes || [];
              const cur = runtime.graph?.current;
              const node = nodes.find(n => n.id === cur);
              return node ? node.label : (cur || "starting");
            })()}
          </span>
        </h2>
        <${StateMachine} graph=${runtime.graph} />
        ${runtime.graph?.current === "parallel_explore" ? html`
          <${ExploreGpuGrid} lanes=${runtime.lanes} />
        ` : null}
      </section>

      <section class="panel dkao-iterations-panel">
        <h2>Iterations <span class="muted">four isolated worker lanes</span></h2>
        <div class="dkao-worker-lanes">
          ${runtime.lanes.map((lane) => html`
            <${WorkerLane} lane=${lane} taskId=${taskId} onSaved=${refresh} />
          `)}
        </div>
      </section>

      <section class="panel dkao-validation-panel">
        <h2>Final serial validation</h2>
        ${bestRoundEntries.length ? html`
          <h3 class="dkao-validation-subtitle">
            Best optimization round per shape
            <span>accepted “best” round from each worker lane</span>
          </h3>
          <table class="iter-table">
            <thead>
              <tr><th>Shape</th><th>Worker</th><th>Best round</th><th>Performance</th><th>Speedup</th><th>INT8 TOPS</th><th>Algorithmic BW</th><th>Variant</th></tr>
            </thead>
            <tbody>
              ${bestRoundEntries.map(([shape, item]) => html`
                <tr>
                  <td>${shape}</td>
                  <td>${item.worker_id}</td>
                  <td class="perf-cell">${item.iteration ?? "—"}</td>
                  <td class="perf-cell">${metric(item.median_us)} µs</td>
                  <td>${metric(item.speedup, 3)}×</td>
                  <td>${metric(item.metrics.logical_tops ?? item.metrics.tflops, 3)}</td>
                  <td>${metric(item.metrics.algorithmic_bandwidth_gb_s ?? item.metrics.bandwidth_gb_s, 2)} GB/s</td>
                  <td>
                    <button class="btn" type="button"
                      disabled=${variantBusy === shape}
                      onClick=${() => addVariant(shape)}>
                      ${variantBusy === shape
                        ? "处理中…"
                        : (variantIndex && variantIndex.has(shape))
                          ? "更新 variant"
                          : "加入 variant"}
                    </button>
                  </td>
                </tr>
              `)}
            </tbody>
          </table>
          ${variantMsg ? html`<p class="muted dkao-variant-msg">${variantMsg}</p>` : null}
        ` : null}
        ${report ? html`
          <table class="iter-table">
            <thead>
              <tr><th>Shape</th><th>Worker</th><th>Performance</th><th>INT8 TOPS</th><th>Algorithmic BW</th><th>Result</th></tr>
            </thead>
            <tbody>
              ${Object.entries(report.final_validation || {}).map(([shape, item]) => html`
                <tr>
                  <td>${shape}</td>
                  <td>${item.worker_id || bestRounds.get(shape)?.worker_id || "—"}</td>
                  <td class="perf-cell">${metric(item.metrics?.median_us ?? item.median_us)} µs</td>
                  <td>${metric(item.metrics?.logical_tops ?? item.metrics?.tflops ?? item.logical_tops, 3)}</td>
                  <td>${metric(
                    item.metrics?.algorithmic_bandwidth_gb_s
                      ?? item.metrics?.bandwidth_gb_s
                      ?? item.algorithmic_bandwidth_gb_s,
                    2,
                  )} GB/s</td>
                  <td><span class="pill ${item.passed ? "success" : "failed"}">${item.passed ? "PASS" : "FAIL"}</span></td>
                </tr>
              `)}
            </tbody>
          </table>
        ` : html`<p class="muted">Final validation has not started.</p>`}
      </section>

      <section class="panel dkao-events-panel">
        <h2>Event timeline</h2>
        <div class="dkao-events">
          ${(data?.timeline?.events || []).map((event) => html`
            <div><code>${event.type}</code><span>${JSON.stringify(event.payload || {})}</span></div>
          `)}
        </div>
      </section>

      <section class="panel dkao-skill-entry">
        <div>
          <h2>Optimization skill library</h2>
          <p class="muted">Inspect worker findings and the main-agent synthesis, then publish manually.</p>
        </div>
        <button class="btn btn-primary" type="button" onClick=${() => setShowSkills(true)}>
          View skill library
        </button>
      </section>

      ${showSkills ? html`
        <${SkillLibrary} taskId=${taskId} onClose=${() => setShowSkills(false)} />
      ` : null}
    </div>
  `;
}
