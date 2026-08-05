/** SGLang Trace Analyze — task detail view.
 *
 * Three tabs: Summary Overview | Batch Detail | Optimization Hints
 */
import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";

const API = (taskId) => `/api/sglang_trace_analyze/${taskId}`;

export default function SADetail({ taskId }) {
  const [summary, setSummary] = useState(null);
  const [hints, setHints] = useState(null);
  const [detail, setDetail] = useState(null);
  const [activeTab, setActiveTab] = useState("summary");
  const [activeBatch, setActiveBatch] = useState(null);
  const [activeStage, setActiveStage] = useState("decode");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API(taskId)}/summary`).then((r) => r.json()),
      fetch(`${API(taskId)}/hints`).then((r) => r.json()),
    ])
      .then(([s, h]) => { setSummary(s); setHints(h); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [taskId]);

  useEffect(() => {
    if (!activeBatch) return;
    fetch(`${API(taskId)}/batch/${activeBatch}/${activeStage}`)
      .then((r) => r.json())
      .then((d) => setDetail(d))
      .catch(() => setDetail(null));
  }, [taskId, activeBatch, activeStage]);

  if (loading) return html`<div class="sa-loading">Loading analysis…</div>`;
  if (error) return html`<div class="sa-error">Error: ${error}</div>`;
  if (!summary || !summary.batches || summary.batches.length === 0) {
    return html`<div class="sa-empty">No analysis data available yet.</div>`;
  }

  const batchList = summary.batches || [];
  if (!activeBatch && batchList.length > 0) {
    setActiveBatch(batchList[0].batch_size);
  }

  return html`
    <div class="sa-detail">
      <div class="sa-header">
        <h2>SGLang Trace Analysis</h2>
        <span class="sa-meta">Model: ${summary.model || "?"} | GPU: ${summary.gpu || "?"}</span>
      </div>

      <div class="sa-tabs">
        <button class="sa-tab-btn ${activeTab === "summary" ? "active-tab" : ""}" onClick=${() => setActiveTab("summary")}>
          Summary Overview
        </button>
        <button class="sa-tab-btn ${activeTab === "batch" ? "active-tab" : ""}" onClick=${() => setActiveTab("batch")}>
          Batch Detail
        </button>
        <button class="sa-tab-btn ${activeTab === "hints" ? "active-tab" : ""}" onClick=${() => setActiveTab("hints")}>
          Optimization Hints
        </button>
      </div>

      ${activeTab === "summary" && html`<${SummaryPage} summary=${summary} />`}
      ${activeTab === "batch" && html`
        <div class="sa-batch-tabs">
          ${batchList.map((b) => html`
            <button class="sa-tab ${activeBatch === b.batch_size ? "active" : ""}" onClick=${() => setActiveBatch(b.batch_size)}>
              BS=${b.batch_size} ${b.stage}
            </button>
          `)}
        </div>
        ${detail ? html`
          <${KernelTable} kt=${detail.kernel_table} batch=${activeBatch} stage=${activeStage} />
          <${CategoryChart} kt=${detail.kernel_table} />
          <${OverlapPanel} ov=${detail.overlap} fu=${detail.fuse} />
        ` : html`<div class="sa-loading">Loading batch detail…</div>`}
      `}
      ${activeTab === "hints" && html`<${HintsPage} hints=${hints} />`}
    </div>
  `;
}

/* ── Summary Overview ── */

function SummaryPage({ summary }) {
  const batches = summary.batches || [];
  return html`
    <div class="sa-panel">
      <h3>Batch Summary</h3>
      <table class="sa-table">
        <thead>
          <tr><th>Batch</th><th>Stage</th><th>Top Kernel</th><th>%</th><th>Kernels</th><th>MFU Avg</th></tr>
        </thead>
        <tbody>
          ${batches.map((b) => html`
            <tr key=${b.batch_size + b.stage}>
              <td class="sa-num">${b.batch_size}</td>
              <td>${b.stage}</td>
              <td class="sa-kernel-name" title=${b.top_kernel || ""}>${(b.top_kernel || "").slice(0, 70)}</td>
              <td class="sa-num">${(b.top_kernel_pct || 0).toFixed(1)}%</td>
              <td class="sa-num">${b.kernel_count}</td>
              <td class="sa-num">${b.mfu_avg != null ? b.mfu_avg.toFixed(1) + "%" : "-"}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

/* ── Kernel Hotspot Table ── */

function KernelTable({ kt, batch, stage }) {
  if (!kt) return null;
  const kernels = kt.kernels || [];
  const totalTime = kt.total_gpu_time_s || 1;

  return html`
    <div class="sa-panel">
      <h3>Kernel Hotspots — BS=${batch} ${stage} (${kernels.length} unique, ${totalTime.toFixed(1)}s GPU)</h3>
      <div class="sa-table-wrap">
      <table class="sa-table">
        <thead>
          <tr><th>#</th><th>%</th><th>Category</th><th>Op</th><th>Layer</th><th>MFU</th><th>Bound</th><th>Count</th><th>Avg μs</th><th>Kernel</th></tr>
        </thead>
        <tbody>
          ${kernels.slice(0, 25).map((k) => html`
            <tr key=${k.rank}>
              <td class="sa-num">${k.rank}</td>
              <td class="sa-pct"><div class="sa-bar-bg"><div class="sa-bar" style="width:${Math.min(k.time_pct || 0, 100)}%"></div><span>${(k.time_pct || 0).toFixed(1)}%</span></div></td>
              <td><span class="sa-cat">${k.category || "?"}</span></td>
              <td class="sa-num sa-sm">${k.op_type || "?"}</td>
              <td class="sa-sm">${k.model_layer || "-"}</td>
              <td class="sa-num">${k.mfu != null ? k.mfu.toFixed(1) + "%" : "-"}</td>
              <td class="sa-sm">${k.bound || "-"}</td>
              <td class="sa-num">${k.count}</td>
              <td class="sa-num">${(k.avg_dur_us || 0).toFixed(1)}</td>
              <td class="sa-kernel-name" title=${k.kernel_name}>${(k.kernel_name || "").slice(0, 60)}</td>
            </tr>
          `)}
        </tbody>
      </table>
      </div>
    </div>
  `;
}

/* ── Category Breakdown ── */

function CategoryChart({ kt }) {
  if (!kt) return null;
  const kernels = kt.kernels || [];
  const cats = {};
  for (const k of kernels) {
    cats[k.category] = (cats[k.category] || 0) + (k.time_pct || 0);
  }
  const sorted = Object.entries(cats).sort((a, b) => b[1] - a[1]);
  const colors = ["#c0392b","#d35400","#e67e22","#27ae60","#2980b9","#8e44ad","#16a085","#7f8c8d","#2c3e50","#e91e63"];

  return html`
    <div class="sa-panel">
      <h3>Category Breakdown</h3>
      <div class="sa-cat-chart">
        ${sorted.map(([cat, pct], i) => html`
          <div class="sa-cat-row">
            <span class="sa-cat-label">${cat}</span>
            <div class="sa-cat-bar-bg">
              <div class="sa-cat-bar" style="width:${Math.min(pct, 100)}%;background:${colors[i % colors.length]}"></div>
            </div>
            <span class="sa-cat-pct">${pct.toFixed(1)}%</span>
          </div>
        `)}
      </div>
    </div>
  `;
}

/* ── Overlap + Fuse ── */

function OverlapPanel({ ov, fu }) {
  if (!ov) return null;
  const sum = ov.summary || {};
  const gaps = ov.gaps || [];
  const high = gaps.filter((g) => g.severity === "high").length;
  const medium = gaps.filter((g) => g.severity === "medium").length;
  const fuseMatches = (fu && fu.matches) || [];

  return html`
    <div class="sa-panel">
      <h3>Overlap & Fuse</h3>
      <p>
        <strong>${gaps.length}</strong> GPU idle gaps (${high} high, ${medium} medium)
        — total: <strong>${(sum.total_gap_us / 1000).toFixed(1)}ms</strong>
        (${sum.total_gap_pct || 0}% of GPU time)
      </p>
      <p class="sa-note">CUDA Graph effective: <strong>${sum.cuda_graph_effective ? "YES" : "NO"}</strong></p>
      ${fuseMatches.length > 0 && html`
        <h4>Fuse Pattern Matches</h4>
        ${fuseMatches.map((m) => html`
          <div class="sa-fuse-card">
            <strong>${m.pattern}</strong> (${m.confidence})
            <p class="sa-note">${m.suggestion}</p>
          </div>
        `)}
      `}
    </div>
  `;
}

/* ── Hints Page ── */

function HintsPage({ hints }) {
  if (!hints) return html`<div class="sa-panel"><p>No hints generated yet.</p></div>`;

  const b = hints.bottleneck || {};
  const suggestions = hints.suggestions || [];
  const surprises = hints.surprises || [];

  if (hints.status === "skipped") {
    return html`<div class="sa-panel"><h3>Optimization Hints</h3><p class="sa-note">Hints generation skipped (${hints.reason || "not wired"}).</p></div>`;
  }

  return html`
    <div class="sa-panel">
      <h3>Optimization Hints</h3>
      <div class="sa-hint-section">
        <h4>Biggest Bottleneck</h4>
        <p><strong>${b.kernel_or_pattern || "?"}</strong> — ${b.reason || ""} (impact: ${b.impact_pct || 0}%)</p>
      </div>
      ${suggestions.length > 0 && html`
        <div class="sa-hint-section">
          <h4>Suggestions</h4>
          ${suggestions.map((s) => html`
            <div class="sa-hint-card">
              <strong>${s.title}</strong> <span class="sa-difficulty sa-diff-${s.difficulty || "medium"}">${s.difficulty}</span>
              <p>${s.what_to_change}</p>
              <p class="sa-note">Why: ${s.why} | Est. saving: ${s.estimated_saving_pct}% | Type: ${s.category}</p>
            </div>
          `)}
        </div>
      `}
      ${surprises.length > 0 && html`
        <div class="sa-hint-section">
          <h4>Surprises</h4>
          ${surprises.map((s) => html`<p class="sa-note">${s}</p>`)}
        </div>
      `}
    </div>
  `;
}
