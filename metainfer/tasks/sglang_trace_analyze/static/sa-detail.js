/** SGLang Trace Analyze — task detail view.
 *
 * Designed for GPU inference optimization engineers.
 * Three tabs: Dashboard | Batch Detail | Optimization Hints
 */
import { html } from "htm/preact";
import { useEffect, useState, useMemo } from "preact/hooks";

const API = (taskId) => `/api/sglang_trace_analyze/${taskId}`;

export default function SADetail({ taskId }) {
  const [summary, setSummary] = useState(null);
  const [hints, setHints] = useState(null);
  const [detail, setDetail] = useState(null);
  const [activeTab, setActiveTab] = useState("dashboard");
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
  if (!activeBatch && batchList.length > 0) setActiveBatch(batchList[0].batch_size);

  return html`
    <div class="sa-detail">
      <div class="sa-header">
        <h2>Trace Analysis</h2>
        <span class="sa-meta">${summary.model || "?"} | ${summary.gpu || "?"}</span>
      </div>

      <div class="sa-tabs">
        <button class="sa-tab-btn ${activeTab === "dashboard" ? "active-tab" : ""}" onClick=${() => setActiveTab("dashboard")}>Dashboard</button>
        <button class="sa-tab-btn ${activeTab === "batch" ? "active-tab" : ""}" onClick=${() => setActiveTab("batch")}>Kernel Table</button>
        <button class="sa-tab-btn ${activeTab === "hints" ? "active-tab" : ""}" onClick=${() => setActiveTab("hints")}>Hints</button>
      </div>

      ${activeTab === "dashboard" && html`<${Dashboard} summary=${summary} detail=${detail} batchList=${batchList} activeBatch=${activeBatch} setActiveBatch=${setActiveBatch} />`}
      ${activeTab === "batch" && html`
        <div class="sa-batch-tabs">
          ${batchList.map((b) => html`
            <button class="sa-tab ${activeBatch === b.batch_size ? "active" : ""}" onClick=${() => setActiveBatch(b.batch_size)}>BS=${b.batch_size} ${b.stage}</button>
          `)}
        </div>
        ${detail ? html`<${KernelTable} kt=${detail.kernel_table} batch=${activeBatch} stage=${activeStage} />` : html`<div class="sa-loading">Loading…</div>`}
      `}
      ${activeTab === "hints" && html`<${HintsPage} hints=${hints} detail=${detail} />`}
    </div>
  `;
}

/* ═══════════════════════════════════════════════════════════════════════
   DASHBOARD
   ═══════════════════════════════════════════════════════════════════════ */

function Dashboard({ summary, detail, batchList, activeBatch, setActiveBatch }) {
  if (!detail) return html`<div class="sa-loading">Loading dashboard…</div>`;
  const kt = detail.kernel_table;
  if (!kt) return null;
  const kernels = kt.kernels || [];

  // Compute stats
  const top = kernels[0] || {};
  const cats = {};
  let mfuVals = [], totalDur = kt.total_gpu_time_s || 0;
  for (const k of kernels) {
    cats[k.category] = (cats[k.category] || 0) + (k.time_pct || 0);
    if (k.mfu != null && k.mfu > 0) mfuVals.push(k.mfu);
    if (k.tflops_actual != null && k.tflops_actual > 0) mfuVals.push(k.tflops_actual / (k.tflops_theoretical || 192) * 100);
  }
  const avgMfu = mfuVals.length ? (mfuVals.reduce((a, b) => a + b, 0) / mfuVals.length).toFixed(1) : null;
  const computePct = kernels.filter(k => k.bound === "compute").reduce((s, k) => s + (k.time_pct || 0), 0);
  const memoryPct = kernels.filter(k => k.bound === "memory").reduce((s, k) => s + (k.time_pct || 0), 0);
  const unknownBound = 100 - computePct - memoryPct;

  const ov = detail.overlap || {};
  const cudaGraphOk = (ov.summary || {}).cuda_graph_effective;
  const gapCount = (ov.gaps || []).length;

  // Category colors
  const catColors = { Reduce: "#c0392b", GEMM: "#d35400", ElementWise: "#e67e22", MoE: "#27ae60",
    NCCL: "#e74c3c", Attention: "#8e44ad", Norm: "#2980b9", Indexing: "#16a085", Memory: "#f1c40f",
    Quantization: "#2c3e50", Other: "#7f8c8d", Transform: "#2ecc71", Activation: "#e91e63" };

  const sortedCats = Object.entries(cats).sort((a, b) => b[1] - a[1]);
  const totalPct = sortedCats.reduce((s, [, v]) => s + v, 0);
  // Build conic-gradient stops for the donut
  const donutStops = [];
  let acc = 0;
  for (const [cat, pct] of sortedCats) {
    donutStops.push(`${catColors[cat] || "#95a5a6"} ${acc}% ${acc + pct}%`);
    acc += pct;
  }

  return html`
    <div class="sa-dashboard">
      ${/* Row 1: Quick stats */""}
      <div class="sa-stats-row">
        <div class="sa-stat-card">
          <div class="sa-stat-value">${totalDur.toFixed(2)}s</div>
          <div class="sa-stat-label">Total GPU Time</div>
        </div>
        <div class="sa-stat-card sa-stat-warn">
          <div class="sa-stat-value">${(top.time_pct || 0).toFixed(1)}%</div>
          <div class="sa-stat-label">Top Bottleneck</div>
          <div class="sa-stat-sub">${(top.kernel_name || "").slice(0, 40)}</div>
        </div>
        <div class="sa-stat-card">
          <div class="sa-stat-value">${avgMfu != null ? avgMfu + "%" : "—"}</div>
          <div class="sa-stat-label">Avg MFU (BF16)</div>
        </div>
        <div class="sa-stat-card ${cudaGraphOk ? "sa-stat-ok" : "sa-stat-warn"}">
          <div class="sa-stat-value">${cudaGraphOk ? "ON" : "OFF"}</div>
          <div class="sa-stat-label">CUDA Graph</div>
        </div>
      </div>

      ${/* Row 2: Category donut + Bound breakdown + Bottleneck detail */""}
      <div class="sa-grid-2col">
        <div class="sa-panel">
          <h3>GPU Time by Category</h3>
          <div class="sa-donut-wrap">
            <div class="sa-donut" style="background: conic-gradient(${donutStops.join(", ")});">
              <div class="sa-donut-hole">
                <span class="sa-donut-val">${kernels.length}</span>
                <span class="sa-donut-lbl">kernels</span>
              </div>
            </div>
            <div class="sa-donut-legend">
              ${sortedCats.slice(0, 8).map(([cat, pct]) => html`
                <div class="sa-legend-item">
                  <span class="sa-legend-swatch" style="background:${catColors[cat] || "#95a5a6"}"></span>
                  <span class="sa-legend-name">${cat}</span>
                  <span class="sa-legend-pct">${pct.toFixed(1)}%</span>
                </div>
              `)}
            </div>
          </div>
        </div>

        <div class="sa-panel">
          <h3>Bottleneck Detail</h3>
          <div class="sa-bottleneck">
            <div class="sa-bn-rank">#1</div>
            <div class="sa-bn-info">
              <div class="sa-bn-name">${top.kernel_name || "?"}</div>
              <div class="sa-bn-meta">
                Category: <strong>${top.category || "?"}</strong> |
                Op: <strong>${top.op_type || "?"}</strong> |
                Count: <strong>${top.count || 0}</strong>
              </div>
              <div class="sa-bn-meta">
                Layer: <strong>${top.model_layer || "unknown"}</strong> |
                Bound: <strong>${top.bound || "unknown"}</strong> |
                MFU: <strong>${top.mfu != null ? top.mfu.toFixed(1) + "%" : "—"}</strong>
              </div>
              <div class="sa-bn-bar-wrap">
                <div class="sa-bn-bar" style="width:${Math.min(top.time_pct || 0, 100)}%"></div>
              </div>
            </div>
          </div>

          <h3 style="margin-top:14px">Compute vs Memory Bound</h3>
          <div class="sa-bound-bars">
            <div class="sa-bound-row">
              <span class="sa-bound-label">Compute-bound</span>
              <div class="sa-bound-bar-bg"><div class="sa-bound-bar sa-bb-compute" style="width:${computePct}%"></div></div>
              <span class="sa-bound-pct">${computePct.toFixed(1)}%</span>
            </div>
            <div class="sa-bound-row">
              <span class="sa-bound-label">Memory-bound</span>
              <div class="sa-bound-bar-bg"><div class="sa-bound-bar sa-bb-memory" style="width:${memoryPct}%"></div></div>
              <span class="sa-bound-pct">${memoryPct.toFixed(1)}%</span>
            </div>
            <div class="sa-bound-row">
              <span class="sa-bound-label">Unknown</span>
              <div class="sa-bound-bar-bg"><div class="sa-bound-bar sa-bb-unknown" style="width:${unknownBound}%"></div></div>
              <span class="sa-bound-pct">${unknownBound.toFixed(1)}%</span>
            </div>
          </div>

          <h3 style="margin-top:14px">Overlap</h3>
          <p>${gapCount} idle gaps detected. ${cudaGraphOk ? "CUDA Graph is active — gaps are minimal." : "CUDA Graph is OFF — explore enabling it."}</p>
        </div>
      </div>

      ${/* Row 3: Top kernels quick preview */""}
      <div class="sa-panel">
        <h3>Top Kernels</h3>
        <table class="sa-table">
          <thead><tr><th>#</th><th>%</th><th>Category</th><th>Kernel</th><th>Count</th><th>Avg μs</th><th>MFU</th><th>Bound</th></tr></thead>
          <tbody>
            ${kernels.slice(0, 10).map((k) => html`
              <tr key=${k.rank}>
                <td class="sa-num">${k.rank}</td>
                <td class="sa-pct"><div class="sa-bar-bg"><div class="sa-bar" style="width:${Math.min(k.time_pct || 0, 100)}%"></div><span>${(k.time_pct || 0).toFixed(1)}%</span></div></td>
                <td><span class="sa-cat">${k.category || "?"}</span></td>
                <td class="sa-kernel-name" title=${k.kernel_name}>${(k.kernel_name || "").slice(0, 55)}</td>
                <td class="sa-num">${k.count}</td>
                <td class="sa-num">${(k.avg_dur_us || 0).toFixed(1)}</td>
                <td class="sa-num">${k.mfu != null ? k.mfu.toFixed(1) + "%" : "—"}</td>
                <td class="sa-sm">${k.bound || "?"}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL TABLE (full, searchable)
   ═══════════════════════════════════════════════════════════════════════ */

function KernelTable({ kt, batch, stage }) {
  if (!kt) return null;
  const kernels = kt.kernels || [];
  const totalTime = kt.total_gpu_time_s || 1;
  const [search, setSearch] = useState("");
  const [catFilter, setCatFilter] = useState("all");

  const categories = [...new Set(kernels.map((k) => k.category || "Other"))];
  const filtered = kernels.filter((k) => {
    if (catFilter !== "all" && k.category !== catFilter) return false;
    if (search && !k.kernel_name.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  return html`
    <div class="sa-panel">
      <h3>Kernel Hotspots — BS=${batch} ${stage} (${kernels.length} unique, ${totalTime.toFixed(1)}s GPU)</h3>
      <div class="sa-filters">
        <input class="sa-search" placeholder="Search kernel name…" value=${search} onInput=${(e) => setSearch(e.target.value)} />
        <select class="sa-select" value=${catFilter} onChange=${(e) => setCatFilter(e.target.value)}>
          <option value="all">All categories</option>
          ${categories.map((c) => html`<option value=${c}>${c}</option>`)}
        </select>
        <span class="sa-filter-count">${filtered.length} of ${kernels.length} kernels</span>
      </div>
      <div class="sa-table-wrap">
        <table class="sa-table">
          <thead><tr><th>#</th><th>%</th><th>Category</th><th>Op</th><th>Layer</th><th>Count</th><th>Avg μs</th><th>MFU</th><th>Bound</th><th>Conf</th><th>Kernel</th></tr></thead>
          <tbody>
            ${filtered.slice(0, 100).map((k) => html`
              <tr key=${k.rank}>
                <td class="sa-num">${k.rank}</td>
                <td class="sa-pct"><div class="sa-bar-bg"><div class="sa-bar" style="width:${Math.min(k.time_pct || 0, 100)}%"></div><span>${(k.time_pct || 0).toFixed(1)}%</span></div></td>
                <td><span class="sa-cat">${k.category || "?"}</span></td>
                <td class="sa-sm">${k.op_type || "?"}</td>
                <td class="sa-sm">${k.model_layer || "—"}</td>
                <td class="sa-num">${k.count}</td>
                <td class="sa-num">${(k.avg_dur_us || 0).toFixed(1)}</td>
                <td class="sa-num">${k.mfu != null ? k.mfu.toFixed(1) + "%" : "—"}</td>
                <td class="sa-sm">${k.bound || "—"}</td>
                <td><span class="sa-conf sa-conf-${k.confidence || "low"}">${k.confidence || "?"}</span></td>
                <td class="sa-kernel-name" title=${k.kernel_name}>${(k.kernel_name || "").slice(0, 60)}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

/* ═══════════════════════════════════════════════════════════════════════
   HINTS PAGE
   ═══════════════════════════════════════════════════════════════════════ */

function HintsPage({ hints, detail }) {
  const kt = detail ? detail.kernel_table : null;
  const fuse = detail ? detail.fuse : null;
  const fuseMatches = (fuse && fuse.matches) || [];

  return html`
    ${kt && html`
      <${BottleneckAnalysis} kt=${kt} />
    `}

    ${fuseMatches.length > 0 && html`
      <div class="sa-panel">
        <h3>Fuse Opportunities (${fuseMatches.length})</h3>
        ${fuseMatches.map((m) => html`
          <div class="sa-hint-card">
            <strong>${m.pattern}</strong>
            <span class="sa-conf sa-conf-${m.confidence}">${m.confidence}</span>
            <span class="sa-note ml8">~${m.estimated_saving_us}μs saving</span>
            <p class="sa-note">${m.suggestion}</p>
          </div>
        `)}
      </div>
    `}

    ${hints && hints.status !== "skipped" && html`
      <div class="sa-panel">
        <h3>AI Optimization Hints</h3>
        ${(hints.suggestions || []).map((s) => html`
          <div class="sa-hint-card">
            <strong>${s.title}</strong>
            <span class="sa-difficulty sa-diff-${s.difficulty || "medium"}">${s.difficulty}</span>
            <span class="sa-note ml8">Est. saving: ${s.estimated_saving_pct}%</span>
            <p>${s.what_to_change}</p>
            <p class="sa-note">${s.why} | Type: ${s.category}</p>
          </div>
        `)}
      </div>
    `}

    ${(hints && hints.status === "skipped" && fuseMatches.length === 0 && !kt) && html`
      <div class="sa-panel"><h3>Optimization Hints</h3><p class="sa-note">No hints or fuse matches available yet.</p></div>
    `}
  `;
}

function BottleneckAnalysis({ kt }) {
  if (!kt) return null;
  const kernels = kt.kernels || [];
  const top = kernels[0];
  const top3 = kernels.slice(0, 3);

  const computeBoundPct = kernels.filter(k => k.bound === "compute").reduce((s, k) => s + (k.time_pct || 0), 0);
  const suggestions = [];
  if (computeBoundPct < 30) suggestions.push("Most kernels are memory-bound — focus on kernel fusion to reduce memory traffic.");
  if ((top.time_pct || 0) > 50) suggestions.push(`"${(top.kernel_name || "").slice(0, 40)}" dominates at ${(top.time_pct || 0).toFixed(1)}%. Consider optimizing or replacing this kernel.`);
  if (suggestions.length === 0) suggestions.push("GPU time is spread across many kernels. Look for fusion opportunities in the table below.");

  return html`
    <div class="sa-panel">
      <h3>Bottleneck Analysis</h3>
      <div class="sa-bn-list">
        ${top3.map((k, i) => html`
          <div class="sa-bn-row">
            <span class="sa-bn-rank-sm">#${i + 1}</span>
            <span class="sa-bn-pct">${(k.time_pct || 0).toFixed(1)}%</span>
            <span class="sa-bn-name-sm">${(k.kernel_name || "").slice(0, 60)}</span>
            <span class="sa-cat">${k.category || "?"}</span>
          </div>
        `)}
      </div>
      ${suggestions.map((s) => html`<p class="sa-suggestion">${s}</p>`)}
    </div>
  `;
}
