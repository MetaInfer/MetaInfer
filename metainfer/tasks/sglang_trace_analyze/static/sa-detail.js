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
  const [mapping, setMapping] = useState(null);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [activeBatch, setActiveBatch] = useState(null);
  const [activeStage, setActiveStage] = useState("decode");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API(taskId)}/summary`).then((r) => r.json()),
      fetch(`${API(taskId)}/hints`).then((r) => r.json()),
      fetch(`${API(taskId)}/mapping`).then((r) => r.json()),
    ])
      .then(([s, h, m]) => { setSummary(s); setHints(h); setMapping(m); setLoading(false); })
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

      ${activeTab === "dashboard" && html`<${Dashboard} summary=${summary} detail=${detail} mapping=${mapping} batchList=${batchList} activeBatch=${activeBatch} setActiveBatch=${setActiveBatch} />`}
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

function Dashboard({ summary, detail, mapping, batchList, activeBatch, setActiveBatch }) {
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

      ${/* Row 3: TFLOPS & Bandwidth + Structure Mapping */""}
      <div class="sa-grid-2col">
        <${TflopsPanel} kernels=${kernels} gpu=${summary.gpu || "K100"} />
        <${StructureMappingPanel} mapping=${mapping} kernels=${kernels} />
      </div>

      ${/* Row 4: Fuse + Mapping confidence */""}
      <${FusePanel} detail=${detail} />

      ${/* Row 5: Inefficiency radar + roofline */""}
      <div class="sa-grid-2col">
        <${InefficiencyRadar} kernels=${kernels} />
        <${RooflinePanel} kernels=${kernels} gpu=${summary.gpu || "K100"} />
      </div>

      ${/* Row 4: Key Findings */""}
      <${KeyFindings} kernels=${kernels} kt=${kt} cudaGraph=${cudaGraphOk} />

      ${/* Row 5: MFU Distribution + Frequency Analysis */""}
      <div class="sa-grid-2col">
        <${MfuDistro} kernels=${kernels} gpu=${summary.gpu || "K100"} />
        <${FrequencyPanel} kernels=${kernels} />
      </div>

      ${/* Row 6: Top kernels quick preview */""}
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

/* ── Inefficiency Radar: high-time, low-MFU kernels ── */

function InefficiencyRadar({ kernels }) {
  if (!kernels || !kernels.length) return null;
  // Top kernels by (time_pct * (100 - mfu)) / 100 — high time, low efficiency
  const inefficiency = kernels
    .filter((k) => (k.time_pct || 0) > 0.03)
    .map((k) => ({
      ...k,
      waste: ((k.time_pct || 0) * (k.mfu != null ? Math.max(0, 100 - k.mfu) : 100)) / 100,
    }))
    .sort((a, b) => b.waste - a.waste);

  return html`
    <div class="sa-panel">
      <h3>Inefficiency Radar</h3>
      <p class="sa-note">Kernels with high GPU time and low MFU — biggest optimization potential.</p>
      <table class="sa-table">
        <thead><tr><th>Kernel</th><th>Time%</th><th>MFU</th><th>Waste Score</th><th>Category</th></tr></thead>
        <tbody>
          ${inefficiency.slice(0, 8).map((k) => html`
            <tr key=${k.rank}>
              <td class="sa-kernel-name" title=${k.kernel_name}>${(k.kernel_name || "").slice(0, 50)}</td>
              <td class="sa-num">${(k.time_pct || 0).toFixed(1)}%</td>
              <td class="sa-num">${k.mfu != null ? k.mfu.toFixed(1) + "%" : "—"}</td>
              <td class="sa-num">${k.waste.toFixed(1).replace(/^-/, "")}</td>
              <td><span class="sa-cat">${k.category || "?"}</span></td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

/* ── Roofline Analysis ── */

function RooflinePanel({ kernels, gpu }) {
  if (!kernels || !kernels.length) return null;

  // GPU peaks
  const peaks = { K100: { bf16: 192, bw: 700 }, A100_80G: { bf16: 312, bw: 2039 },
    H100: { bf16: 989, bw: 3350 }, B200: { bf16: 2250, bw: 8000 } };
  const pk = peaks[gpu] || peaks.K100;
  const peakFlops = pk.bf16 * 1e12; // TFLOPS → FLOPS
  const peakBw = pk.bw * 1e9;       // GB/s → B/s
  const ridgePoint = peakFlops / peakBw; // ops/byte at the ridge

  // Classify each kernel with valid data
  const pts = kernels
    .filter((k) => k.tflops_actual != null && k.tflops_actual > 0 && k.bandwidth_gb_s != null && k.bandwidth_gb_s > 0)
    .map((k) => ({
      name: k.kernel_name, category: k.category, time_pct: k.time_pct,
      flops: k.tflops_actual * 1e12, bw: k.bandwidth_gb_s * 1e9,
      opsPerByte: (k.tflops_actual * 1e12) / (k.bandwidth_gb_s * 1e9),
      bound: k.bound, rank: k.rank,
    }));

  const computeBound = pts.filter((p) => p.bound === "compute").length;
  const memoryBound = pts.filter((p) => p.bound === "memory").length;

  return html`
    <div class="sa-panel">
      <h3>Roofline Analysis</h3>
      <p class="sa-note">GPU: ${gpu} | Peak BF16: ${pk.bf16} TFLOPS | BW: ${pk.bw} GB/s | Ridge: ${ridgePoint.toFixed(0)} ops/byte</p>
      <p>
        <strong>${computeBound}</strong> compute-bound |
        <strong>${memoryBound}</strong> memory-bound
        ${pts.length < 5 ? html`<span class="sa-note"> (${kernels.length - pts.length} kernels lack dims for roofline)</span>` : ""}
      </p>
      <div class="sa-roofline-bars">
        ${pts.slice(0, 12).map((p) => {
          const barW = Math.min(Math.log10(Math.max(p.opsPerByte, 1)) / Math.log10(ridgePoint * 10) * 100, 100);
          const onRidge = p.opsPerByte > ridgePoint;
          return html`
            <div class="sa-rf-row">
              <span class="sa-rf-name" title=${p.name}>${(p.name || "").slice(0, 40)}</span>
              <span class="sa-rf-bar-wrap">
                <div class="sa-rf-bar ${onRidge ? "sa-rf-compute" : "sa-rf-memory"}" style="width:${barW}%"></div>
              </span>
              <span class="sa-rf-val">${p.opsPerByte.toFixed(0)} op/B</span>
              <span class="sa-rf-bound">${onRidge ? "compute" : "memory"}</span>
            </div>
          `;
        })}
      </div>
      <p class="sa-note">Ridge point: ${ridgePoint.toFixed(0)} ops/byte. Left of ridge = memory-bound. Right = compute-bound.</p>
    </div>
  `;
}

/* ── Key Findings auto-summary ── */

function KeyFindings({ kernels, kt, cudaGraph }) {
  if (!kernels || !kernels.length) return null;

  const total = kt.total_gpu_time_s || 0;
  const top = kernels[0];
  const top3 = kernels.slice(0, 3);

  // Build findings from data
  const findings = [];

  // 1. Dominant kernel
  if ((top.time_pct || 0) > 30) {
    findings.push({
      icon: "🔴", title: "Single kernel dominates",
      text: `"${(top.kernel_name || "").slice(0, 45)}" consumes ${(top.time_pct || 0).toFixed(1)}% of GPU time alone. This is your primary optimization target.`,
    });
  } else if ((top.time_pct || 0) > 15) {
    findings.push({
      icon: "🟡", title: "Moderate hotspot",
      text: `Top kernel "${(top.kernel_name || "").slice(0, 45)}" at ${(top.time_pct || 0).toFixed(1)}%. Consider fusion or replacement.`,
    });
  } else {
    findings.push({
      icon: "🟢", title: "Well-distributed workload",
      text: "GPU time is spread across many kernels. Focus on fusion and reducing kernel launch overhead.",
    });
  }

  // 2. CUDA Graph
  if (cudaGraph) {
    findings.push({
      icon: "🟢", title: "CUDA Graph active",
      text: `Total GPU time: ${total.toFixed(2)}s with CUDA Graph. Kernel launch overhead is minimized.`,
    });
  } else {
    findings.push({
      icon: "🔴", title: "CUDA Graph disabled",
      text: "Enable CUDA Graph to reduce kernel launch overhead and CPU-GPU synchronization. Expected 3-5x speedup on decode.",
    });
  }

  // 3. Category concentration
  const cats = {};
  for (const k of kernels) cats[k.category] = (cats[k.category] || 0) + (k.time_pct || 0);
  const topCat = Object.entries(cats).sort((a, b) => b[1] - a[1])[0];
  if (topCat && topCat[1] > 50) {
    findings.push({
      icon: "🔴", title: `Category "${topCat[0]}" dominates at ${topCat[1].toFixed(0)}%`,
      text: topCat[0] === "Reduce" ? "TP allreduce is the bottleneck. Consider communication-computation overlap or reducing TP degree." :
           topCat[0] === "GEMM" ? "GEMM is the bottleneck. Explore quantization (FP8/INT8) or faster GEMM backends." :
           `Focus optimization efforts on ${topCat[0]} operations.`,
    });
  }

  // 4. Top 3 summary
  const top3Summary = top3.map((k, i) =>
    `#${i + 1} ${(k.category || "?").slice(0, 10)} ${(k.time_pct || 0).toFixed(1)}%`
  ).join("  |  ");
  findings.push({
    icon: "📊", title: "Top 3 kernels",
    text: top3Summary,
  });

  // 5. MFU note
  const withMfu = kernels.filter((k) => k.mfu != null && k.mfu > 0);
  if (withMfu.length === 0) {
    findings.push({
      icon: "💡", title: "No MFU data available",
      text: "Profiler was run without record_shapes=True. Enable it to get per-kernel TFLOPS and MFU analysis.",
    });
  } else if (withMfu.length < 10) {
    findings.push({
      icon: "💡", title: `MFU data available for ${withMfu.length} kernels`,
      text: "Limited TFLOPS data (only CK GEMM tiles). Enable record_shapes=True for full MFU coverage.",
    });
  }

  return html`
    <div class="sa-panel">
      <h3>Key Findings</h3>
      <div class="sa-findings">
        ${findings.map((f) => html`
          <div class="sa-finding-card">
            <span class="sa-finding-icon">${f.icon}</span>
            <div class="sa-finding-body">
              <strong>${f.title}</strong>
              <p class="sa-note">${f.text}</p>
            </div>
          </div>
        `)}
      </div>
    </div>
  `;
}

/* ── MFU Distribution Histogram ── */

function MfuDistro({ kernels, gpu }) {
  if (!kernels || !kernels.length) return null;
  const peaks = { K100: { bf16: 192 }, A100_80G: { bf16: 312 }, H100: { bf16: 989 }, B200: { bf16: 2250 } };
  const pk = (peaks[gpu] || peaks.K100).bf16;

  // Compute MFU for ALL kernels from tflops_actual / theoretical
  const mfuVals = kernels.map((k) => {
    if (k.mfu != null) return k.mfu;
    if (k.tflops_actual != null && k.tflops_actual > 0) return k.tflops_actual / pk * 100;
    return null;
  }).filter((v) => v != null);

  if (mfuVals.length === 0) return html`<div class="sa-panel"><h3>MFU Distribution</h3><p class="sa-note">No MFU data available (no Input Dims in trace).</p></div>`;

  const buckets = [0, 5, 10, 25, 50, 75, 90, 100];
  const labels = ["0-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"];
  const hist = new Array(buckets.length - 1).fill(0);
  for (const v of mfuVals) {
    for (let i = buckets.length - 1; i >= 0; i--) {
      if (v >= buckets[i]) { hist[i]++; break; }
    }
  }

  const maxN = Math.max(...hist, 1);
  const avg = mfuVals.reduce((a, b) => a + b, 0) / mfuVals.length;
  const median = mfuVals.sort((a, b) => a - b)[Math.floor(mfuVals.length / 2)];

  return html`
    <div class="sa-panel">
      <h3>MFU Distribution</h3>
      <p class="sa-note">${mfuVals.length} kernels with TFLOPS data | avg=${avg.toFixed(1)}% | median=${median.toFixed(1)}%</p>
      <div class="sa-hist">
        ${hist.map((n, i) => html`
          <div class="sa-hist-row">
            <span class="sa-hist-label">${labels[i]}</span>
            <div class="sa-hist-bar-bg">
              <div class="sa-hist-bar" style="width:${(n / maxN * 100).toFixed(0)}%"></div>
            </div>
            <span class="sa-hist-count">${n}</span>
          </div>
        `)}
      </div>
    </div>
  `;
}

/* ── Frequency Analysis ── */

function FrequencyPanel({ kernels }) {
  if (!kernels || !kernels.length) return null;
  // Top kernels by call count
  const byCount = [...kernels].sort((a, b) => (b.count || 0) - (a.count || 0));

  return html`
    <div class="sa-panel">
      <h3>Top by Invocation Count</h3>
      <p class="sa-note">High invocation count kernels may indicate repeated small operations that could be batched.</p>
      <table class="sa-table">
        <thead><tr><th>Kernel</th><th>Calls</th><th>Time%</th><th>Avg μs</th><th>Category</th></tr></thead>
        <tbody>
          ${byCount.slice(0, 10).map((k) => html`
            <tr key=${k.rank}>
              <td class="sa-kernel-name" title=${k.kernel_name}>${(k.kernel_name || "").slice(0, 45)}</td>
              <td class="sa-num">${k.count}</td>
              <td class="sa-num">${(k.time_pct || 0).toFixed(1)}%</td>
              <td class="sa-num">${(k.avg_dur_us || 0).toFixed(1)}</td>
              <td><span class="sa-cat">${k.category || "?"}</span></td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

/* ── TFLOPS & Bandwidth Panel ── */

function TflopsPanel({ kernels, gpu }) {
  if (!kernels || !kernels.length) return null;
  const peaks = { K100: { bf16: 192, bw: 700 }, A100_80G: { bf16: 312, bw: 2039 },
    H100: { bf16: 989, bw: 3350 }, B200: { bf16: 2250, bw: 8000 } };
  const pk = peaks[gpu] || peaks.K100;

  // Kernels with actual TFLOPS data
  const withData = kernels.filter((k) => k.tflops_actual != null && k.tflops_actual > 0);
  const withBw = kernels.filter((k) => k.bandwidth_gb_s != null && k.bandwidth_gb_s > 0);

  return html`
    <div class="sa-panel">
      <h3>TFLOPS & Bandwidth</h3>
      <p class="sa-note">GPU: ${gpu} | Theoretical peak BF16: ${pk.bf16} TFLOPS | BW: ${pk.bw} GB/s</p>
      <p class="sa-note">${withData.length}/${kernels.length} kernels have TFLOPS data (CK GEMM tile dims extracted from kernel names).</p>
      <table class="sa-table">
        <thead><tr><th>Kernel</th><th>TFLOPS</th><th>Peak%</th><th>BW GB/s</th><th>BW%</th><th>Bound</th></tr></thead>
        <tbody>
          ${kernels.filter(k => k.tflops_actual != null || k.bandwidth_gb_s != null).slice(0, 10).map((k) => html`
            <tr key=${k.rank}>
              <td class="sa-kernel-name" title=${k.kernel_name}>${(k.kernel_name || "").slice(0, 45)}</td>
              <td class="sa-num">${k.tflops_actual != null ? k.tflops_actual.toFixed(3) : "—"}</td>
              <td class="sa-num">${k.mfu != null ? k.mfu.toFixed(1) + "%" : "—"}</td>
              <td class="sa-num">${k.bandwidth_gb_s != null ? k.bandwidth_gb_s.toFixed(1) : "—"}</td>
              <td class="sa-num">${k.bandwidth_gb_s != null ? (k.bandwidth_gb_s / pk.bw * 100).toFixed(1) + "%" : "—"}</td>
              <td class="sa-sm">${k.bound || "—"}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

/* ── Model Structure → Operator Mapping Panel ── */

function StructureMappingPanel({ mapping, kernels }) {
  if (!mapping || !mapping.entries) return html`<div class="sa-panel"><h3>Model Structure Mapping</h3><p class="sa-note">No mapping data available.</p></div>`;

  const entries = mapping.entries || [];
  // Group by model_layer
  const layerGroups = {};
  for (const e of entries) {
    const layer = e.model_layer || "unknown";
    if (!layerGroups[layer]) layerGroups[layer] = { kernels: [], categories: {} };
    layerGroups[layer].kernels.push(e);
    layerGroups[layer].categories[e.category] = (layerGroups[layer].categories[e.category] || 0) + 1;
  }

  const layers = Object.entries(layerGroups).sort((a, b) => b[1].kernels.length - a[1].kernels.length);

  // Confidence stats
  const confStats = { high: 0, medium: 0, low: 0 };
  for (const e of entries) { confStats[e.confidence || "low"]++; }
  const total = entries.length || 1;

  return html`
    <div class="sa-panel">
      <h3>Model Structure → Operator Mapping</h3>
      <p class="sa-note">${entries.length} kernel↔layer mappings |
        <span class="sa-conf sa-conf-high">high ${confStats.high} (${(confStats.high/total*100).toFixed(0)}%)</span>
        <span class="sa-conf sa-conf-medium">med ${confStats.medium} (${(confStats.medium/total*100).toFixed(0)}%)</span>
        <span class="sa-conf sa-conf-low">low ${confStats.low} (${(confStats.low/total*100).toFixed(0)}%)</span>
      </p>
      <div class="sa-mapping-grid">
        ${layers.slice(0, 10).map(([layer, group]) => html`
          <div class="sa-mapping-row">
            <span class="sa-mapping-layer">${layer}</span>
            <span class="sa-mapping-count">${group.kernels.length} kernels</span>
            <span class="sa-mapping-cats">
              ${Object.entries(group.categories).slice(0, 4).map(([cat, n]) => html`
                <span class="sa-cat">${cat}×${n}</span>
              `)}
            </span>
          </div>
        `)}
      </div>
    </div>
  `;
}

/* ── Fuse Opportunities Panel ── */

function FusePanel({ detail }) {
  const fuse = detail ? detail.fuse : null;
  const matches = fuse ? (fuse.matches || []) : [];

  if (matches.length === 0) return html`
    <div class="sa-panel">
      <h3>Fuse Opportunities</h3>
      <p class="sa-note">No fuse pattern matches found in rule engine. Try enabling LLM hints for AI-generated suggestions.</p>
    </div>
  `;

  return html`
    <div class="sa-panel">
      <h3>Fuse Opportunities (${matches.length})</h3>
      ${matches.map((m) => html`
        <div class="sa-fuse-card">
          <div class="sa-fuse-header">
            <strong>${m.pattern}</strong>
            <span class="sa-conf sa-conf-${m.confidence}">${m.confidence}</span>
            <span class="sa-note ml8">~${m.estimated_saving_us}μs estimated saving</span>
          </div>
          <p class="sa-note">${m.suggestion}</p>
          <p class="sa-note" style="font-family:monospace;font-size:10px">Kernels: ${(m.kernels || []).join(" → ")}</p>
        </div>
      `)}
    </div>
  `;
}
