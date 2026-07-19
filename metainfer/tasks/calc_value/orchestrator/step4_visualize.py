"""Step 4: generate an interactive HTML visualization.

One agent reads graph.json + the calc/ dir layout and produces
``viz.html`` — a self-contained page with:
* SVG graph of nodes
* batch_size + seq_len input controls
* JS that fetches ``/api/calc-theoretical-value/<id>/calc/compute?...`` and updates the
  per-node TFLOPs/GB badges
* totals bar

If the agent fails or the HTML is unusable, fall back to a deterministic
HTML template generated from graph.json (less pretty but always works).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from metainfer.orchestrator.subagent_manager import AgentSpec
from . import deterministic as det
from . import prompts as P


PER_AGENT_TIMEOUT_S = 1800


def _format_env_block(env_vars: str) -> str:
    if not env_vars:
        return "(none)"
    return "\n".join(f"  {ln}" for ln in env_vars.splitlines() if ln.strip())


def _write_prompt(workdir: Path, name: str, text: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / f"{name}.prompt.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _generate_fallback_html(graph: Dict[str, Any], task_id: str) -> str:
    """Minimal HTML scaffold if the agent fails. Always works.

    Sectioned-aware: renders one card per section (input / N layer
    templates / output), with a ×N badge for ``layer_template``
    sections. The /compute endpoint returns per-instance numbers;
    this view multiplies by ``repeat_count`` for display.
    """
    norm = det.normalize_graph(graph)
    sections_js = json.dumps([
        {
            "id": sec.get("id", "?"),
            "kind": sec.get("kind", "other"),
            "description": sec.get("description", ""),
            "repeat_count": (
                sec.get("repeat_count") if sec.get("kind") == "layer_template"
                else 1
            ),
            "nodes": [
                {"id": n.get("id", "?"),
                 "compound_id": f'{sec.get("id", "?")}__{n.get("id", "?")}',
                 "op": n.get("op", "?"),
                 "purpose": n.get("purpose", ""),
                 "category": _category(n.get("op", ""))}
                for n in (sec.get("graph") or {}).get("nodes") or []
                if isinstance(n, dict)
            ],
        }
        for sec in norm.get("sections") or []
    ])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MetaInfer calc-value — {task_id}</title>
<style>
  body {{ background: #0e1117; color: #e6edf3; font-family: -apple-system, sans-serif; font-size: 13px; margin: 0; padding: 16px; }}
  h1 {{ font-size: 16px; margin: 0 0 12px; }}
  .controls {{ background: #161b22; padding: 12px; border-radius: 6px; margin-bottom: 16px; display: flex; gap: 12px; align-items: center; }}
  label {{ color: #8b949e; }}
  input {{ background: #0d1117; color: #e6edf3; border: 1px solid #30363d; padding: 4px 8px; border-radius: 4px; width: 100px; }}
  button {{ background: #238636; color: white; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-weight: 600; }}
  button:hover {{ background: #2ea043; }}
  .totals {{ background: #161b22; padding: 12px; border-radius: 6px; margin-top: 16px; display: flex; gap: 24px; }}
  .totals div span {{ font-family: monospace; font-size: 18px; font-weight: 600; color: #58a6ff; margin-left: 6px; }}
  .section {{ background: #161b22; padding: 12px; border-radius: 6px; margin-bottom: 12px; }}
  .section-head {{ display: flex; gap: 8px; align-items: baseline; margin-bottom: 8px; }}
  .section-title {{ font-size: 14px; font-weight: 600; color: #e6edf3; }}
  .section-kind {{ font-size: 10px; text-transform: uppercase; color: #8b949e; padding: 2px 6px; background: #21262d; border-radius: 3px; }}
  .section-repeat {{ font-size: 11px; color: #bc8cff; font-family: monospace; padding: 2px 6px; background: #8957e540; border-radius: 3px; }}
  .section-subtotal {{ margin-left: auto; font-family: monospace; font-size: 12px; color: #58a6ff; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; }}
  th {{ color: #8b949e; font-weight: 600; font-size: 10px; text-transform: uppercase; }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-family: monospace; }}
  .cat-attention {{ background: #da363340; color: #ff7b72; }}
  .cat-mlp {{ background: #bf870040; color: #d29922; }}
  .cat-norm {{ background: #23863640; color: #3fb950; }}
  .cat-quant {{ background: #1f6feb40; color: #58a6ff; }}
  .cat-embedding {{ background: #8957e540; color: #bc8cff; }}
  .cat-other {{ background: #30363d80; color: #8b949e; }}
  .approx {{ color: #d29922; font-style: italic; margin-left: 6px; font-size: 10px; }}
  #error {{ color: #ff7b72; margin-top: 8px; }}
</style>
</head>
<body>
<h1>MetaInfer calc-value — task {task_id}</h1>
<div class="controls">
  <label>batch_size: <input type="number" id="batch_size" value="1" min="1"></label>
  <label>seq_len: <input type="number" id="seq_len" value="1" min="1"></label>
  <button onclick="recompute()">Recalculate</button>
  <span id="error"></span>
</div>
<div id="sections"></div>
<div class="totals">
  <div>Total TFLOPs:<span id="total_tflops">—</span></div>
  <div>Total GB:<span id="total_gb">—</span></div>
  <div>Arithmetic Intensity:<span id="total_ai">—</span></div>
</div>
<script>
const SECTIONS = {sections_js};
const TASK_ID = "{task_id}";

let API_BASE = "";
window.addEventListener("message", (ev) => {{
  if (ev.data && ev.data.metainfer_api_base) {{
    API_BASE = ev.data.metainfer_api_base;
  }}
}});

function categoryClass(cat) {{ return "cat-" + cat; }}

function fmt(n) {{ return (n || 0).toFixed(4); }}

function renderEmpty() {{
  document.getElementById("sections").innerHTML = SECTIONS.map(sec => `
    <div class="section" data-section="${{sec.id}}">
      <div class="section-head">
        <span class="section-title">${{sec.id}}</span>
        <span class="section-kind">${{sec.kind}}</span>
        ${{sec.repeat_count > 1 ? `<span class="section-repeat">× ${{sec.repeat_count}}</span>` : ""}}
        <span class="section-subtotal" data-subtotal="${{sec.id}}">—</span>
      </div>
      <table>
        <thead><tr><th>Node</th><th>Op</th><th>Category</th><th>Purpose</th>
                <th>pre.tflops/inst</th><th>pre.gb/inst</th>
                <th>dec.tflops/inst</th><th>dec.gb/inst</th>
                <th>×N pre.tf</th></tr></thead>
        <tbody>
          ${{sec.nodes.map(n => `
            <tr>
              <td><code>${{n.id}}</code></td>
              <td>${{n.op}}</td>
              <td><span class="badge ${{categoryClass(n.category)}}">${{n.category}}</span></td>
              <td>${{n.purpose}}</td>
              <td data-node="${{n.compound_id}}" data-field="pre_tflops">—</td>
              <td data-node="${{n.compound_id}}" data-field="pre_gb">—</td>
              <td data-node="${{n.compound_id}}" data-field="dec_tflops">—</td>
              <td data-node="${{n.compound_id}}" data-field="dec_gb">—</td>
              <td data-node="${{n.compound_id}}" data-field="pre_tflops_total">—</td>
            </tr>
          `).join("")}}
        </tbody>
      </table>
    </div>
  `).join("");
}}

async function recompute() {{
  const b = parseInt(document.getElementById("batch_size").value, 10);
  const s = parseInt(document.getElementById("seq_len").value, 10);
  if (!b || !s) return;
  const err = document.getElementById("error");
  err.textContent = "";
  const url = `${{API_BASE}}/api/calc-theoretical-value/${{TASK_ID}}/calc/compute?batch_size=${{b}}&seq_len=${{s}}`;
  try {{
    const r = await fetch(url, {{ cache: "no-store" }});
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    const approx = new Set(data.approximate_compounds || []);
    let grandPreT = 0, grandPreG = 0;
    let grandDecT = 0, grandDecG = 0;
    for (const sec of SECTIONS) {{
      let secPreT = 0, secPreG = 0, secDecT = 0, secDecG = 0;
      for (const n of sec.nodes) {{
        const v = (data.per_compound || {{}})[n.compound_id] ||
                  (data.per_node || {{}})[n.id] ||
                  {{prefill: {{tflops: 0, access_gb: 0}},
                    decode:  {{tflops: 0, access_gb: 0}}}};
        // Backward-compat: legacy per_compound has top-level tflops/access_gb.
        const pre = v.prefill || {{tflops: v.tflops || 0, access_gb: v.access_gb || 0}};
        const dec = v.decode  || {{tflops: 0, access_gb: 0}};
        const preT = pre.tflops || 0, preG = pre.access_gb || 0;
        const decT = dec.tflops || 0, decG = dec.access_gb || 0;
        const rc = sec.repeat_count;
        secPreT += preT * rc; secPreG += preG * rc;
        secDecT += decT * rc; secDecG += decG * rc;
        const ap = approx.has(n.compound_id) ? '<span class="approx">approx</span>' : "";
        document.querySelectorAll(`td[data-node="${{n.compound_id}}"]`).forEach(cell => {{
          const f = cell.getAttribute("data-field");
          if (f === "pre_tflops") cell.textContent = fmt(preT) + (ap ? " " + ap : "");
          if (f === "pre_gb") cell.textContent = fmt(preG);
          if (f === "dec_tflops") cell.textContent = fmt(decT);
          if (f === "dec_gb") cell.textContent = fmt(decG);
          if (f === "pre_tflops_total") cell.textContent = fmt(preT * rc);
        }});
      }}
      grandPreT += secPreT; grandPreG += secPreG;
      grandDecT += secDecT; grandDecG += secDecG;
      const sub = document.querySelector(`span[data-subtotal="${{sec.id}}"]`);
      if (sub) {{
        sub.textContent = `pre: ${{fmt(secPreT)}} TF / ${{fmt(secPreG)}} GB · `
                        + `dec: ${{fmt(secDecT)}} TF / ${{fmt(secDecG)}} GB`;
      }}
    }}
    const ai = (t, g) => g > 0 ? (t/g).toFixed(3) : "—";
    document.getElementById("total_tflops").textContent =
      `pre ${{fmt(grandPreT)}} · dec ${{fmt(grandDecT)}}`;
    document.getElementById("total_gb").textContent =
      `pre ${{fmt(grandPreG)}} · dec ${{fmt(grandDecG)}}`;
    document.getElementById("total_ai").textContent =
      `pre ${{ai(grandPreT, grandPreG)}} · dec ${{ai(grandDecT, grandDecG)}}`;
  }} catch (e) {{
    err.textContent = "compute failed: " + e.message;
  }}
}}

renderEmpty();
recompute();
</script>
</body>
</html>
"""


def _category(op: str) -> str:
    """Map an op name to a display category. Used by fallback HTML."""
    s = (op or "").lower()
    if "atten" in s or "sdpa" in s or "softmax" in s or "qkv" in s:
        return "attention"
    if "mlp" in s or "matmul" in s or "gelu" in s or "swiglu" in s or "ffn" in s:
        return "mlp"
    if "norm" in s:
        return "norm"
    if "quant" in s or "dequant" in s or "gptq" in s or "awq" in s:
        return "quant"
    if "embed" in s:
        return "embedding"
    return "other"


def run_step4_visualize(
    *,
    req: Dict[str, Any],
    store,
    manager,
    paths: Dict[str, Path],
    graph_path: Path,
    calc_dir: Path,
) -> Path:
    """Generate viz.html via 1 agent, with a deterministic fallback."""
    step4_dir = paths["step4_dir"]
    task_id = req.get("task_id", "task")
    store.append_timeline("calc_value.s4.start", {})

    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    # The compute URL is relative because the iframe is served from the
    # same origin as the WebUI; the viz JS uses TASK_ID to build the
    # absolute path.
    common = {
        "graph_json": json.dumps(graph, indent=2, ensure_ascii=False),
        "calc_dir": str(calc_dir),
        "task_id": task_id,
        "compute_url": f"/api/calc-theoretical-value/{task_id}/calc/compute",
    }

    name = "viz_builder"
    workdir = step4_dir / name
    log_dir = step4_dir / "logs" / name
    text = P.STEP3_VIZ_BUILDER_PROMPT.format(**common)
    prompt_file = _write_prompt(workdir, name, text)
    spec = AgentSpec(
        name=name, role="viz_builder",
        prompt_file=prompt_file, workdir=workdir, log_dir=log_dir,
        timeout_s=PER_AGENT_TIMEOUT_S, stuck_timeout_s=600, max_retries=2,
    )

    t0 = time.time()
    manager.launch(spec)
    result = manager.result(spec.name)
    elapsed = time.time() - t0

    final_path = step4_dir / "viz.html"
    used_fallback = False

    if result is None or not result.success or not (result.final_text or "").strip():
        store.append_timeline(
            "calc_value.s4.fallback",
            {"reason": "agent failed", "elapsed_s": round(elapsed, 1)},
        )
        used_fallback = True
    else:
        text_out = result.final_text or ""
        (workdir / "response.txt").write_text(text_out, encoding="utf-8")
        # File-first: prefer viz.html the agent Wrote directly; only
        # fall back to scraping response.txt if the file is missing.
        html, source = det.load_agent_text_file(
            workdir, "viz.html", text_out, _extract_html,
        )
        if not html:
            store.append_timeline(
                "calc_value.s4.fallback",
                {"reason": "no HTML found in viz.html or response.txt",
                 "elapsed_s": round(elapsed, 1)},
            )
            used_fallback = True
        else:
            final_path.write_text(html, encoding="utf-8")
            store.append_timeline(
                "calc_value.s4.done",
                {"agent_html_chars": len(html), "source": source,
                 "elapsed_s": round(elapsed, 1)},
            )

    if used_fallback:
        html = _generate_fallback_html(graph, task_id)
        final_path.write_text(html, encoding="utf-8")
        store.append_timeline(
            "calc_value.s4.fallback_used",
            {"fallback_html_chars": len(html)},
        )
        print(f"[calc-value.S4] agent HTML generation failed; "
              f"used deterministic fallback.", flush=True)

    return final_path


def _extract_html(text: str) -> str:
    """Pull the HTML document out of an LLM response."""
    if not text:
        return ""
    # Fenced HTML first.
    import re
    m = re.search(r"```(?:html|HTML)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m and "<html" in m.group(1).lower():
        return m.group(1)
    # Bare <!doctype html> ... </html>
    m = re.search(r"(<!doctype html>.*?</html>)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # Bare <html> ... </html>
    m = re.search(r"(<html.*?</html>)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    return ""
