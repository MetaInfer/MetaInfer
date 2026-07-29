// Orchestrator log viewer — fetches and displays the orchestrator log file.
// Polls every 5s when visible, matching the old KE behavior.

import { html } from "htm/preact";
import { useCallback, useEffect, useState, useRef } from "preact/hooks";
import { getLog } from "app/ke-runtime-api";

export function LogViewer({ taskId, visible }) {
  const [log, setLog] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const ref = useRef(null);

  const refresh = useCallback(async () => {
    if (!taskId) return;
    setErr(null);
    try {
      const d = await getLog(taskId);
      setLog(d.text || d.log || "");
      setLoading(false);
    } catch (e) {
      setErr(String(e));
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    if (!visible || !taskId) {
      setLog("");
      return;
    }
    setLoading(true);
    refresh();
  }, [taskId, visible, refresh]);

  // Poll every 5s when visible
  useEffect(() => {
    if (!visible || !taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, visible, refresh]);

  useEffect(() => {
    if (ref.current && visible) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [log, visible]);

  if (!visible) return null;

  return html`
    <div class="log-viewer">
      <div class="log-viewer-header">
        <strong>Orchestrator log</strong>
        ${loading ? html`<span class="muted">loading…</span>` : null}
        <button class="btn ghost" style="font-size:10px;margin-left:auto"
          onClick=${() => { setLoading(true); refresh(); }}>
          Refresh
        </button>
      </div>
      <pre ref=${ref} class="log-content">${log || err || "(empty)"}</pre>
    </div>
  `;
}
