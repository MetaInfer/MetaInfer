// Agent framework → model selection for dcu-kernel-auto-opt.
//
// Self-registers one override component ("agent-model") via
// globalThis.__metainferOverrides. The agent_framework field is a plain
// select; this component reads its current value from allValues and shows
// only the models valid for that framework (ccb: Sonnet/Opus; dsh:
// deepseek-v4-flash), auto-correcting the value when the framework changes.
//
// The label→model mapping must stay in sync with
// orchestrator/config.py (AGENT_FRAMEWORKS).

import { html } from "htm/preact";
import { useEffect } from "preact/hooks";

// Framework label → allowed model labels (mirror of config.py).
const FRAMEWORK_MODELS = {
  ccb: ["Opus", "Sonnet"],
  dsh: ["deepseek-v4-flash"],
};
const DEFAULT_MODEL = {
  ccb: "Opus",
  dsh: "deepseek-v4-flash",
};

function AgentModelField({ field, value, onChange, allValues }) {
  const framework = String(
    (allValues && allValues.agent_framework) || "ccb"
  ).toLowerCase();
  const models = FRAMEWORK_MODELS[framework] || FRAMEWORK_MODELS.ccb;

  // Keep the submitted model valid for the active framework: whenever the
  // framework changes (or the form first loads) and the current value is not
  // an option of that framework, snap it to the framework default.
  useEffect(() => {
    if (!models.includes(value)) {
      onChange(DEFAULT_MODEL[framework] || models[0]);
    }
    // onChange is stable (setField wrapper); only framework changes matter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [framework]);

  return html`
    <select class="input" value=${value || ""}
      onChange=${(e) => onChange(e.target.value)}>
      ${models.map(
        (label) => html`
          <option value=${label} selected=${label === value}>${label}</option>
        `
      )}
    </select>
  `;
}

// ---- register ---------------------------------------------------------------

var _g = (typeof globalThis !== "undefined" ? globalThis : window);
var _bridge = (_g.__metainferOverrides = _g.__metainferOverrides || {});
_bridge["agent-model"] = AgentModelField;

// Named export for the form-renderer to pick up.
export var AgentModelFieldComponent = AgentModelField;
