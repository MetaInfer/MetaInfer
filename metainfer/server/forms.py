"""Task type definitions → frontend form schema.

Each task type ships a ``form.yaml`` inside its self-contained task package
at ``metainfer/tasks/<task_pkg>/form.yaml`` (see CLAUDE.md for the layout).
The task-type's friendly label + description live on its WebPlugin
(``metainfer/tasks/<task_pkg>/server/plugin.py``); this module
reads them from the registry, so adding a new task type does NOT require
editing any central metadata table.

Schema shape (compatible with the legacy questions.yaml format):

    - key: target_model           # unique field key
      question: "Enter the model weight path:"
      header: "Target model"       # short label, <= 12 chars
      required: true
      multi: false                 # omit for free-form text
      options:                     # omit for free-form text
        - label: "..."
          description: "..."
      default: "..."
      # NEW: explicit form widget hint. If omitted, the type is inferred
      # from multi / options.
      form: text|textarea|select|multiselect|file|number

This module normalizes the YAML into a stable JSON schema the frontend
form renderer consumes:

    {
      "type": "<task-type-id>",
      "label": "<plugin label>",
      "description": "...",
      "fields": [
        {
          "key": "target_model",
          "label": "Target model",
          "help": "Enter the model weight path: ...",
          "type": "file",            # canonical widget type
          "required": true,
          "default": null,
          "options": null | [{"label":..., "description":...}, ...],
          "override_component": null | "file-picker" | "shape-input" | ...
        }
      ]
    }

The renderer walks ``fields`` generically. A non-null ``override_component``
tells it to delegate to a task-specific widget instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

def _infer_field_type(entry: Dict[str, Any]) -> str:
    """Canonical widget type. Explicit ``form`` wins; otherwise infer
    from multi/options presence."""
    explicit = entry.get("form")
    if explicit:
        return explicit
    multi = bool(entry.get("multi"))
    has_options = bool(entry.get("options"))
    if multi:
        return "multiselect"
    if has_options:
        return "select"
    return "text"


def _normalize_field(entry: Dict[str, Any]) -> Dict[str, Any]:
    key = entry.get("key")
    if not key:
        raise ValueError(f"field missing 'key': {entry!r}")
    ftype = _infer_field_type(entry)
    out: Dict[str, Any] = {
        "key": key,
        "label": entry.get("header") or key,
        "help": entry.get("question") or "",
        "type": ftype,
        "required": bool(entry.get("required", False)),
        "default": entry.get("default"),
        "options": None,
        # Override hook for task-specific widgets. Read from the YAML so
        # task authors can opt a field into a custom renderer without
        # code changes to the generic form.
        "override_component": entry.get("override_component"),
    }
    opts = entry.get("options")
    if opts:
        out["options"] = [
            {"label": o.get("label", ""), "description": o.get("description", "")}
            for o in opts
        ]
    return out


def _form_yaml_for_task_type(task_type: str) -> Optional[Path]:
    """Resolve the form.yaml path for a task type.

    Looks for ``form.yaml`` at the registered WebPlugin's task-package
    root (parent of ``frontend_dir``). This is the only location we
    support now — every task type must own a full plugin package.

    Returns ``None`` if no plugin is registered for this task type, or
    the plugin's package doesn't ship a form.yaml.
    """
    from .registry import get as _get_plugin
    plugin = _get_plugin(task_type)
    if plugin is None or plugin.frontend_dir is None:
        return None
    cand = plugin.frontend_dir.parent / "form.yaml"
    if cand.exists():
        return cand
    return None


def load_form_schema(task_type: str) -> Optional[Dict[str, Any]]:
    """Return the normalized form schema for ``task_type``, or None if
    no form.yaml exists for it."""
    yaml_path = _form_yaml_for_task_type(task_type)
    if yaml_path is None:
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    if not isinstance(raw, list):
        raise ValueError(f"{yaml_path}: expected a list of field entries")
    fields = [_normalize_field(e) for e in raw]
    # Label/description come from the WebPlugin (single source of truth
    # per task type), NOT from a central dict here.
    from .registry import get as _get_plugin
    plugin = _get_plugin(task_type)
    label = (plugin.label if plugin else "") or task_type
    description = (plugin.description if plugin else "") or ""
    return {
        "type": task_type,
        "label": label,
        "description": description,
        "fields": fields,
    }


def list_task_types() -> List[Dict[str, str]]:
    """Return the compact list of available task types for the New Task
    type picker.

    One entry per registered WebPlugin whose task-package ships a
    ``form.yaml``. The label + description come from the plugin itself.
    """
    out: List[Dict[str, str]] = []
    from .registry import all_plugins as _all_plugins
    for plugin in _all_plugins():
        if plugin.frontend_dir is None:
            continue
        cand = plugin.frontend_dir.parent / "form.yaml"
        if not cand.exists():
            continue
        out.append({
            "id": plugin.type,
            "label": plugin.label or plugin.type,
            "description": plugin.description or "",
        })
    return out


def validate_submission(task_type: str, answers: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a form submission against the schema. Returns a dict of
    ``{ok: bool, errors: {field_key: message}}``."""
    schema = load_form_schema(task_type)
    if schema is None:
        return {"ok": False, "errors": {"_": f"unknown task type {task_type!r}"}}
    errors: Dict[str, str] = {}
    for field in schema["fields"]:
        key = field["key"]
        val = answers.get(key)
        if field["required"] and (
            val is None
            or (isinstance(val, str) and val.strip() == "")
            or (isinstance(val, list) and len(val) == 0)
        ):
            errors[key] = "this field is required"
            continue
        if val is None:
            continue
        # Type-check against declared widget type
        t = field["type"]
        if t in ("select",) and field["options"]:
            valid_labels = {o["label"] for o in field["options"]}
            if val not in valid_labels:
                errors[key] = f"must be one of: {sorted(valid_labels)}"
        elif t == "multiselect" and field["options"]:
            valid_labels = {o["label"] for o in field["options"]}
            if not isinstance(val, list):
                errors[key] = "expected a list"
            else:
                bad = [v for v in val if v not in valid_labels]
                if bad:
                    errors[key] = f"unknown options: {bad}"
    return {"ok": not errors, "errors": errors}
