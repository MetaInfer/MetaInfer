"""Fine-grained implementation variants for agent reference.

Variants are accepted kernels stored as a directory tree that mirrors the
form taxonomy:

    variant/<operator+dtype>/<model>/<TP>/<M>/<operator>.hip

e.g. ``variant/int8w8a8-gemm/hy3/TP4/M4096/o_proj.hip``.

Each leaf file carries a machine-parseable header:

    // @@variant shape=<shape_id> commit=<git sha> added=<YYYY-MM-DD>
    //   median_us=.. p90_us=.. tops=.. bandwidth_gb_s=.. speedup=.. baseline_us=..
    //   source=<task id>
    <kernel source>

Semantics (kept in sync with prompts.py): variants are evidence, not policy —
agents may read and adapt the file matching their (operator/dtype, model, TP,
M, operator) family, but must not be locked into it, and any reuse requires
full re-validation. The tree is staged read-only into generated kernel
repositories under ``references/variants/`` and is not part of the build.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HEADER_RE = re.compile(r"(?m)^// @@variant shape=(\S+)(.*?)\n(?:.*?\n)*?// @@end")
# Captures the whole header block (first line tail + ``//   key=value`` lines)
# up to ``// @@end`` for structured parsing of metrics/source/commit.
_HEADER_BLOCK_RE = re.compile(
    r"^// @@variant shape=(\S+)(.*?)\n// @@end",
    re.DOTALL | re.MULTILINE,
)

MODEL_SLUGS = {
    "deepseek v4 flash": "deepseek-v4",
    "deepseek-v4": "deepseek-v4",
    "hy3 (hunyuan 3)": "hy3",
    "hy3": "hy3",
    "minimax m3": "minimax-m3",
    "minimax-m3": "minimax-m3",
    "glm5.2": "glm52",
    "glm52": "glm52",
}

OPERATOR_TYPE_SLUGS = {
    "quantized gemm": "gemm",
    "attention": "attention",
    "rmsnorm / layernorm": "rmsnorm",
    "rope": "rope",
    "custom operator": "custom",
}


def _slug(value: str) -> str:
    # "INT8 W8A8" -> "int8w8a8", "Quantized GEMM" -> "gemm"
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def operator_type_slug(operator: str) -> str:
    key = operator.strip().lower()
    return OPERATOR_TYPE_SLUGS.get(key, re.sub(r"[^a-z0-9]+", "-", key).strip("-") or "custom")


def dtype_slug(dtype: str) -> str:
    return _slug(dtype) or "other"


def model_slug(label: str) -> str:
    key = label.strip().lower()
    return MODEL_SLUGS.get(key, re.sub(r"[^a-z0-9]+", "-", key).strip("-") or "unknown-model")


def parse_shape_meta(shape_id: str) -> Dict[str, Any]:
    """Split a shape id like ``hy3_tp4_o_proj_m4096`` into tp/operator/m."""
    tp_match = re.search(r"(?:^|_)(tp\d+)_", shape_id)
    tp = int(tp_match.group(1)[2:]) if tp_match else None
    m_match = re.search(r"_m(\d+)$", shape_id)
    m = int(m_match.group(1)) if m_match else None
    operator = shape_id
    if m_match:
        operator = operator[: m_match.start()]
    if tp_match:
        operator = operator[tp_match.end():]
    operator = operator.strip("_") or shape_id
    return {"tp": tp, "operator": operator, "m": m}


def derive_variant_meta(answers: Dict[str, Any], shape_id: str) -> Dict[str, Any]:
    """Derive the fine-grained taxonomy for one shape from the form answers."""
    parsed = parse_shape_meta(shape_id)
    operator = str(answers.get("operator") or "")
    dtype = str(answers.get("dtype") or "")
    return {
        "shape": shape_id,
        "operator": operator,
        "dtype": dtype,
        "family": f"{dtype_slug(dtype)}-{operator_type_slug(operator)}",
        "model": model_slug(str(answers.get("model") or "")),
        "tp": parsed["tp"],
        "operator_name": parsed["operator"],
        "m": parsed["m"],
    }


def variant_root() -> Path:
    return Path(__file__).resolve().parents[1] / "variant"


def variant_path(meta: Dict[str, Any]) -> Path:
    """The leaf file path for one specific operator's variant."""
    tp = f"TP{meta['tp']}" if meta.get("tp") is not None else "TP?"
    m = f"M{meta['m']}" if meta.get("m") is not None else "M?"
    return (
        variant_root() / meta["family"] / meta["model"] / tp / m
        / f"{meta['operator_name']}.hip"
    )


def section_header(
    meta: Dict[str, Any],
    *,
    commit: str,
    metrics: Dict[str, Any],
    source_task: str = "",
) -> str:
    lines = [
        f"// @@variant shape={meta['shape']} commit={commit or '?'} "
        f"added={time.strftime('%Y-%m-%d')}",
    ]
    metric_parts = []
    for key, label in (
        ("median_us", "median_us"),
        ("p90_us", "p90_us"),
        ("logical_tops", "tops"),
        ("algorithmic_bandwidth_gb_s", "bandwidth_gb_s"),
        ("speedup", "speedup"),
        ("baseline_us", "baseline_us"),
    ):
        value = metrics.get(key)
        if value is not None:
            metric_parts.append(
                f"{label}={value:.4g}" if isinstance(value, float) else f"{label}={value}"
            )
    if metric_parts:
        lines.append("//   " + " ".join(metric_parts))
    if source_task:
        lines.append(f"//   source={source_task}")
    lines.append("")
    return "\n".join(lines)


def add_variant(
    *,
    meta: Dict[str, Any],
    kernel_source: str,
    commit: str,
    metrics: Dict[str, Any],
    source_task: str = "",
    backup: bool = True,
    reject_slower_than_existing: bool = False,
) -> Dict[str, Any]:
    """Write one specific operator's accepted kernel into the variant tree.

    Replaces an existing file for the same leaf (backing it up first). When
    ``reject_slower_than_existing`` is set and both the existing file's header
    and ``metrics`` carry a ``median_us``, a candidate that is strictly slower
    than the existing variant is rejected with ``ValueError`` (equal or faster
    may replace).
    """
    target = variant_path(meta)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and reject_slower_than_existing:
        _reject_slower_replacement(target, metrics)
    header = section_header(
        meta, commit=commit, metrics=metrics, source_task=source_task
    )
    content = header + kernel_source.rstrip() + "\n// @@end\n"
    action = "updated" if target.exists() else "added"
    backup_path = None
    if target.exists() and backup:
        backup_path = target.with_name(f"{target.name}.bak-{int(time.time() * 1000)}")
        shutil.copy2(target, backup_path)
    target.write_text(content, encoding="utf-8")
    return {
        "action": action,
        "shape": meta["shape"],
        "path": str(target),
        "backup": str(backup_path) if backup_path else None,
        "meta": meta,
    }


def _reject_slower_replacement(target: Path, metrics: Dict[str, Any]) -> None:
    """Raise ValueError when the candidate is strictly slower than the
    existing variant's recorded median (both must be known)."""
    existing = _parse_variant_header(target.read_text(encoding="utf-8", errors="replace"))
    existing_median = existing.get("median_us")
    new_median = metrics.get("median_us")
    if existing_median is None or new_median is None:
        # Cannot compare -> allow (guard is best-effort on known medians).
        return
    if float(new_median) > float(existing_median):
        raise ValueError(
            "rejecting slower variant replacement: existing median_us="
            f"{float(existing_median):.4g} vs candidate median_us="
            f"{float(new_median):.4g} "
            "(only an equal-or-faster candidate may replace)"
        )


def list_variant_index() -> List[Dict[str, Any]]:
    """Walk the variant tree and return the section index (path-derived
    taxonomy + header fields)."""
    out: List[Dict[str, Any]] = []
    root = variant_root()
    if not root.exists():
        return out
    # skip the legacy single-file variant (kept for historical reference)
    for file in sorted(root.rglob("*.hip")):
        if file.name.startswith("w8a8_gemm_variants"):
            continue
        text = file.read_text(encoding="utf-8", errors="replace")
        header = _parse_variant_header(text)
        shape = header.get("shape") or file.stem
        rel = file.relative_to(root).parts
        out.append({
            "path": str(file.relative_to(root)),
            "shape": shape,
            "family": rel[0] if len(rel) > 0 else "",
            "model": rel[1] if len(rel) > 1 else "",
            "tp": rel[2] if len(rel) > 2 else "",
            "m": rel[3] if len(rel) > 3 else "",
            "operator": file.stem,
            "commit": header.get("commit", ""),
            "added": header.get("added", ""),
            "source": header.get("source", ""),
            "median_us": header.get("median_us"),
            "p90_us": header.get("p90_us"),
            "speedup": header.get("speedup"),
            "baseline_us": header.get("baseline_us"),
        })
    return out


# Header fields that carry machine-readable performance evidence. Only these
# are surfaced in the variant index (and the WebUI comparison dialog); the
# rest of the header/body is opaque to the index.
_HEADER_NUMERIC_FIELDS = frozenset({
    "median_us", "p90_us", "tops", "bandwidth_gb_s", "speedup", "baseline_us",
})
_HEADER_TEXT_FIELDS = frozenset({"commit", "added", "source"})


def _parse_variant_header(text: str) -> Dict[str, Any]:
    """Parse the machine-readable variant header of one kernel file.

    Returns ``{"shape": ..., "commit": ..., "median_us": ..., ...}`` with
    numeric metric fields converted to float; unknown ``key=value`` tokens
    inside the header block are ignored.
    """
    block = _HEADER_BLOCK_RE.search(text)
    if block is None:
        return {}
    fields: Dict[str, Any] = {"shape": block.group(1)}
    for key, value in re.findall(r"(\w+)=(\S+)", block.group(2)):
        if key in _HEADER_NUMERIC_FIELDS:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
        elif key in _HEADER_TEXT_FIELDS:
            fields[key] = value
    return fields
