"""Canonical user-visible capabilities for the native C++ task.

The form is YAML, so it cannot import this module directly. Tests compare the
form options with this matrix to prevent a feature from being displayed
without knowledge routing and immutable validation ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class CapabilitySpec:
    label: str
    notebooks: Tuple[str, ...]
    oracle_owners: Tuple[str, ...]


CAPABILITIES: Dict[str, CapabilitySpec] = {
    "paged kv cache": CapabilitySpec(
        label="Paged KV cache",
        notebooks=(
            "00_contracts/attention_kv_contracts.md",
            "01_framework_design/03_kv_cache.md",
            "03_operators/01_attention_ops.md",
            "07_improvementPlan/03_paged_kv_and_prefix_cache.md",
        ),
        oracle_owners=(
            "C:long-context-boundary",
            "E:memory-and-concurrency",
        ),
    ),
    "continuous batching": CapabilitySpec(
        label="Continuous batching",
        notebooks=(
            "01_framework_design/07_request_lifecycle.md",
            "01_framework_design/02_scheduler.md",
            "07_improvementPlan/02_continuous_batching.md",
        ),
        oracle_owners=(
            "C:concurrent-deterministic-isolation",
            "E:concurrency-sweep",
        ),
    ),
    "tensor parallelism": CapabilitySpec(
        label="Tensor parallelism",
        notebooks=(
            "00_contracts/tp_communication_contracts.md",
            "04_parallel_strategies/01_tensor_parallel.md",
            "04_parallel_strategies/02_qwen_dense_tp.md",
            "04_parallel_strategies/04_rccl_collectives.md",
            "07_improvementPlan/05_tensor_parallel_rccl.md",
        ),
        oracle_owners=(
            "C:native-rank-device-evidence",
            "E:tp-rank-activity",
        ),
    ),
    "streaming responses": CapabilitySpec(
        label="Streaming responses",
        notebooks=(
            "00_contracts/native_service_contracts.md",
            "05_inference_service/02_openai_http_api.md",
            "05_inference_service/03_sse_streaming.md",
            "05_inference_service/04_process_lifecycle.md",
            "07_improvementPlan/07_native_service_streaming.md",
        ),
        oracle_owners=("C:sse-streaming",),
    ),
}


def normalize_features(value: Any) -> Tuple[str, ...]:
    if not value:
        return ()
    raw: Iterable[Any] = [value] if isinstance(value, str) else value
    return tuple(str(item).strip().casefold() for item in raw if str(item).strip())


def unsupported_features(value: Any) -> Tuple[str, ...]:
    return tuple(
        feature for feature in normalize_features(value)
        if feature not in CAPABILITIES
    )


def request_validation_errors(req: Dict[str, Any]) -> Tuple[str, ...]:
    errors = [
        "unsupported C++ feature: " + feature
        for feature in unsupported_features(req.get("features"))
    ]
    features = set(normalize_features(req.get("features")))
    raw_tp = str(req.get("tensor_parallel_size") or "1").strip()
    if raw_tp.isdigit() and int(raw_tp) > 1 and "tensor parallelism" not in features:
        errors.append(
            "tensor_parallel_size > 1 requires the Tensor parallelism feature"
        )
    if "tensor parallelism" in features and raw_tp == "1":
        errors.append(
            "Tensor parallelism requires tensor_parallel_size Auto or greater than 1"
        )
    return tuple(errors)


def visible_feature_labels() -> Tuple[str, ...]:
    return tuple(spec.label for spec in CAPABILITIES.values())


__all__ = [
    "CAPABILITIES",
    "CapabilitySpec",
    "normalize_features",
    "request_validation_errors",
    "unsupported_features",
    "visible_feature_labels",
]
