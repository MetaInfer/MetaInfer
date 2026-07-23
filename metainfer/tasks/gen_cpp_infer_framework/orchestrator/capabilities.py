"""Compile task-form answers into a frozen capability contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from metainfer.orchestrator.requirements import req_field


_REGISTRY_FILE = Path(__file__).resolve().parent.parent / "capabilities.yaml"
_RESOLVED_FILE = "resolved_requirements.json"
_MATURITIES = {"supported", "experimental", "unavailable"}


class CapabilityResolutionError(ValueError):
    """Raised when a task requests an invalid or unsupported capability set."""

    def __init__(self, message: str, *, field: str = "features") -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class CapabilityContract:
    id: str
    label: str
    selection: str
    maturity: str
    dependencies: Tuple[str, ...]
    knowledge: Mapping[str, Tuple[str, ...]]
    correctness_suites: Tuple[str, ...]
    first_iteration_suites: Tuple[str, ...]
    metrics: Tuple[str, ...]


def resolve_capabilities(
    req: Dict[str, Any], *, registry_path: Path = _REGISTRY_FILE
) -> Dict[str, Any]:
    """Return a deterministic, JSON-serializable task contract.

    Selected form features are required. Unselected optional features are
    disabled unless explicitly listed in ``allowed_features``. Dependencies
    are expanded without coupling independent optional capabilities.
    """
    registry, registry_bytes = _load_registry(registry_path)
    capabilities = _parse_capabilities(registry)
    selection = _mapping(registry.get("selection"), "selection")
    required_field = str(selection.get("required_field", "features"))
    allowed_field = str(selection.get("allowed_field", "allowed_features"))

    label_to_id = {
        contract.label.casefold(): capability_id
        for capability_id, contract in capabilities.items()
        if contract.selection == "optional"
    }
    required_selected = _resolve_selection(
        req_field(req, required_field, []), label_to_id, capabilities, required_field
    )
    allowed_selected = _resolve_selection(
        req_field(req, allowed_field, []), label_to_id, capabilities, allowed_field
    )
    allowed_selected = tuple(
        capability_id
        for capability_id in allowed_selected
        if capability_id not in set(required_selected)
    )

    baseline = tuple(
        capability_id
        for capability_id, contract in capabilities.items()
        if contract.selection == "baseline"
    )
    required = _expand_dependencies(baseline + required_selected, capabilities)
    allowed = _expand_allowed_dependencies(allowed_selected, required, capabilities)
    disabled = tuple(
        capability_id
        for capability_id, contract in capabilities.items()
        if contract.selection == "optional"
        and capability_id not in required
        and capability_id not in allowed
    )

    warnings = []
    for capability_id in (*required, *allowed):
        contract = capabilities[capability_id]
        if contract.maturity == "unavailable":
            reason = _mapping(
                registry["capabilities"][capability_id],
                f"capabilities.{capability_id}",
            ).get("unavailable_reason")
            suffix = f" {reason}" if reason else ""
            raise CapabilityResolutionError(
                f"{contract.label} is currently unavailable.{suffix}",
                field=required_field if capability_id in required else allowed_field,
            )
        if contract.maturity == "experimental":
            warnings.append(f"{contract.label} is experimental")

    parameters, parameter_sources = _resolve_parameter_group(
        req, registry, "global_parameters"
    )
    performance, _performance_sources = _resolve_parameter_group(
        req, registry, "performance_parameters"
    )
    performance["notes"] = str(req_field(req, "perf_budget", "") or "").strip()
    capability_parameters: Dict[str, Dict[str, Any]] = {}
    for capability_id in (*required, *allowed):
        raw_capability = _mapping(
            registry["capabilities"][capability_id],
            f"capabilities.{capability_id}",
        )
        specs = _mapping(
            raw_capability.get("parameters", {}),
            f"capabilities.{capability_id}.parameters",
        )
        if specs:
            values: Dict[str, Any] = {}
            for name, raw_spec in specs.items():
                value, _source = _resolve_parameter(
                    req, str(name), _mapping(raw_spec, f"{capability_id}.{name}")
                )
                values[str(name)] = value
            capability_parameters[capability_id] = values

    combinations = _matching_combinations(registry, required)
    _validate_compatibility(registry, required, parameters)
    resource_contract = _compile_resource_contract(
        registry, required, parameters, capability_parameters
    )

    correctness_suites = _dedupe(
        suite
        for capability_id in required
        for suite in capabilities[capability_id].correctness_suites
    )
    metrics = _dedupe(
        metric
        for capability_id in required
        for metric in capabilities[capability_id].metrics
    )
    for combination in combinations:
        correctness_suites = _dedupe(
            (*correctness_suites, *_strings(combination.get("correctness_suites", [])))
        )
        metrics = _dedupe((*metrics, *_strings(combination.get("metrics", []))))

    first_iteration_suites = _dedupe(
        suite
        for capability_id in required
        for suite in capabilities[capability_id].first_iteration_suites
    )
    for combination in combinations:
        first_iteration_suites = _dedupe((
            *first_iteration_suites,
            *_strings(combination.get("first_iteration_suites", [])),
        ))

    source_hash = _sha256_json(_source_requirements(req))
    return {
        "schema_version": 1,
        "source_requirements_sha256": source_hash,
        "capability_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "task": {
            "id": str(req.get("task_id", "")),
            "type": str(req.get("task_type", "gen-cpp-infer-framework")),
        },
        "required_capabilities": list(required),
        "allowed_capabilities": list(allowed),
        "disabled_capabilities": list(disabled),
        "parameters": parameters,
        "parameter_sources": parameter_sources,
        "performance": performance,
        "capability_parameters": capability_parameters,
        "resource_contract": resource_contract,
        "active_combination_contracts": [
            str(combination["id"]) for combination in combinations
        ],
        "correctness_suites": list(correctness_suites),
        "first_iteration_suites": list(first_iteration_suites),
        "metrics": list(metrics),
        "warnings": warnings,
    }


def freeze_resolved_requirements(
    req: Dict[str, Any], state_dir: Path, *, registry_path: Path = _REGISTRY_FILE
) -> Dict[str, Any]:
    """Create or reuse the immutable resolved-requirements snapshot."""
    path = state_dir / _RESOLVED_FILE
    source_hash = _sha256_json(_source_requirements(req))
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CapabilityResolutionError(
                f"cannot read frozen {path.name}: {exc}", field="_"
            ) from exc
        if existing.get("source_requirements_sha256") != source_hash:
            raise CapabilityResolutionError(
                "requirements.json changed after resolved requirements were frozen; "
                "reset the task before changing its inputs",
                field="_",
            )
        return existing

    resolved = resolve_capabilities(req, registry_path=registry_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)
    return resolved


def resolved_from_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """Use the orchestrator-injected snapshot, or resolve for legacy callers."""
    embedded = req.get("resolved_requirements") if isinstance(req, dict) else None
    if isinstance(embedded, dict):
        return embedded
    return resolve_capabilities(req)


def capability_contracts(
    *, registry_path: Path = _REGISTRY_FILE
) -> Dict[str, CapabilityContract]:
    registry, _raw = _load_registry(registry_path)
    return _parse_capabilities(registry)


def capability_knowledge_for_role(
    req: Dict[str, Any], role: str, *, registry_path: Path = _REGISTRY_FILE
) -> Tuple[str, ...]:
    """Return required knowledge ids for active capabilities and combinations."""
    resolved = resolved_from_request(req)
    registry, _raw = _load_registry(registry_path)
    capabilities = _parse_capabilities(registry)
    documents = []
    for capability_id in resolved.get("required_capabilities", []):
        contract = capabilities.get(str(capability_id))
        if contract is not None:
            documents.extend(contract.knowledge.get(role, ()))
    active = set(resolved.get("active_combination_contracts", []))
    for combination in _sequence(
        registry.get("combination_contracts", []), "combination_contracts"
    ):
        item = _mapping(combination, "combination_contract")
        if item.get("id") not in active:
            continue
        knowledge = _mapping(item.get("knowledge", {}), "combination.knowledge")
        documents.extend(_strings(knowledge.get(role, [])))
    return _dedupe(documents)


def validate_capability_submission(answers: Dict[str, Any]) -> Dict[str, str]:
    """Web form hook: return field errors without starting an orchestrator."""
    req = {"task_type": "gen-cpp-infer-framework", **answers}
    try:
        resolve_capabilities(req)
    except CapabilityResolutionError as exc:
        return {exc.field: str(exc)}
    return {}


def _load_registry(path: Path) -> Tuple[Dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        parsed = yaml.safe_load(raw) or {}
    except OSError as exc:
        raise CapabilityResolutionError(f"capability registry not found: {path}") from exc
    if not isinstance(parsed, dict) or parsed.get("schema_version") != 1:
        raise CapabilityResolutionError(f"unsupported capability registry: {path}")
    _validate_registry(parsed)
    return parsed, raw


def _validate_registry(registry: Mapping[str, Any]) -> None:
    capabilities = _parse_capabilities(registry)
    labels = [
        contract.label.casefold()
        for contract in capabilities.values()
        if contract.selection == "optional"
    ]
    if len(labels) != len(set(labels)):
        raise CapabilityResolutionError("optional capability labels must be unique")
    for capability_id, contract in capabilities.items():
        if contract.selection == "baseline" and contract.maturity == "unavailable":
            raise CapabilityResolutionError(
                f"baseline capability {capability_id!r} cannot be unavailable"
            )
        _expand_dependencies((capability_id,), capabilities)
        unknown_first_suites = sorted(
            set(contract.first_iteration_suites) - set(contract.correctness_suites)
        )
        if unknown_first_suites:
            raise CapabilityResolutionError(
                f"capability {capability_id!r} has first-iteration suites outside "
                f"its correctness suites: {unknown_first_suites}"
            )

    combination_ids = []
    for raw in _sequence(
        registry.get("combination_contracts", []), "combination_contracts"
    ):
        combination = _mapping(raw, "combination_contract")
        combination_id = str(combination.get("id", "")).strip()
        if not combination_id:
            raise CapabilityResolutionError("combination contract is missing id")
        combination_ids.append(combination_id)
        _validate_capability_ids(
            capabilities,
            _strings(combination.get("capabilities_all", [])),
            f"combination {combination_id!r}",
        )
        if combination.get("maturity") not in _MATURITIES:
            raise CapabilityResolutionError(
                f"combination {combination_id!r} has invalid maturity"
            )
        combination_suites = set(_strings(combination.get("correctness_suites", [])))
        unknown_first_suites = sorted(
            set(_strings(combination.get("first_iteration_suites", [])))
            - combination_suites
        )
        if unknown_first_suites:
            raise CapabilityResolutionError(
                f"combination {combination_id!r} has first-iteration suites "
                f"outside its correctness suites: {unknown_first_suites}"
            )
    if len(combination_ids) != len(set(combination_ids)):
        raise CapabilityResolutionError("combination contract ids must be unique")

    global_parameters = set(
        _mapping(registry.get("global_parameters", {}), "global_parameters")
    )
    for raw in _sequence(
        registry.get("compatibility_rules", []), "compatibility_rules"
    ):
        rule = _mapping(raw, "compatibility_rule")
        rule_id = str(rule.get("id", "")).strip() or "<unnamed>"
        _validate_capability_ids(
            capabilities,
            _strings(rule.get("capabilities_all", [])),
            f"compatibility rule {rule_id!r}",
        )
        parameter_names = set(
            _mapping(rule.get("parameter_values", {}), "parameter_values")
        )
        unknown_parameters = sorted(parameter_names - global_parameters)
        if unknown_parameters:
            raise CapabilityResolutionError(
                f"compatibility rule {rule_id!r} has unknown parameters: "
                f"{unknown_parameters}"
            )


def _validate_capability_ids(
    capabilities: Mapping[str, CapabilityContract], ids: Iterable[str], owner: str
) -> None:
    unknown = [capability_id for capability_id in ids if capability_id not in capabilities]
    if unknown:
        raise CapabilityResolutionError(
            f"{owner} references unknown capabilities: {unknown}"
        )


def _parse_capabilities(registry: Mapping[str, Any]) -> Dict[str, CapabilityContract]:
    raw_capabilities = _mapping(registry.get("capabilities"), "capabilities")
    contracts: Dict[str, CapabilityContract] = {}
    for capability_id, raw in raw_capabilities.items():
        item = _mapping(raw, f"capabilities.{capability_id}")
        maturity = str(item.get("maturity", "")).strip()
        selection = str(item.get("selection", "")).strip()
        if maturity not in _MATURITIES:
            raise CapabilityResolutionError(
                f"capabilities.{capability_id}.maturity is invalid: {maturity!r}"
            )
        if selection not in {"baseline", "optional"}:
            raise CapabilityResolutionError(
                f"capabilities.{capability_id}.selection is invalid: {selection!r}"
            )
        knowledge = {
            str(role): _strings(document_ids)
            for role, document_ids in _mapping(
                item.get("knowledge", {}), f"capabilities.{capability_id}.knowledge"
            ).items()
        }
        contracts[str(capability_id)] = CapabilityContract(
            id=str(capability_id),
            label=str(item.get("label", capability_id)),
            selection=selection,
            maturity=maturity,
            dependencies=_strings(item.get("dependencies", [])),
            knowledge=knowledge,
            correctness_suites=_strings(item.get("correctness_suites", [])),
            first_iteration_suites=_strings(item.get("first_iteration_suites", [])),
            metrics=_strings(item.get("metrics", [])),
        )
    for contract in contracts.values():
        unknown = [dep for dep in contract.dependencies if dep not in contracts]
        if unknown:
            raise CapabilityResolutionError(
                f"capability {contract.id!r} has unknown dependencies: {unknown}"
            )
    return contracts


def _resolve_selection(
    raw: Any,
    label_to_id: Mapping[str, str],
    capabilities: Mapping[str, CapabilityContract],
    field: str,
) -> Tuple[str, ...]:
    values = raw if isinstance(raw, (list, tuple, set, frozenset)) else ([raw] if raw else [])
    resolved = []
    for value in values:
        text = str(value).strip()
        capability_id = text if text in capabilities else label_to_id.get(text.casefold())
        if capability_id is None or capabilities[capability_id].selection != "optional":
            raise CapabilityResolutionError(
                f"unknown optional capability: {text!r}", field=field
            )
        resolved.append(capability_id)
    return _dedupe(resolved)


def _expand_dependencies(
    seeds: Sequence[str], capabilities: Mapping[str, CapabilityContract]
) -> Tuple[str, ...]:
    ordered = []
    visiting = set()
    visited = set()

    def visit(capability_id: str) -> None:
        if capability_id in visited:
            return
        if capability_id in visiting:
            raise CapabilityResolutionError(
                f"capability dependency cycle includes {capability_id!r}"
            )
        visiting.add(capability_id)
        for dependency in capabilities[capability_id].dependencies:
            visit(dependency)
        visiting.remove(capability_id)
        visited.add(capability_id)
        ordered.append(capability_id)

    for seed in seeds:
        visit(seed)
    return tuple(ordered)


def _expand_allowed_dependencies(
    seeds: Sequence[str], required: Sequence[str], capabilities: Mapping[str, CapabilityContract]
) -> Tuple[str, ...]:
    expanded = _expand_dependencies(seeds, capabilities)
    required_set = set(required)
    return tuple(
        capability_id
        for capability_id in expanded
        if capability_id not in required_set
        and capabilities[capability_id].selection == "optional"
    )


def _resolve_parameter_group(
    req: Dict[str, Any], registry: Mapping[str, Any], group: str
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    parameters: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    specs = _mapping(registry.get(group, {}), group)
    for name, raw_spec in specs.items():
        value, source = _resolve_parameter(
            req, str(name), _mapping(raw_spec, f"{group}.{name}")
        )
        parameters[str(name)] = value
        sources[str(name)] = source
    if group == "global_parameters" and parameters.get("weight_format") == "auto":
        inferred = _infer_weight_format(str(parameters.get("target_model") or ""))
        if inferred != "auto":
            parameters["weight_format"] = inferred
            sources["weight_format"] = "inferred:target_model"
    return parameters, sources


def _resolve_parameter(
    req: Dict[str, Any], name: str, spec: Mapping[str, Any]
) -> Tuple[Any, str]:
    source_field = str(spec.get("source_field", name))
    raw = req_field(req, source_field, None)
    source = f"requirements:{source_field}"
    if raw is None or raw == "":
        if "default" in spec:
            raw = spec["default"]
            source = "registry:default"
        elif spec.get("required"):
            raise CapabilityResolutionError(
                f"{source_field} is required", field=source_field
            )
        else:
            return None, "unset"

    value_type = str(spec.get("type", "string"))
    if value_type == "integer":
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise CapabilityResolutionError(
                f"{source_field} must be an integer", field=source_field
            ) from exc
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and value < int(minimum):
            raise CapabilityResolutionError(
                f"{source_field} must be >= {minimum}", field=source_field
            )
        if maximum is not None and value > int(maximum):
            raise CapabilityResolutionError(
                f"{source_field} must be <= {maximum}", field=source_field
            )
        return value, source
    if value_type == "number":
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise CapabilityResolutionError(
                f"{source_field} must be a number", field=source_field
            ) from exc
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and value < float(minimum):
            raise CapabilityResolutionError(
                f"{source_field} must be >= {minimum}", field=source_field
            )
        if maximum is not None and value > float(maximum):
            raise CapabilityResolutionError(
                f"{source_field} must be <= {maximum}", field=source_field
            )
        return value, source
    if value_type == "enum":
        choices = _mapping(spec.get("choices", {}), f"parameter {name}.choices")
        canonical = {str(key).casefold(): str(value) for key, value in choices.items()}
        canonical.update({str(value).casefold(): str(value) for value in choices.values()})
        value = canonical.get(str(raw).casefold())
        if value is None:
            raise CapabilityResolutionError(
                f"{source_field} must be one of {list(choices)}", field=source_field
            )
        return value, source
    if value_type in {"string", "path"}:
        value = str(raw).strip()
        if spec.get("required") and not value:
            raise CapabilityResolutionError(
                f"{source_field} is required", field=source_field
            )
        return value, source
    raise CapabilityResolutionError(
        f"unsupported parameter type {value_type!r} for {name!r}"
    )


def _infer_weight_format(target_model: str) -> str:
    name = Path(target_model).name.casefold()
    if "q8_0" in name or "q8-0" in name:
        return "q8_0_gguf"
    if "bf16" in name:
        return "bf16_gguf"
    if "f16" in name or "fp16" in name:
        return "f16_gguf"
    if name.endswith(".gguf"):
        return "gguf_unknown"
    return "auto"


def _matching_combinations(
    registry: Mapping[str, Any], required: Sequence[str]
) -> Tuple[Mapping[str, Any], ...]:
    required_set = set(required)
    matches = []
    for raw in _sequence(registry.get("combination_contracts", []), "combination_contracts"):
        item = _mapping(raw, "combination_contract")
        capability_ids = set(_strings(item.get("capabilities_all", [])))
        if capability_ids.issubset(required_set):
            if item.get("maturity") == "unavailable":
                raise CapabilityResolutionError(
                    str(item.get("message") or f"combination {item.get('id')} is unavailable")
                )
            matches.append(item)
    return tuple(matches)


def _validate_compatibility(
    registry: Mapping[str, Any], required: Sequence[str], parameters: Mapping[str, Any]
) -> None:
    required_set = set(required)
    for raw in _sequence(registry.get("compatibility_rules", []), "compatibility_rules"):
        rule = _mapping(raw, "compatibility_rule")
        needed = set(_strings(rule.get("capabilities_all", [])))
        if not needed.issubset(required_set):
            continue
        expected = _mapping(rule.get("parameter_values", {}), "parameter_values")
        if not all(parameters.get(name) in set(_strings(values)) for name, values in expected.items()):
            continue
        if rule.get("maturity") == "unavailable":
            raise CapabilityResolutionError(
                str(rule.get("message") or f"combination {rule.get('id')} is unavailable"),
                field="weight_format",
            )


def _compile_resource_contract(
    registry: Mapping[str, Any],
    required: Sequence[str],
    parameters: Mapping[str, Any],
    capability_parameters: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compile cross-field KV and concurrency promises into exact numbers.

    ``max_context_length`` and ``max_concurrency`` are not independent when
    requests share a KV pool.  Keeping this calculation in the requirement
    compiler makes the promise visible to A/B, the runtime metadata validator,
    and the immutable numeric-test contract.
    """
    required_set = set(required)
    max_context = int(parameters.get("max_context_length", 4096) or 4096)
    max_active = 1
    if "continuous_batching" in required_set:
        max_active = int(
            capability_parameters.get("continuous_batching", {}).get(
                "max_concurrency", 4
            )
            or 4
        )

    tp_size = 1
    if "tensor_parallelism" in required_set:
        tp_size = int(
            capability_parameters.get("tensor_parallelism", {}).get("tp_size", 2)
            or 2
        )

    model = _mapping(registry.get("resource_model", {}), "resource_model")
    num_layers = int(model.get("num_layers", 36))
    num_kv_heads = int(model.get("num_kv_heads", 8))
    head_dim = int(model.get("head_dim", 128))
    scalable_kv = (
        "paged_kv_cache" in required_set
        or "continuous_batching" in required_set
    )
    if scalable_kv:
        dtype = str(model.get("scalable_kv_dtype", "fp16"))
        dtype_bytes = int(model.get("scalable_kv_dtype_bytes", 2))
    else:
        dtype = str(model.get("dense_kv_dtype", "fp32"))
        dtype_bytes = int(model.get("dense_kv_dtype_bytes", 4))
    if num_kv_heads % tp_size != 0:
        raise CapabilityResolutionError(
            f"tp_size={tp_size} must divide num_kv_heads={num_kv_heads}",
            field="tp_size",
        )
    local_kv_heads = num_kv_heads // tp_size
    # K and V are both resident for every layer and token.
    bytes_per_token_per_rank = (
        2 * num_layers * local_kv_heads * head_dim * dtype_bytes
    )

    paged = "paged_kv_cache" in required_set
    block_size: Optional[int] = None
    blocks_per_request: Optional[int] = None
    total_blocks: Optional[int] = None
    policy = "full_context_per_request"
    requested_shared_tokens = 0
    if paged:
        paged_params = capability_parameters.get("paged_kv_cache", {})
        block_size = int(paged_params.get("block_size", 16) or 16)
        policy = str(
            paged_params.get("capacity_policy", "full_context_per_request")
            or "full_context_per_request"
        )
        requested_shared_tokens = int(
            paged_params.get("max_total_cached_tokens", 0) or 0
        )
        blocks_per_request = (max_context + block_size - 1) // block_size
        if policy == "shared_token_budget":
            requested = requested_shared_tokens or max_context
            if requested < max_context:
                raise CapabilityResolutionError(
                    "max_total_cached_tokens must be at least max_context_length "
                    "so one request can fulfill the advertised context limit",
                    field="max_total_cached_tokens",
                )
            total_blocks = (requested + block_size - 1) // block_size
            max_total_cached_tokens = total_blocks * block_size
            guaranteed_full_context_requests = min(
                max_active, total_blocks // blocks_per_request
            )
        else:
            total_blocks = blocks_per_request * max_active
            max_total_cached_tokens = total_blocks * block_size
            guaranteed_full_context_requests = max_active
    else:
        # A contiguous/non-paged implementation still owes one complete KV
        # allocation per active request; it simply has no block accounting.
        max_total_cached_tokens = max_context * max_active
        guaranteed_full_context_requests = max_active

    required_metadata = []
    if paged or "continuous_batching" in required_set:
        required_metadata.append("max_context_length")
    if "continuous_batching" in required_set:
        required_metadata.append("max_concurrency")
    if paged:
        required_metadata.extend(("kv_capacity_policy", "kv_block_size"))

    advisory_metadata = []
    if paged or "continuous_batching" in required_set:
        advisory_metadata.extend((
            "max_total_cached_tokens",
            "guaranteed_full_context_requests",
            "kv_bytes_per_token_per_rank",
            "kv_pool_bytes_per_rank",
        ))
    if paged:
        advisory_metadata.append("kv_total_blocks")

    return {
        "model_contract": {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "local_kv_heads_per_rank": local_kv_heads,
            "head_dim": head_dim,
            "kv_dtype": dtype,
            "kv_dtype_bytes": dtype_bytes,
            "tp_size": tp_size,
        },
        "max_context_per_request": max_context,
        "max_active_requests": max_active,
        "kv_capacity_policy": policy if paged else "per_sequence_allocation",
        "requested_shared_token_budget": requested_shared_tokens,
        "kv_block_size": block_size,
        "kv_blocks_per_full_context_request": blocks_per_request,
        "kv_total_blocks_per_rank": total_blocks,
        "max_total_cached_tokens": max_total_cached_tokens,
        "guaranteed_full_context_requests": guaranteed_full_context_requests,
        "kv_bytes_per_token_per_rank": bytes_per_token_per_rank,
        "kv_pool_bytes_per_rank": (
            max_total_cached_tokens * bytes_per_token_per_rank
        ),
        "required_runtime_metadata": required_metadata,
        "advisory_runtime_metadata": advisory_metadata,
    }


def _source_requirements(req: Dict[str, Any]) -> Dict[str, Any]:
    source = dict(req)
    source.pop("resolved_requirements", None)
    return source


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapabilityResolutionError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CapabilityResolutionError(f"{name} must be a sequence")
    return value


def _strings(value: Any) -> Tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value, "list value"))


def _dedupe(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


__all__ = [
    "CapabilityContract",
    "CapabilityResolutionError",
    "capability_contracts",
    "capability_knowledge_for_role",
    "freeze_resolved_requirements",
    "resolve_capabilities",
    "resolved_from_request",
    "validate_capability_submission",
]
