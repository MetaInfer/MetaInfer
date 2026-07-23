"""Task-local acceptance contracts for the generated C++ framework.

The shared orchestrator only knows how to move phases.  This module keeps the
meaning of *complete* inside the task package: the selected capabilities,
their immutable suites, artifact checks, runtime evidence, and the optional
performance gate are all resolved from the task's frozen requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from metainfer.orchestrator.requirements import req_field, req_field_int

from .capabilities import resolved_from_request


@dataclass(frozen=True)
class AcceptanceContract:
    required_capabilities: Tuple[str, ...]
    allowed_capabilities: Tuple[str, ...]
    disabled_capabilities: Tuple[str, ...]
    correctness_suites: Tuple[str, ...]
    first_iteration_suites: Tuple[str, ...]
    active_combinations: Tuple[str, ...]
    parameters: Mapping[str, Any]
    capability_parameters: Mapping[str, Any]
    resource_contract: Mapping[str, Any]
    performance: Mapping[str, Any]

    @classmethod
    def from_request(cls, req: Dict[str, Any]) -> "AcceptanceContract":
        resolved = resolved_from_request(req)
        return cls(
            required_capabilities=tuple(resolved.get("required_capabilities", [])),
            allowed_capabilities=tuple(resolved.get("allowed_capabilities", [])),
            disabled_capabilities=tuple(resolved.get("disabled_capabilities", [])),
            correctness_suites=tuple(resolved.get("correctness_suites", [])),
            first_iteration_suites=tuple(
                resolved.get("first_iteration_suites", [])
            ),
            active_combinations=tuple(resolved.get("active_combination_contracts", [])),
            parameters=dict(resolved.get("parameters", {})),
            capability_parameters=dict(resolved.get("capability_parameters", {})),
            resource_contract=dict(resolved.get("resource_contract", {})),
            performance=dict(resolved.get("performance", {})),
        )

    def requires(self, capability_id: str) -> bool:
        return capability_id in self.required_capabilities

    def has_combination(self, combination_id: str) -> bool:
        return combination_id in self.active_combinations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_capabilities": list(self.required_capabilities),
            "allowed_capabilities": list(self.allowed_capabilities),
            "disabled_capabilities": list(self.disabled_capabilities),
            "correctness_suites": list(self.correctness_suites),
            "first_iteration_suites": list(self.first_iteration_suites),
            "active_combination_contracts": list(self.active_combinations),
            "parameters": dict(self.parameters),
            "capability_parameters": dict(self.capability_parameters),
            "resource_contract": dict(self.resource_contract),
            "performance": dict(self.performance),
        }


def validate_plan_artifacts(
    iter_dir: Path,
    req: Dict[str, Any],
    *,
    iteration: int = 1,
    prior_failure: Optional[str] = None,
) -> Tuple[str, ...]:
    """Validate A's deliverables before B is allowed to act on them."""
    missing = [
        name
        for name in ("plan.md", "test_spec.md")
        if not (iter_dir / name).is_file()
    ]
    if missing:
        return tuple(f"planner did not create {name}" for name in missing)

    try:
        plan = (iter_dir / "plan.md").read_text(encoding="utf-8", errors="replace")
        spec = (iter_dir / "test_spec.md").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return (f"planner deliverables cannot be read: {exc}",)

    errors = []
    combined = f"{plan}\n{spec}".casefold()
    if not any(word in combined for word in ("architecture", "design", "架构")):
        errors.append("plan/test_spec must contain an architecture or design section")
    if not any(word in combined for word in ("roadmap", "milestone", "iteration", "轮")):
        errors.append("plan must contain an iteration roadmap or milestones section")
    if not any(word in combined for word in ("test", "correctness", "验收")):
        errors.append("plan/test_spec must contain a correctness test section")
    if prior_failure:
        if not re.search(
            r"(?im)^#{1,6}\s+(?:previous\s+)?failure\s+(?:evidence|diagnosis)\b",
            plan,
        ):
            errors.append(
                "plan.md must contain a Failure evidence heading after a prior phase failure"
            )
        evidence_anchors = (
            "events.jsonl",
            "retrospective.md",
            "status.json",
            "oracle-report.json",
            "server.stderr",
        )
        if not any(anchor in plan.casefold() for anchor in evidence_anchors):
            errors.append(
                "Failure evidence must cite a prior diagnostic artifact, not only a hypothesis"
            )
        target_model = str(req_field(req, "target_model", "") or "").casefold()
        weight_format = str(req_field(req, "weight_format", "") or "").casefold()
        if target_model.endswith(".gguf") or "gguf" in weight_format:
            has_data_base = any(
                term in combined
                for term in ("data_offset", "data offset", "data blob", "data section")
            )
            has_relative_offset = any(
                term in combined
                for term in ("tensor.offset", "relative offset", "relative to data")
            )
            has_alignment = any(
                term in combined
                for term in ("align_up", "alignment", "aligned file offset")
            )
            if not (has_data_base and has_relative_offset and has_alignment):
                errors.append(
                    "GGUF recovery plan must verify that tensor offsets are relative "
                    "to the aligned tensor data blob: data_offset = "
                    "align_up(tensor_info_end, general.alignment), then read each "
                    "tensor at data_offset + tensor.offset. Treat this loader audit "
                    "and a known-byte/value check as the first recovery gate; finite "
                    "or non-zero embeddings do not prove that GGUF offsets are correct"
                )

    contract = AcceptanceContract.from_request(req)
    for capability_id in contract.required_capabilities:
        aliases = _capability_aliases(capability_id)
        if not any(alias in combined for alias in aliases):
            errors.append(
                f"required capability {capability_id} is absent from plan.md/test_spec.md"
            )
    for suite in contract.correctness_suites:
        if suite.casefold() not in combined:
            # Suite names are implementation-facing IDs. Accept a readable
            # version as well so planners do not have to copy punctuation.
            readable = suite.replace("_", " ").replace(".", " ")
            if readable.casefold() not in combined:
                errors.append(f"correctness suite {suite} is absent from test_spec.md")
    runtime_evidence_terms = []
    if contract.requires("tensor_parallelism"):
        runtime_evidence_terms.extend((
            "tp_size",
            "world_size",
            "rank",
            "device_ids",
            "weight_sharding",
            "collective_backend",
        ))
    if contract.requires("paged_kv_cache"):
        runtime_evidence_terms.append("kv_block_size")
    if contract.requires("continuous_batching"):
        runtime_evidence_terms.extend(("max_concurrency", "max_observed_batch_size"))
    runtime_evidence_terms.extend(
        str(term)
        for term in contract.resource_contract.get("required_runtime_metadata", [])
    )
    runtime_evidence_terms = list(dict.fromkeys(runtime_evidence_terms))
    if runtime_evidence_terms and "/v1/models" not in combined:
        errors.append("plan/test_spec must define runtime evidence in GET /v1/models")
    for term in runtime_evidence_terms:
        if term.casefold() not in combined:
            errors.append(
                f"plan/test_spec does not define /v1/models runtime evidence field {term}"
            )
    if contract.requires("tensor_parallelism") and _plans_full_model_tp1(combined):
        errors.append(
            "TP plan may not use a full-target-model single-rank/TP1 E2E path; "
            "use the frozen tp_size for real-model loading, forward, generation, "
            "and integration validation. Reserve one-rank tests for reduced "
            "synthetic or rank-local checks with an explicit memory bound; TP "
            "numeric parity may compare reduced synthetic TP1 with the required "
            "topology, but must not load the complete real weights on one device"
        )
    if _plans_unsubstantiated_gemm_layout_rewrite(combined):
        errors.append(
            "a hipBLAS layout rewrite must not be inferred from column-major "
            "terminology alone; cite an observed CPU-reference mismatch and "
            "show transA/transB/M/N/K/lda/ldb/ldc before swapping dimensions "
            "or inserting a transpose. Column-major C[N,T] with ldc=N already "
            "has the contiguous row-major [T,N] byte layout"
        )
    if _plans_active_work_deferral(combined, contract, iteration):
        errors.append(
            "plan prose may not defer an active capability, correctness suite, "
            "or implementation phase to a later iteration; plan_manifest.json "
            "milestone deadlines are authoritative"
        )
    errors.extend(_validate_plan_manifest(iter_dir, contract, iteration, req))
    return tuple(errors)


def _validate_plan_manifest(
    iter_dir: Path,
    contract: AcceptanceContract,
    iteration: int,
    req: Dict[str, Any],
) -> Tuple[str, ...]:
    path = iter_dir / "plan_manifest.json"
    if not path.is_file():
        return ("planner did not create plan_manifest.json",)
    manifest, load_error = _load_plan_manifest(path)
    if load_error:
        return (load_error,)
    assert manifest is not None

    errors = []
    if manifest.get("schema_version") != 1:
        errors.append("plan_manifest.json schema_version must be 1")
    manifest_iteration = manifest.get("iteration")
    if not isinstance(manifest_iteration, int) or isinstance(manifest_iteration, bool):
        errors.append("plan_manifest.json iteration must be an integer")
    elif manifest_iteration != iteration:
        errors.append(
            f"plan_manifest.json iteration={manifest_iteration} does not match {iteration}"
        )

    required_caps = _manifest_string_list(
        manifest, "required_capabilities", errors
    )
    current_caps = _manifest_string_list(
        manifest, "current_iteration_capabilities", errors
    )
    current_suites = _manifest_string_list(manifest, "current_suites", errors)
    deferred_suites = _manifest_string_list(manifest, "deferred_suites", errors)
    _compare_manifest_list(
        errors,
        "required_capabilities",
        required_caps,
        contract.required_capabilities,
    )

    max_iterations = max(1, req_field_int(req, "max_iterations", 20))
    milestones = manifest.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        errors.append("plan_manifest.json milestones must be a non-empty array")
        milestones = []

    capability_at: Dict[str, int] = {}
    suite_at: Dict[str, int] = {}
    required_cap_set = set(contract.required_capabilities)
    required_suite_set = set(contract.correctness_suites)
    for index, raw_milestone in enumerate(milestones):
        owner = f"plan_manifest.json milestones[{index}]"
        if not isinstance(raw_milestone, Mapping):
            errors.append(f"{owner} must be an object")
            continue
        milestone_iteration = raw_milestone.get("iteration")
        if (
            not isinstance(milestone_iteration, int)
            or isinstance(milestone_iteration, bool)
            or not 1 <= milestone_iteration <= max_iterations
        ):
            errors.append(
                f"{owner}.iteration must be between 1 and {max_iterations}"
            )
            continue
        milestone_caps = _mapping_string_list(
            raw_milestone, "capabilities", owner, errors
        )
        milestone_suites = _mapping_string_list(
            raw_milestone, "suites", owner, errors
        )
        deliverables = _mapping_string_list(
            raw_milestone, "deliverables", owner, errors
        )
        if not deliverables:
            errors.append(f"{owner}.deliverables must name at least one runnable outcome")
        for capability_id in milestone_caps:
            if capability_id not in required_cap_set:
                errors.append(f"{owner} assigns unknown capability {capability_id}")
            elif capability_id in capability_at:
                errors.append(
                    f"capability {capability_id} is assigned by more than one milestone"
                )
            else:
                capability_at[capability_id] = milestone_iteration
        for suite in milestone_suites:
            if suite not in required_suite_set:
                errors.append(f"{owner} assigns unknown correctness suite {suite}")
            elif suite in suite_at:
                errors.append(f"correctness suite {suite} is assigned by more than one milestone")
            else:
                suite_at[suite] = milestone_iteration

    missing_caps = [cap for cap in contract.required_capabilities if cap not in capability_at]
    missing_suites = [suite for suite in contract.correctness_suites if suite not in suite_at]
    if missing_caps:
        errors.append(f"milestones do not assign required capabilities: {missing_caps}")
    if missing_suites:
        errors.append(f"milestones do not assign correctness suites: {missing_suites}")

    prior_capability_at, prior_suite_at = _prior_milestone_deadlines(
        iter_dir, iteration, contract, errors
    )
    for capability_id, prior_deadline in prior_capability_at.items():
        current_deadline = capability_at.get(capability_id, max_iterations + 1)
        if current_deadline > prior_deadline:
            errors.append(
                f"capability {capability_id} cannot be postponed from iteration "
                f"{prior_deadline} to {current_deadline}"
            )
    for suite, prior_deadline in prior_suite_at.items():
        current_deadline = suite_at.get(suite, max_iterations + 1)
        if current_deadline > prior_deadline:
            errors.append(
                f"correctness suite {suite} cannot be postponed from iteration "
                f"{prior_deadline} to {current_deadline}"
            )

    expected_current_caps = tuple(
        cap
        for cap in contract.required_capabilities
        if capability_at.get(cap, max_iterations + 1) <= iteration
    )
    expected_current_suites = tuple(
        suite
        for suite in contract.correctness_suites
        if suite_at.get(suite, max_iterations + 1) <= iteration
    )
    expected_deferred_suites = tuple(
        suite for suite in contract.correctness_suites if suite not in expected_current_suites
    )
    _compare_manifest_list(
        errors,
        "current_iteration_capabilities",
        current_caps,
        expected_current_caps,
    )
    _compare_manifest_list(
        errors, "current_suites", current_suites, expected_current_suites
    )
    _compare_manifest_list(
        errors, "deferred_suites", deferred_suites, expected_deferred_suites
    )

    if iteration == 1:
        deferred_caps = [
            cap for cap in contract.required_capabilities if capability_at.get(cap) != 1
        ]
        if deferred_caps:
            errors.append(
                "iteration 1 must include executable paths for every required "
                f"capability; deferred capabilities: {deferred_caps}"
            )
        missing_first_suites = [
            suite
            for suite in contract.first_iteration_suites
            if suite_at.get(suite) != 1
        ]
        if missing_first_suites:
            errors.append(
                "iteration 1 must implement the runnable vertical-slice suites: "
                f"{missing_first_suites}"
            )
    if iteration >= max_iterations and expected_deferred_suites:
        errors.append(
            "the final iteration cannot defer correctness suites: "
            f"{list(expected_deferred_suites)}"
        )
    return tuple(errors)


def annotate_development_gate(
    report_path: Path,
    iter_dir: Path,
    req: Dict[str, Any],
    *,
    iteration: int,
) -> Dict[str, Any]:
    """Attach the current milestone gate to a full immutable-oracle report.

    The oracle always probes the frozen final contract. During development,
    only the cumulative suites activated by A's manifest block C. The final
    iteration and FinalAudit still require the complete frozen suite set.
    """
    contract = AcceptanceContract.from_request(req)
    report = _read_json(report_path)
    report_mapping = report if isinstance(report, Mapping) else {}
    manifest, manifest_error = _load_plan_manifest(iter_dir / "plan_manifest.json")
    errors = []
    if manifest_error:
        errors.append(manifest_error)
        active_suites = tuple(contract.correctness_suites)
    else:
        raw_active = manifest.get("current_suites") if manifest is not None else None
        if not isinstance(raw_active, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_active
        ):
            errors.append("plan_manifest.json current_suites is invalid")
            active_suites = tuple(contract.correctness_suites)
        else:
            active_set = set(raw_active)
            unknown = sorted(active_set - set(contract.correctness_suites))
            if unknown:
                errors.append(f"plan manifest activates unknown suites: {unknown}")
            active_suites = tuple(
                suite for suite in contract.correctness_suites if suite in active_set
            )

    max_iterations = max(1, req_field_int(req, "max_iterations", 20))
    if iteration >= max_iterations:
        active_suites = tuple(contract.correctness_suites)

    acceptance = report_mapping.get("acceptance")
    suite_results = (
        acceptance.get("suite_results", [])
        if isinstance(acceptance, Mapping)
        else []
    )
    suite_by_id = {
        str(item.get("suite")): item
        for item in suite_results
        if isinstance(item, Mapping)
    }
    missing = [suite for suite in active_suites if suite not in suite_by_id]
    failed = [
        suite
        for suite in active_suites
        if suite in suite_by_id and suite_by_id[suite].get("passed") is not True
    ]
    if not isinstance(acceptance, Mapping):
        errors.append("oracle report has no capability acceptance section")
    if missing:
        errors.append(f"oracle report is missing active suites: {missing}")
    if failed:
        errors.append(f"active milestone suites failed: {failed}")
    gate = {
        "iteration": iteration,
        "final_iteration": iteration >= max_iterations,
        "required_suites": list(active_suites),
        "deferred_suites": [
            suite for suite in contract.correctness_suites if suite not in active_suites
        ],
        "failed_suites": failed,
        "missing_suites": missing,
        "full_acceptance_passed": bool(
            isinstance(acceptance, Mapping) and acceptance.get("passed") is True
        ),
        "passed": not errors,
        "errors": errors,
    }
    if isinstance(report, dict):
        report["development_gate"] = gate
        tmp = report_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            tmp.replace(report_path)
        except OSError as exc:
            gate["passed"] = False
            gate["errors"].append(f"cannot persist development gate: {exc}")
    return gate


def validate_implementation_artifacts(iter_dir: Path, req: Dict[str, Any]) -> Tuple[str, ...]:
    """Check the minimum runnable artifact contract after B."""
    required = ["CMakeLists.txt", "serve.sh", "src", "include"]
    errors = [f"implementer did not create {name}" for name in required if not (iter_dir / name).exists()]
    serve = iter_dir / "serve.sh"
    if serve.is_file() and not _is_executable(serve):
        errors.append("serve.sh is not executable")
    if (iter_dir / "src").is_dir():
        for name in ("main.cpp", "engine.cpp"):
            if not (iter_dir / "src" / name).is_file():
                errors.append(f"implementer did not create src/{name}")
    target_model = str(req_field(req, "target_model", "") or "").casefold()
    weight_format = str(req_field(req, "weight_format", "") or "").casefold()
    if target_model.endswith(".gguf") or "gguf" in weight_format:
        loader_sources = [
            path
            for root in (iter_dir / "src", iter_dir / "include")
            if root.is_dir()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
            and any(term in path.name.casefold() for term in ("gguf", "loader"))
        ]
        loader_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in loader_sources
        ).casefold()
        has_data_base = any(
            term in loader_text
            for term in (
                "data_offset", "data offset", "tensor_data_base",
                "tensor data base", "data_section_offset",
            )
        )
        has_alignment = any(
            term in loader_text
            for term in ("general.alignment", "align_up", "alignment")
        )
        has_relative_tensor_offset = any(
            term in loader_text
            for term in ("tensor.offset", "info.offset", "tensor_offset")
        )
        if not loader_sources or not (
            has_data_base and has_alignment and has_relative_tensor_offset
        ):
            errors.append(
                "GGUF loader implementation must compute an aligned tensor data "
                "base and apply each relative tensor offset: data_offset = "
                "align_up(tensor_info_end, general.alignment), then file_offset "
                "= data_offset + tensor.offset"
            )
    return tuple(errors)


def validate_runtime_metadata(payload: Any, req: Dict[str, Any]) -> Tuple[str, ...]:
    """Validate the observable capability metadata returned by /v1/models.

    Metadata is not accepted as the sole proof of a capability.  The oracle
    also runs behavioral probes.  It is nevertheless required so a TP task
    cannot silently boot as TP1 without exposing its world size and rank.
    """
    contract = AcceptanceContract.from_request(req)
    metadata = extract_runtime_metadata(payload)
    if metadata is None:
        if any(contract.requires(cap) for cap in ("tensor_parallelism", "paged_kv_cache", "continuous_batching")):
            return ("/v1/models does not expose metainfer runtime capability metadata",)
        return ()

    errors = []
    advertised = _string_set(metadata.get("capabilities", []))
    for capability_id in ("tensor_parallelism", "paged_kv_cache", "continuous_batching"):
        if contract.requires(capability_id) and not _contains_capability(advertised, capability_id):
            errors.append(f"runtime metadata does not advertise required capability {capability_id}")

    if contract.requires("tensor_parallelism"):
        expected = int(contract.capability_parameters.get("tensor_parallelism", {}).get("tp_size", 0) or 0)
        for key in ("tp_size", "world_size"):
            value = _first_number(metadata, key)
            if value is None:
                errors.append(f"runtime metadata is missing {key} for TP")
            elif int(value) != expected:
                errors.append(f"runtime metadata {key}={value} does not match required TP size {expected}")
        if _first_number(metadata, "rank") is None:
            errors.append("runtime metadata is missing rank for TP")
        devices = metadata.get("devices") or metadata.get("device_ids")
        if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes)):
            errors.append("runtime metadata is missing TP device_ids/devices")
        elif len(devices) != expected:
            errors.append(
                f"runtime metadata exposes {len(devices)} TP devices, expected {expected}"
            )
        sharding = metadata.get("weight_sharding") or metadata.get("sharded_weights")
        if sharding not in (True, "tensor_parallel", "sharded", "tp", "column_row"):
            errors.append("runtime metadata does not confirm tensor-parallel weight sharding")
        if not str(metadata.get("collective_backend") or "").strip():
            errors.append("runtime metadata is missing collective_backend for TP")

    if contract.requires("paged_kv_cache"):
        expected_block = int(
            contract.capability_parameters.get("paged_kv_cache", {}).get("block_size", 16)
            or 16
        )
        actual_block = _first_number(metadata, "kv_block_size")
        if actual_block is None:
            errors.append("runtime metadata is missing kv_block_size for Paged KV")
        elif int(actual_block) != expected_block:
            errors.append(
                f"runtime metadata kv_block_size={actual_block} does not match {expected_block}"
            )

    resource = contract.resource_contract
    if contract.requires("paged_kv_cache") or contract.requires("continuous_batching"):
        expected_context = int(resource.get("max_context_per_request", 0) or 0)
        actual_context = None
        for key in ("max_context_length", "max_context_per_request", "max_seq_len"):
            actual_context = _first_number(metadata, key)
            if actual_context is not None:
                break
        if actual_context is None:
            errors.append("runtime metadata is missing resource field max_context_length")
        elif int(actual_context) < expected_context:
            errors.append(
                "runtime resource max_context_length="
                f"{actual_context} is below frozen contract {expected_context}"
            )

    if contract.requires("paged_kv_cache"):
        expected_policy = str(resource.get("kv_capacity_policy") or "")
        actual_policy = str(metadata.get("kv_capacity_policy") or "")
        if actual_policy != expected_policy:
            errors.append(
                "runtime kv_capacity_policy="
                f"{actual_policy or '<missing>'} does not match {expected_policy}"
            )

    if contract.requires("continuous_batching"):
        expected_concurrency = int(
            contract.capability_parameters.get("continuous_batching", {}).get(
                "max_concurrency", 4
            ) or 4
        )
        actual_concurrency = _first_number(metadata, "max_concurrency")
        if actual_concurrency is None:
            errors.append("runtime metadata is missing max_concurrency for Continuous Batching")
        elif int(actual_concurrency) < expected_concurrency:
            errors.append(
                f"runtime max_concurrency={actual_concurrency} is below required "
                f"{expected_concurrency}"
            )
    return tuple(errors)


def compile_suite_results(
    contract: AcceptanceContract,
    *,
    baseline_passed: bool,
    probe_verdicts: Mapping[str, str],
) -> Tuple[Dict[str, Any], ...]:
    """Bind every declared suite to concrete immutable-oracle evidence."""
    evidence_map = {
        "base.build_and_lifecycle": ("build-and-health",),
        "base.real_model_loading": ("build-and-health", "baseline-hard-cases"),
        "base.operator_numerics": ("numeric-operator-contract",),
        "base.forward_numerics": ("baseline-hard-cases",),
        "base.tokenizer_and_chat_template": ("baseline-hard-cases",),
        "base.single_sequence_generation": ("baseline-hard-cases",),
        "base.openai_http_api": ("baseline-hard-cases",),
        "paged_kv.block_lifecycle": (
            "capability-runtime-metadata", "capability-paged-kv-long-context",
        ),
        "paged_kv.paged_attention": ("capability-paged-kv-long-context",),
        "continuous_batching.admission": (
            "capability-runtime-metadata",
            "capability-continuous-batching-concurrency",
        ),
        "continuous_batching.sequence_isolation": (
            "capability-continuous-batching-concurrency",
        ),
        "tensor_parallel.rank_startup": ("capability-runtime-metadata",),
        "tensor_parallel.sharding": ("capability-runtime-metadata",),
        "tensor_parallel.numeric_parity": (
            "numeric-operator-contract", "capability-runtime-metadata",
            "baseline-hard-cases",
        ),
        "integration.paged_kv_continuous_batching": (
            "capability-paged-kv-long-context",
            "capability-continuous-batching-concurrency",
        ),
        "integration.tp_paged_kv": (
            "capability-runtime-metadata",
            "capability-paged-kv-long-context",
        ),
        "integration.tp_continuous_batching": (
            "capability-runtime-metadata",
            "capability-continuous-batching-concurrency",
        ),
        "integration.tp_paged_kv_continuous_batching": (
            "capability-tp-paged-cb-integration",
        ),
    }
    results = []
    for suite in contract.correctness_suites:
        evidence = evidence_map.get(suite, ())
        checks = []
        for item in evidence:
            if item == "build-and-health":
                checks.append(True)
            elif item == "baseline-hard-cases":
                checks.append(baseline_passed)
            else:
                checks.append(probe_verdicts.get(item) == "pass")
        results.append({
            "suite": suite,
            "evidence": list(evidence),
            "passed": bool(evidence) and all(checks),
            "reason": "evidence passed" if evidence and all(checks)
            else ("suite has no immutable evidence mapping" if not evidence
                  else "one or more evidence checks failed"),
        })
    return tuple(results)


def performance_gate(req: Dict[str, Any], report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Evaluate performance only when explicitly requested outside the form."""
    explicit = bool(req_field(req, "enforce_performance", False))
    if not explicit:
        return {"required": False, "passed": True, "reason": "informational performance only"}
    if not report:
        return {"required": True, "passed": False, "reason": "missing performance report"}
    errors = int(float(report.get("errors_total", 0) or 0))
    requests = int(float(report.get("num_requests", 0) or 0))
    tps = float(report.get("tokens_per_sec", 0) or 0)
    error_rate = errors / requests if requests else 1.0
    performance = AcceptanceContract.from_request(req).performance
    minimum_tps = float(performance.get("minimum_tokens_per_second", 0) or 0)
    maximum_error_rate = float(performance.get("maximum_error_rate", 0) or 0)
    reasons = []
    if requests <= 0:
        reasons.append("performance report contains no requests")
    if tps <= 0:
        reasons.append("tokens_per_sec must be positive")
    if tps < minimum_tps:
        reasons.append(f"tokens_per_sec {tps} < minimum {minimum_tps}")
    if error_rate > maximum_error_rate:
        reasons.append(f"error rate {error_rate:.4f} > maximum {maximum_error_rate:.4f}")
    return {
        "required": True,
        "passed": not reasons,
        "tokens_per_sec": tps,
        "errors_total": errors,
        "num_requests": requests,
        "error_rate": error_rate,
        "reason": "; ".join(reasons) if reasons else "performance gate passed",
    }


def audit_iteration(
    req: Dict[str, Any],
    iter_dir: Path,
    logs_dir: Path,
    record: Mapping[str, Any],
    *,
    require_success_status: bool = True,
) -> Dict[str, Any]:
    """Final deterministic audit. It never trusts an agent's prose verdict."""
    failures = []
    if require_success_status and record.get("status") != "success":
        failures.append(f"iteration status is {record.get('status')!r}")
    artifact_errors = validate_implementation_artifacts(iter_dir, req)
    failures.extend(artifact_errors)
    phases = record.get("phases", {}) if isinstance(record, Mapping) else {}
    c_phase = phases.get("C_test", {}) if isinstance(phases, Mapping) else {}
    if c_phase.get("outcome") != "ok":
        failures.append(f"C_test outcome is {c_phase.get('outcome')!r}, expected 'ok'")
    d_phase = phases.get("D_review", {}) if isinstance(phases, Mapping) else {}
    if d_phase.get("outcome") != "ok":
        failures.append(f"D_review outcome is {d_phase.get('outcome')!r}, expected 'ok'")

    report_path = logs_dir / "oracle-report.json"
    report = _read_json(report_path)
    if not isinstance(report, Mapping):
        failures.append("missing oracle-report.json")
        acceptance = None
    else:
        if report.get("passed") is not True:
            failures.append(str(report.get("failure_reason") or "correctness oracle did not pass"))
        acceptance = report.get("acceptance")
        contract = AcceptanceContract.from_request(req)
        if not isinstance(acceptance, Mapping) or acceptance.get("passed") is not True:
            failures.append("capability acceptance probes did not pass")
        elif list(acceptance.get("required_capabilities", [])) != list(contract.required_capabilities):
            failures.append("oracle acceptance report does not match frozen required capabilities")
        else:
            suite_results = acceptance.get("suite_results", [])
            suite_by_id = {
                str(item.get("suite")): item
                for item in suite_results
                if isinstance(item, Mapping)
            }
            missing_suites = [
                suite for suite in contract.correctness_suites
                if suite not in suite_by_id
            ]
            failed_suites = [
                suite for suite in contract.correctness_suites
                if suite in suite_by_id and suite_by_id[suite].get("passed") is not True
            ]
            if missing_suites:
                failures.append(f"oracle report is missing suites: {missing_suites}")
            if failed_suites:
                failures.append(f"oracle report has failed suites: {failed_suites}")

    perf_path = iter_dir / "perf_report.json"
    perf_report = _read_json(perf_path)
    perf = performance_gate(req, perf_report if isinstance(perf_report, Mapping) else None)
    if perf["required"] and not perf["passed"]:
        failures.append(perf["reason"])
    return {
        "passed": not failures,
        "failures": failures,
        "required_capabilities": list(AcceptanceContract.from_request(req).required_capabilities),
        "acceptance": acceptance,
        "performance": perf,
    }


def read_review_verdict(path: Path) -> Optional[str]:
    """Read the reviewer's required PASS/NEEDS_FIX line."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?Verdict(?:\*\*)?\s*:\s*\**\s*"
        r"(PASS|NEEDS_FIX)\b",
        text,
        re.I | re.M,
    )
    return match.group(1).upper() if match else None


def _capability_aliases(capability_id: str) -> Tuple[str, ...]:
    aliases = {
        "cpp_build": ("cpp_build", "native c++", "build"),
        "model_loading": ("model_loading", "model loading", "gguf"),
        "model_forward": ("model_forward", "model forward", "forward"),
        "tokenization": ("tokenization", "tokenizer"),
        "kv_cache": ("kv_cache", "kv cache"),
        "openai_http_api": ("openai_http_api", "http", "openai"),
        "paged_kv_cache": ("paged_kv_cache", "paged kv", "paged kv cache", "paged-kv"),
        "continuous_batching": ("continuous_batching", "continuous batching"),
        "tensor_parallelism": (
            "tensor_parallelism", "tensor parallel", "tensor parallelism", "tp2", "tp",
        ),
    }
    return aliases.get(capability_id, (capability_id,))


def _plans_full_model_tp1(text: str) -> bool:
    """Detect a TP plan that substitutes full-model TP1 for TP validation."""
    single_rank = (
        r"(?:single[\s_-]*(?:rank|device|gpu)|one[\s_-]*(?:device|gpu)|"
        r"tp[\s_-]*1|device\s*0\s*only)"
    )
    full_model = (
        r"(?:full[\s_-]*(?:target[\s_-]*)?model|full[\s_-]*36[\s_-]*layer|"
        r"36[\s_-]*layer|end[\s_-]*to[\s_-]*end|e2e|full[\s_-]*weights|"
        r"real[\s_-]*(?:model|f16|gguf|weights)|qwen3[\s_-]*8b|"
        r"(?:model[\s_-]*|runtime[\s_-]*)?forward[\s_-]*pass)"
    )
    prohibited = re.compile(
        r"(?:\b(?:no|never|not|cannot|forbids?|forbidden|prohibits?|prohibited)\b|"
        r"\b(?:do|does|must|may|should)\s+not\b|\binstead\s+of\b|"
        r"禁止|不得|不能|不可)",
        re.I,
    )
    comparison = re.compile(
        rf"(?:{single_rank}.{{0,180}}{full_model}|"
        rf"{full_model}.{{0,180}}{single_rank})",
        re.I | re.S,
    )
    reduced_check = re.compile(
        r"\b(?:reduced|synthetic|rank[\s_-]*local|operator|microbench(?:mark)?)\b|"
        r"\b(?:explicit|bounded)\s+(?:memory|tensor|size)\b|"
        r"\b(?:memory|tensor|size)\s+bound\b|\b\d+\s*(?:kib|mib)\b",
        re.I,
    )

    # Evaluate one statement at a time. This avoids treating a valid contrast
    # such as "synthetic TP1; full target TP2" as a forbidden full-model TP1
    # plan, and avoids rejecting a plan that quotes the validator's prohibition.
    statements = re.split(
        r"(?<=[.!?。；;])(?:[*_`]+)?\s+|\n\s*\n",
        text,
    )
    quoted_validator_error = re.compile(
        r"(?:validator|validation|review)\s+(?:error|failure)|"
        r"plan\s+validation\s+failed",
        re.I,
    )
    return any(
        comparison.search(statement)
        and not prohibited.search(statement)
        and not reduced_check.search(statement)
        and not quoted_validator_error.search(statement)
        for statement in statements
    )


def _plans_active_work_deferral(
    text: str,
    contract: AcceptanceContract,
    iteration: int,
) -> bool:
    """Reject prose that contradicts the manifest's current milestone."""
    # Structured examples may list every suite beside a performance-only note.
    # The manifest validator owns those blocks; this detector evaluates prose.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    defer = re.compile(r"\b(?:defer|deferred|postpone|postponed|push|move)\b", re.I)
    future_name = re.compile(r"\b(?:next|later|future)\s+(?:iteration|iter|round)\b", re.I)
    iteration_number = re.compile(
        r"\b(?:iteration|iter|round)\s*(?:#\s*)?(\d+)\b", re.I
    )
    prohibited = re.compile(
        r"(?:\b(?:never|not|cannot|forbid|forbidden|prohibit|prohibited)\b|"
        r"\b(?:do|does|must|may|should)\s+not\b|禁止|不得|不能|不可)",
        re.I,
    )
    active_terms = {"phase", "capability", "suite"}
    active_terms.update(contract.required_capabilities)
    active_terms.update(contract.correctness_suites)

    for statement in re.split(r"(?<=[.!?。；;])\s+|\n\s*\n", text):
        if not defer.search(statement) or prohibited.search(statement):
            continue
        future = future_name.search(statement) or any(
            int(value) > iteration for value in iteration_number.findall(statement)
        )
        if not future:
            continue
        normalized = statement.casefold().replace("_", " ").replace(".", " ")
        if any(
            term.casefold().replace("_", " ").replace(".", " ") in normalized
            for term in active_terms
        ):
            return True
    return False


def _plans_unsubstantiated_gemm_layout_rewrite(text: str) -> bool:
    """Require address-level evidence before changing a hipBLAS layout."""
    proposes_rewrite = re.compile(
        r"(?:hipblas.{0,240}(?:row[\s_-]*major|column[\s_-]*major).{0,160}"
        r"(?:mismatch|transpose|swap)|"
        r"(?:swap|exchange).{0,80}(?:m\s*(?:/|and)\s*n|m/n).{0,200}hipblas|"
        r"(?:insert|add).{0,100}transpose.{0,120}(?:gemm|hipblas))",
        re.I | re.S,
    )
    if not proposes_rewrite.search(text):
        return False

    normalized = text.casefold()
    parameters = ("transa", "transb", "lda", "ldb", "ldc")
    has_parameters = all(parameter in normalized for parameter in parameters)
    has_dimensions = bool(re.search(r"\bm\s*[,=/].*\bn\s*[,=/].*\bk\b", text, re.I | re.S))
    has_observed_reference = bool(
        re.search(
            r"(?:observed|measured|recorded|reported).{0,180}"
            r"(?:cpu[\s_-]*reference|reference[\s_-]*cpu).{0,120}"
            r"(?:mismatch|error|fail)",
            text,
            re.I | re.S,
        )
    )
    return not (has_parameters and has_dimensions and has_observed_reference)


def extract_runtime_metadata(payload: Any) -> Optional[Dict[str, Any]]:
    """Return the task runtime metadata object from an OpenAI models payload."""
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    first = data[0] if isinstance(data, list) and data and isinstance(data[0], Mapping) else payload
    for key in ("metainfer", "metadata", "meta", "runtime"):
        value = first.get(key) if isinstance(first, Mapping) else None
        if isinstance(value, Mapping):
            return dict(value)
    if isinstance(first, Mapping) and any(key in first for key in ("capabilities", "tp_size", "world_size", "rank")):
        return dict(first)
    return None


def _contains_capability(advertised: set[str], capability_id: str) -> bool:
    aliases = _capability_aliases(capability_id)
    return any(alias.casefold() in advertised for alias in aliases)


def _load_plan_manifest(path: Path) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"planner deliverable plan_manifest.json cannot be read: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"plan_manifest.json is not valid JSON: {exc}"
    if not isinstance(payload, Mapping):
        return None, "plan_manifest.json root must be an object"
    return payload, None


def _prior_milestone_deadlines(
    iter_dir: Path,
    iteration: int,
    contract: AcceptanceContract,
    errors: list[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return the earliest deadlines promised by earlier iteration manifests.

    A planner may pull work forward, but it may not keep moving an unfinished
    suite into the next iteration. Numeric iteration siblings are the durable
    history used by the task-local orchestrator.
    """
    capability_at: Dict[str, int] = {}
    suite_at: Dict[str, int] = {}
    required_caps = set(contract.required_capabilities)
    required_suites = set(contract.correctness_suites)
    seen_paths: set[Path] = set()

    for prior_iteration in range(1, iteration):
        candidates = (
            iter_dir.parent / f"{prior_iteration:03d}" / "plan_manifest.json",
            iter_dir.parent / str(prior_iteration) / "plan_manifest.json",
        )
        prior_path = next(
            (path for path in candidates if path.is_file() and path not in seen_paths),
            None,
        )
        if prior_path is None:
            continue
        seen_paths.add(prior_path)
        prior_manifest, load_error = _load_plan_manifest(prior_path)
        if load_error:
            errors.append(
                f"cannot enforce prior milestone deadlines from {prior_path}: "
                f"{load_error}"
            )
            continue
        assert prior_manifest is not None
        milestones = prior_manifest.get("milestones")
        if not isinstance(milestones, list):
            errors.append(
                f"cannot enforce prior milestone deadlines: {prior_path} has "
                "no valid milestones array"
            )
            continue
        for raw_milestone in milestones:
            if not isinstance(raw_milestone, Mapping):
                continue
            deadline = raw_milestone.get("iteration")
            if not isinstance(deadline, int) or isinstance(deadline, bool):
                continue
            raw_caps = raw_milestone.get("capabilities", [])
            if isinstance(raw_caps, list):
                for capability_id in raw_caps:
                    if isinstance(capability_id, str) and capability_id in required_caps:
                        capability_at[capability_id] = min(
                            deadline, capability_at.get(capability_id, deadline)
                        )
            raw_suites = raw_milestone.get("suites", [])
            if isinstance(raw_suites, list):
                for suite in raw_suites:
                    if isinstance(suite, str) and suite in required_suites:
                        suite_at[suite] = min(deadline, suite_at.get(suite, deadline))

    return capability_at, suite_at


def _manifest_string_list(
    manifest: Mapping[str, Any], field: str, errors: list[str]
) -> Tuple[str, ...]:
    return _mapping_string_list(manifest, field, "plan_manifest.json", errors)


def _mapping_string_list(
    mapping: Mapping[str, Any], field: str, owner: str, errors: list[str]
) -> Tuple[str, ...]:
    value = mapping.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        errors.append(f"{owner}.{field} must be an array of non-empty strings")
        return ()
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        errors.append(f"{owner}.{field} must not contain duplicates")
    return normalized


def _compare_manifest_list(
    errors: list[str], field: str, actual: Sequence[str], expected: Sequence[str]
) -> None:
    if tuple(actual) != tuple(expected):
        errors.append(
            f"plan_manifest.json {field} does not match the milestone contract: "
            f"actual={list(actual)}, expected={list(expected)}"
        )


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.casefold()}
    if isinstance(value, Sequence):
        return {str(item).casefold() for item in value}
    return set()


def _first_number(mapping: Mapping[str, Any], key: str) -> Optional[float]:
    value = mapping.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_executable(path: Path) -> bool:
    try:
        return bool(path.stat().st_mode & 0o111)
    except OSError:
        return False


__all__ = [
    "AcceptanceContract",
    "annotate_development_gate",
    "audit_iteration",
    "compile_suite_results",
    "performance_gate",
    "read_review_verdict",
    "validate_implementation_artifacts",
    "validate_plan_artifacts",
    "validate_runtime_metadata",
]
