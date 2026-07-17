"""Deterministic validation for C++ planner output before implementation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .capabilities import request_validation_errors


SCHEMA_VERSION = 2
TASK_TYPE = "gen-infer-framework-cpp"
FULL_GATE = "full"
INCREMENTAL_GATE = "incremental"
REQUIRED_ARTIFACTS = (
    "architecture.md",
    "plan.md",
    "test_spec.md",
    "plan_manifest.json",
)
CORE_CAPABILITIES = (
    "native_cpp_runtime",
    "real_model_loading",
    "tokenizer_chat_template",
    "target_device_execution",
    "deterministic_decode",
    "openai_chat_endpoint",
    "long_running_server",
    "graceful_shutdown",
)
REQUIREMENT_KEYS = (
    "target_model",
    "model_family",
    "weight_dtype",
    "kv_cache_dtype",
    "quantization_format",
    "target_hardware",
    "accelerator_backend",
    "assigned_devices",
    "tensor_parallel_size",
    "cpp_standard",
    "build_type",
    "features",
    "perf_target",
    "perf_budget",
    "target_tokens_per_sec",
    "execution_mode",
    "max_wall_time_minutes",
    "extra_notes",
)
_ARCHITECTURE_MARKERS = {
    "native C++ runtime": ("native c++", "c++ runtime"),
    "native build": ("cmake",),
    "model and weight loading": ("model loading", "weight loading", "checkpoint loading"),
    "tokenizer": ("tokenizer",),
    "chat template": ("chat template",),
    "target-device execution": ("device execution", "model runner", "accelerator execution"),
    "scheduler": ("scheduler", "scheduling"),
    "KV cache": ("kv cache", "kv-cache"),
    "decode loop": ("decode",),
    "sampler": ("sampler", "sampling"),
    "deterministic greedy decoding": ("temperature=0", "temperature = 0", "greedy"),
    "seeded stochastic sampling": ("stochastic", "temperature>0", "temperature > 0"),
    "OpenAI service": ("/v1/chat/completions",),
    "server lifecycle": ("lifecycle", "shutdown", "sigterm"),
}
_CORE_ALIASES = {
    "nativecppruntime", "cppruntime", "nativeruntime",
    "modelloading", "realmodelloading", "weightloading", "checkpointloading",
    "tokenizer", "chattemplate", "tokenizerchattemplate",
    "targetdeviceexecution", "deviceexecution", "modelexecution", "inference",
    "decode", "decoding", "deterministicdecode", "sampler", "sampling",
    "openaichatendpoint", "chatendpoint", "v1chatcompletions",
    "server", "longrunningserver", "gracefulshutdown", "lifecycle",
}

_AGENT_TEST_SCRIPT_RE = re.compile(
    r"\b(?:write|create|author)\s+(?:an?\s+)?`?test\.sh`?",
    re.IGNORECASE,
)
_NEGATED_ACTION_RE = re.compile(
    r"(?:"
    r"\b(?:do|does|did|must|should|shall|will|may|can)\s+not\b|"
    r"\bnever\b|"
    r"\bnot\s+to\b|"
    r"\bwithout\b"
    r")[^.;:!?]{0,80}$",
    re.IGNORECASE,
)
_STABLE_TEST_ID_RE = re.compile(
    r"^\s*`?(?P<test_id>(?:"
    r"[a-z][a-z0-9]*(?:[_.-][a-z0-9]+)+|"
    r"[a-z][a-z0-9_.-]*\d[a-z0-9_.-]*"
    r"))`?\s*:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlanGateContext:
    """Select the strict first-delivery gate or a verified delta gate."""

    mode: str
    reason: str
    baseline_iteration: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "baseline_iteration": self.baseline_iteration,
        }


@dataclass(frozen=True)
class PlanGateResult:
    passed: bool
    errors: Tuple[str, ...]
    context: PlanGateContext
    manifest: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_artifacts": list(REQUIRED_ARTIFACTS),
            "gate": self.context.to_dict(),
        }

    def diagnostics(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def resolve_plan_gate_context(
    iter_dir: Path,
    req: Mapping[str, Any],
    iteration: int,
    previous_record: Optional[Mapping[str, Any]] = None,
) -> PlanGateContext:
    """Use an incremental gate only after a durable, C-verified iteration.

    The completion sentinel alone is insufficient because failed iterations are
    also closed cleanly.  Eligibility therefore requires the previous record,
    its C result, the sentinel, and a compatible manifest.
    """
    if iteration <= 1:
        return PlanGateContext(FULL_GATE, "first iteration has no verified baseline")

    baseline = iteration - 1
    if not isinstance(previous_record, Mapping):
        return PlanGateContext(FULL_GATE, "previous iteration record is unavailable")
    if previous_record.get("status") != "success":
        return PlanGateContext(FULL_GATE, "previous iteration did not close successfully")
    if previous_record.get("outcome") != "ok":
        return PlanGateContext(FULL_GATE, "previous C outcome was not ok")
    if previous_record.get("interrupted") is True:
        return PlanGateContext(FULL_GATE, "previous iteration was interrupted")
    phases = previous_record.get("phases")
    c_phase = phases.get("C_test") if isinstance(phases, Mapping) else None
    if not isinstance(c_phase, Mapping) or c_phase.get("outcome") != "ok":
        return PlanGateContext(FULL_GATE, "previous iteration lacks a passing C_test record")

    previous_dir = iter_dir.parent / f"{baseline:03d}"
    if not (previous_dir / ".metainfer-completed").is_file():
        return PlanGateContext(FULL_GATE, "previous iteration lacks its completion sentinel")

    manifest_path = previous_dir / "plan_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return PlanGateContext(FULL_GATE, "previous verified manifest is unavailable or invalid")
    problem = _baseline_manifest_problem(manifest, req, baseline)
    if problem is not None:
        return PlanGateContext(FULL_GATE, problem)

    return PlanGateContext(
        INCREMENTAL_GATE,
        "previous iteration passed C and has durable planning evidence",
        baseline_iteration=baseline,
    )


def validate_plan(
    iter_dir: Path,
    req: Mapping[str, Any],
    iteration: int,
    context: Optional[PlanGateContext] = None,
) -> PlanGateResult:
    """Validate the C++ A-stage contract without interpreting agent intent."""
    if context is None:
        context = resolve_plan_gate_context(iter_dir, req, iteration)
    errors: List[str] = []
    texts: Dict[str, str] = {}
    for filename in REQUIRED_ARTIFACTS[:-1]:
        path = iter_dir / filename
        try:
            texts[filename] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            errors.append(f"missing required planning artifact: {filename}")
        except OSError as exc:
            errors.append(f"cannot read {filename}: {exc}")

    manifest_path = iter_dir / "plan_manifest.json"
    manifest: Optional[Mapping[str, Any]] = None
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            manifest = parsed
        else:
            errors.append("plan_manifest.json root must be a JSON object")
    except FileNotFoundError:
        errors.append("missing required planning artifact: plan_manifest.json")
    except json.JSONDecodeError as exc:
        errors.append(f"plan_manifest.json is invalid JSON: {exc.msg}")
    except OSError as exc:
        errors.append(f"cannot read plan_manifest.json: {exc}")

    _validate_document_text(texts, context, errors)
    if manifest is not None:
        _validate_manifest(manifest, req, iteration, texts, context, errors)
    return PlanGateResult(not errors, tuple(errors), context, manifest)


def _validate_document_text(
    texts: Mapping[str, str], context: PlanGateContext, errors: List[str],
) -> None:
    minimum_sizes = {"architecture.md": 200, "plan.md": 200, "test_spec.md": 120}
    for filename, minimum in minimum_sizes.items():
        text = texts.get(filename)
        if text is not None and len(text.strip()) < minimum:
            errors.append(f"{filename} is too short to satisfy its contract (minimum {minimum} chars)")

    architecture = _normalized_whitespace(texts.get("architecture.md", "").casefold())
    for label, alternatives in _ARCHITECTURE_MARKERS.items():
        if architecture and not any(token in architecture for token in alternatives):
            errors.append(f"architecture.md does not cover {label}")

    plan = texts.get("plan.md", "")
    plan_lower = plan.casefold()
    if context.mode == INCREMENTAL_GATE:
        required_plan_markers = {
            "iteration objective": ("iteration objective", "current objective"),
            "change scope": ("change scope", "changed files"),
            "regression verification": ("regression", "immutable full c"),
        }
    else:
        required_plan_markers = {
            "minimum end-to-end delivery": ("minimum end-to-end", "minimal end-to-end"),
            "real model execution": ("real model", "checkpoint"),
            "OpenAI chat endpoint": ("/v1/chat/completions",),
            "foreground serving entrypoint": ("serve.sh",),
            "native build": ("cmakelists.txt", "cmake"),
            "seeded stochastic sampling": (
                "stochastic", "temperature>0", "temperature > 0",
            ),
        }
    for label, alternatives in required_plan_markers.items():
        if plan and not any(token in plan_lower for token in alternatives):
            errors.append(f"plan.md does not specify {label}")
    errors.extend(_forbidden_plan_errors(plan))

    test_spec = texts.get("test_spec.md", "").casefold()
    for label, alternatives in {
        "immutable C contract": ("immutable",),
        "C Oracle": ("oracle",),
        "serving entrypoint": ("serve.sh",),
        "OpenAI chat endpoint": ("/v1/chat/completions",),
        "response schema": ("choices[0].message.content",),
        "deterministic decoding": ("temperature=0", "temperature = 0"),
        "real checkpoint execution": ("real model", "checkpoint"),
        "native C++ process": ("native c++", "native server"),
    }.items():
        if test_spec and not any(token in test_spec for token in alternatives):
            errors.append(f"test_spec.md does not mirror {label}")
    for label, alternatives in {
        "temperature>0 stochastic mode": ("temperature>0", "temperature > 0"),
        "top_p sampling": ("top_p", "top-p"),
        "seeded sampling": ("seed",),
    }.items():
        if test_spec and not any(token in test_spec for token in alternatives):
            errors.append(f"test_spec.md does not mirror {label}")
    if _proposes_agent_test_script(test_spec):
        errors.append("test_spec.md replaces the immutable Oracle with an agent-authored test.sh")


def _validate_manifest(
    manifest: Mapping[str, Any],
    req: Mapping[str, Any],
    iteration: int,
    texts: Mapping[str, str],
    context: PlanGateContext,
    errors: List[str],
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("task_type") != TASK_TYPE:
        errors.append(f"manifest task_type must be {TASK_TYPE!r}")
    if manifest.get("iteration") != iteration:
        errors.append(f"manifest iteration must be {iteration}")
    if manifest.get("gate_mode") != context.mode:
        errors.append(f"manifest gate_mode must be {context.mode!r}")

    _require_contract(manifest, "architecture", "file", "architecture.md", errors)
    architecture = manifest.get("architecture")
    if isinstance(architecture, dict) and architecture.get("complete") is not True:
        errors.append("manifest architecture.complete must be true")
    _require_contract(manifest, "plan", "file", "plan.md", errors)
    plan_contract = manifest.get("plan")
    expected_goal = (
        "incremental_change_ready_for_oracle"
        if context.mode == INCREMENTAL_GATE
        else "minimum_end_to_end_ready_for_oracle"
    )
    if isinstance(plan_contract, dict) and plan_contract.get("goal") != expected_goal:
        errors.append(f"manifest plan.goal must be {expected_goal!r}")
    _require_contract(manifest, "test_spec", "file", "test_spec.md", errors)
    test_contract = manifest.get("test_spec")
    if isinstance(test_contract, dict) and test_contract.get("oracle") != "immutable_full_c":
        errors.append("manifest test_spec.oracle must be 'immutable_full_c'")

    expected = frozen_requirements(req)
    actual = manifest.get("requirements")
    if not isinstance(actual, dict):
        errors.append("manifest requirements must be an object")
    else:
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"manifest requirements is missing frozen key: {key}")
            elif actual[key] != value:
                errors.append(f"manifest requirement {key!r} does not match the frozen request")

    if context.mode == INCREMENTAL_GATE:
        _validate_incremental_manifest(manifest, texts, context, errors)
    else:
        _validate_full_capabilities(manifest, texts, errors)

    errors.extend(request_validation_errors(dict(req)))
    _validate_requested_features(manifest, req, texts.get("architecture.md", ""), errors)
    _validate_delivery_items(manifest, texts, errors)
    _validate_deferred(manifest.get("deferred"), errors)


def _validate_full_capabilities(
    manifest: Mapping[str, Any], texts: Mapping[str, str], errors: List[str],
) -> None:
    capabilities = manifest.get("core_capabilities")
    plan_lower = texts.get("plan.md", "").casefold()
    if not isinstance(capabilities, dict):
        errors.append("manifest core_capabilities must be an object")
        return
    for capability in CORE_CAPABILITIES:
        spec = capabilities.get(capability)
        if not isinstance(spec, dict):
            errors.append(f"manifest core_capabilities is missing {capability}")
            continue
        if spec.get("status") != "delivered_after_b":
            errors.append(f"core capability {capability} must have status 'delivered_after_b'")
        files = spec.get("files")
        if not _relative_path_list(files, non_empty=True):
            errors.append(f"core capability {capability} must name non-empty relative files")
            continue
        for path in files:
            if str(path).casefold() not in plan_lower:
                errors.append(f"plan.md does not account for {capability} file: {path}")


def _validate_incremental_manifest(
    manifest: Mapping[str, Any],
    texts: Mapping[str, str],
    context: PlanGateContext,
    errors: List[str],
) -> None:
    baseline = context.baseline_iteration
    if baseline is None:
        errors.append("incremental gate requires a verified baseline iteration")
        return
    if manifest.get("inherits_verified_iteration") != baseline:
        errors.append(f"manifest inherits_verified_iteration must be {baseline}")

    objective = manifest.get("iteration_objective")
    if not isinstance(objective, dict):
        errors.append("manifest iteration_objective must be an object")
    else:
        if not str(objective.get("summary") or "").strip():
            errors.append("manifest iteration_objective.summary must be non-empty")
        if objective.get("source") not in {
            "requirements", "failure", "review", "perf_plan", "maintenance",
        }:
            errors.append("manifest iteration_objective.source is invalid")
        capabilities = objective.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities or not all(
            isinstance(item, str) and item.strip() for item in capabilities
        ):
            errors.append("manifest iteration_objective.capabilities must be a non-empty string array")

    scope = manifest.get("change_scope")
    changed_files: List[str] = []
    test_files: List[str] = []
    affected: List[str] = []
    if not isinstance(scope, dict):
        errors.append("manifest change_scope must be an object")
    else:
        raw_changed = scope.get("changed_files")
        raw_tests = scope.get("test_files")
        raw_affected = scope.get("affected_core_capabilities")
        if not _relative_path_list(raw_changed, non_empty=True):
            errors.append("manifest change_scope.changed_files must name non-empty relative files")
        else:
            changed_files = list(raw_changed)
        if not _relative_path_list(raw_tests, non_empty=True):
            errors.append("manifest change_scope.test_files must name non-empty relative files")
        else:
            test_files = list(raw_tests)
        if not isinstance(raw_affected, list) or not raw_affected or not all(
            item in CORE_CAPABILITIES for item in raw_affected
        ):
            errors.append(
                "manifest change_scope.affected_core_capabilities must be a non-empty core-capability array"
            )
        else:
            affected = list(raw_affected)
        if scope.get("regression_oracle") != "immutable_full_c":
            errors.append("manifest change_scope.regression_oracle must be 'immutable_full_c'")

    plan_lower = texts.get("plan.md", "").casefold()
    test_lower = texts.get("test_spec.md", "").casefold()
    for path in changed_files:
        if path.casefold() not in plan_lower:
            errors.append(f"plan.md does not account for changed file: {path}")
    for path in test_files:
        folded = path.casefold()
        if folded not in plan_lower and folded not in test_lower:
            errors.append(f"incremental test plan does not account for test file: {path}")

    core = manifest.get("core_capabilities")
    if not isinstance(core, dict):
        errors.append("manifest core_capabilities must be an object")
        return
    for capability in CORE_CAPABILITIES:
        spec = core.get(capability)
        if not isinstance(spec, dict):
            errors.append(f"manifest core_capabilities is missing {capability}")
            continue
        if capability in affected:
            if spec.get("status") != "delivered_after_b":
                errors.append(
                    f"affected core capability {capability} must have status 'delivered_after_b'"
                )
            files = spec.get("files")
            if not _relative_path_list(files, non_empty=True):
                errors.append(f"affected core capability {capability} must name changed files")
                continue
            for path in files:
                if path not in changed_files:
                    errors.append(
                        f"affected core capability {capability} file is outside change_scope: {path}"
                    )
                if path.casefold() not in plan_lower:
                    errors.append(f"plan.md does not account for {capability} file: {path}")
        else:
            if spec.get("status") != "inherited_verified":
                errors.append(
                    f"unchanged core capability {capability} must have status 'inherited_verified'"
                )
            if spec.get("from_iteration") != baseline:
                errors.append(
                    f"inherited core capability {capability} must reference iteration {baseline}"
                )


def _baseline_manifest_problem(
    manifest: Any, req: Mapping[str, Any], baseline: int,
) -> Optional[str]:
    if not isinstance(manifest, Mapping):
        return "previous verified manifest root is not an object"
    if manifest.get("schema_version") not in {1, SCHEMA_VERSION}:
        return "previous verified manifest has an unsupported schema version"
    if manifest.get("task_type") != TASK_TYPE or manifest.get("iteration") != baseline:
        return "previous verified manifest identity does not match the baseline"
    actual_requirements = manifest.get("requirements")
    if not isinstance(actual_requirements, Mapping):
        return "previous verified manifest lacks frozen requirements"
    for key, value in frozen_requirements(req).items():
        if actual_requirements.get(key) != value:
            return f"previous verified manifest requirement {key!r} changed"
    core = manifest.get("core_capabilities")
    if not isinstance(core, Mapping) or any(
        not isinstance(core.get(capability), Mapping) for capability in CORE_CAPABILITIES
    ):
        return "previous verified manifest lacks complete core-capability evidence"
    return None


def _require_contract(
    manifest: Mapping[str, Any], section: str, key: str, value: str, errors: List[str],
) -> None:
    contract = manifest.get(section)
    if not isinstance(contract, dict):
        errors.append(f"manifest {section} must be an object")
    elif contract.get(key) != value:
        errors.append(f"manifest {section}.{key} must be {value!r}")


def frozen_requirements(req: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the request fields that the planner must reproduce verbatim."""
    expected: Dict[str, Any] = {}
    for key in REQUIREMENT_KEYS:
        value = _req_value(req, key)
        if key == "features":
            expected[key] = list(value) if isinstance(value, (list, tuple)) else []
        elif value not in (None, "", []):
            expected[key] = value
    return expected


def _req_value(req: Mapping[str, Any], key: str) -> Any:
    for container in (req, req.get("form"), req.get("answers")):
        if isinstance(container, Mapping) and key in container:
            return container[key]
    return None


def _validate_requested_features(
    manifest: Mapping[str, Any], req: Mapping[str, Any], architecture: str,
    errors: List[str],
) -> None:
    requested = _req_value(req, "features")
    requested_names = [str(item) for item in requested] if isinstance(requested, (list, tuple)) else []
    actual = manifest.get("requested_features")
    if not isinstance(actual, dict):
        errors.append("manifest requested_features must be an object")
        return
    if set(actual) != set(requested_names):
        errors.append("manifest requested_features keys must exactly match the frozen feature list")
    architecture_lower = _normalized_whitespace(architecture.casefold())
    for feature in requested_names:
        spec = actual.get(feature)
        if not isinstance(spec, dict):
            continue
        if spec.get("architecture") != "covered":
            errors.append(f"requested feature {feature!r} must be covered by the full architecture")
        if spec.get("delivery") not in {"now", "deferred"}:
            errors.append(f"requested feature {feature!r} delivery must be 'now' or 'deferred'")
        if spec.get("delivery") == "deferred" and not str(spec.get("reason") or "").strip():
            errors.append(f"deferred requested feature {feature!r} needs a reason")
        if _normalized_whitespace(feature.casefold()) not in architecture_lower:
            errors.append(f"architecture.md does not mention requested feature: {feature}")


def _validate_delivery_items(
    manifest: Mapping[str, Any], texts: Mapping[str, str], errors: List[str],
) -> None:
    value = manifest.get("delivery_items")
    if not isinstance(value, list) or not value:
        errors.append("manifest delivery_items must be a non-empty array")
        return
    seen = set()
    accounted_files = set()
    plan = texts.get("plan.md", "").casefold()
    test_spec = texts.get("test_spec.md", "").casefold()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"delivery_items[{index}] must be an object")
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            errors.append(f"delivery_items[{index}] must have a stable id")
        elif item_id in seen:
            errors.append(f"duplicate delivery item id: {item_id}")
        seen.add(item_id)
        if not str(item.get("summary") or "").strip():
            errors.append(f"delivery item {item_id or index!r} needs a summary")
        files = item.get("files")
        if not _relative_path_list(files, non_empty=True):
            errors.append(f"delivery item {item_id or index!r} must name relative files")
        else:
            for path in files:
                accounted_files.add(str(path))
                if str(path).casefold() not in plan:
                    errors.append(
                        f"plan.md does not account for delivery item file: {path}"
                    )
        tests = item.get("tests")
        if not isinstance(tests, list) or not any(str(test).strip() for test in tests):
            errors.append(f"delivery item {item_id or index!r} must name tests")
        elif not any(
            _test_spec_accounts_for_test(item_id, str(test), test_spec)
            for test in tests
        ):
            errors.append(
                "test_spec.md does not account for delivery item tests: "
                f"{item_id or index} (expected an exact test description or a "
                "stable TEST-ID mapped to the delivery item)"
            )

    required_files = set()
    capabilities = manifest.get("core_capabilities")
    if isinstance(capabilities, Mapping):
        for spec in capabilities.values():
            if not isinstance(spec, Mapping) or spec.get("status") != "delivered_after_b":
                continue
            files = spec.get("files")
            if _relative_path_list(files, non_empty=True):
                required_files.update(str(path) for path in files)
    scope = manifest.get("change_scope")
    if isinstance(scope, Mapping):
        for key in ("changed_files", "test_files"):
            files = scope.get(key)
            if _relative_path_list(files, non_empty=False):
                required_files.update(str(path) for path in files)
    missing = sorted(required_files - accounted_files)
    if missing:
        errors.append(
            "manifest delivery_items do not account for delivery-scope files: "
            + ", ".join(missing)
        )


def _validate_deferred(value: Any, errors: List[str]) -> None:
    if not isinstance(value, list):
        errors.append("manifest deferred must be an array")
        return
    for item in value:
        capability = item.get("capability") if isinstance(item, dict) else item
        normalized = _normalized(str(capability or ""))
        if not normalized:
            errors.append("manifest deferred entries must name a capability")
        elif normalized in _CORE_ALIASES or any(alias in normalized for alias in _CORE_ALIASES if len(alias) > 7):
            errors.append(f"core capability cannot be deferred: {capability}")


def _forbidden_plan_errors(plan: str) -> Sequence[str]:
    errors: List[str] = []
    for line in plan.splitlines():
        lowered = line.casefold()
        if any(negation in lowered for negation in ("forbidden", "must not", "do not", "without mock", "no mock")):
            continue
        if (
            "test" not in lowered
            and re.search(r"\b(?:implement|ship|use|start with)\s+(?:an?\s+)?(?:mock|stub|placeholder)\b", lowered)
        ):
            errors.append("plan.md proposes a mock/stub/placeholder production path")
        if re.search(r"\b(?:fixed|hardcoded|always return|only return)\b.{0,32}\b503\b", lowered):
            errors.append("plan.md proposes a fixed HTTP 503 response")
        if re.search(r"\b(?:defer|postpone)\b.{0,80}\b(?:model loading|inference|decode|chat endpoint|server)\b", lowered):
            errors.append("plan.md defers a core end-to-end capability")
    return errors


def _proposes_agent_test_script(test_spec: str) -> bool:
    """Reject positive test.sh authoring plans without flagging prohibitions."""
    for match in _AGENT_TEST_SCRIPT_RE.finditer(test_spec):
        clause_start = max(
            test_spec.rfind(delimiter, 0, match.start())
            for delimiter in ".;:!?\n"
        )
        prefix = test_spec[clause_start + 1:match.start()]
        if not _NEGATED_ACTION_RE.search(prefix):
            return True
    return False


def _test_spec_accounts_for_test(
    item_id: str, test: str, test_spec: str,
) -> bool:
    """Match test evidence by exact prose or a stable ID-to-item mapping.

    Markdown tables naturally split a manifest entry such as
    ``P01-T01: configure succeeds`` into separate ID and assertion cells.  The
    old full-string substring check rejected that lossless representation.  A
    leading stable test ID is therefore sufficient when the same logical line
    in ``test_spec.md`` also names the delivery item.  Free-form entries keep
    the stricter exact-description behavior.
    """
    candidate = _normalized_whitespace(test.casefold())
    normalized_spec = _normalized_whitespace(test_spec.casefold())
    if candidate and candidate in normalized_spec:
        return True

    match = _STABLE_TEST_ID_RE.match(test)
    if match is None or not item_id:
        return False
    test_id = match.group("test_id")
    return any(
        _contains_identifier(line, item_id)
        and _contains_identifier(line, test_id)
        for line in test_spec.splitlines()
    )


def _contains_identifier(text: str, identifier: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9]){re.escape(identifier.casefold())}(?![a-z0-9])",
        text.casefold(),
    ) is not None


def _relative_path_list(value: Any, *, non_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not non_empty)
        and all(_safe_relative_path(item) for item in value)
    )


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _normalized_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


__all__ = [
    "CORE_CAPABILITIES",
    "FULL_GATE",
    "INCREMENTAL_GATE",
    "PlanGateContext",
    "PlanGateResult",
    "frozen_requirements",
    "resolve_plan_gate_context",
    "validate_plan",
]
