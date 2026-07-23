"""Capability compilation and frozen-requirement contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from metainfer.tasks.gen_cpp_infer_framework.orchestrator.capabilities import (
    CapabilityResolutionError,
    freeze_resolved_requirements,
    resolve_capabilities,
)
from metainfer.tasks.gen_cpp_infer_framework.orchestrator.acceptance import (
    AcceptanceContract,
    annotate_development_gate,
    audit_iteration,
    compile_suite_results,
    performance_gate,
    read_review_verdict,
    validate_implementation_artifacts,
    validate_plan_artifacts,
    validate_runtime_metadata,
)


FIXTURES = Path(__file__).parent / "fixtures" / "requirements"


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _write_plan_manifest(
    iter_dir: Path,
    req,
    *,
    iteration: int = 1,
    defer_first_iteration: bool = False,
):
    resolved = req.get("resolved_requirements") or resolve_capabilities(req)
    required_caps = list(resolved["required_capabilities"])
    all_suites = list(resolved["correctness_suites"])
    first_suites = list(resolved["first_iteration_suites"])
    if defer_first_iteration:
        first_caps = required_caps[:2]
        first_suites = first_suites[:2]
    else:
        first_caps = required_caps
    milestones = [{
        "iteration": 1,
        "capabilities": first_caps,
        "suites": first_suites,
        "deliverables": ["runnable vertical slice"],
    }]
    later_caps = [cap for cap in required_caps if cap not in first_caps]
    later_suites = [suite for suite in all_suites if suite not in first_suites]
    if later_caps or later_suites:
        milestones.append({
            "iteration": 2,
            "capabilities": later_caps,
            "suites": later_suites,
            "deliverables": ["remaining correctness coverage"],
        })
    current_caps = [
        cap
        for cap in required_caps
        if any(
            cap in milestone["capabilities"] and milestone["iteration"] <= iteration
            for milestone in milestones
        )
    ]
    current_suites = [
        suite
        for suite in all_suites
        if any(
            suite in milestone["suites"] and milestone["iteration"] <= iteration
            for milestone in milestones
        )
    ]
    manifest = {
        "schema_version": 1,
        "iteration": iteration,
        "required_capabilities": required_caps,
        "current_iteration_capabilities": current_caps,
        "current_suites": current_suites,
        "deferred_suites": [
            suite for suite in all_suites if suite not in current_suites
        ],
        "milestones": milestones,
    }
    (iter_dir / "plan_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_optional_capabilities_are_independent():
    paged = resolve_capabilities(_fixture("paged_only_q8.json"))
    continuous = resolve_capabilities(_fixture("continuous_only_q8.json"))
    tp = resolve_capabilities(_fixture("tp2_f16.json"))

    assert "paged_kv_cache" in paged["required_capabilities"]
    assert "continuous_batching" not in paged["required_capabilities"]
    assert "tensor_parallelism" not in paged["required_capabilities"]

    assert "continuous_batching" in continuous["required_capabilities"]
    assert "paged_kv_cache" not in continuous["required_capabilities"]
    assert "tensor_parallelism" not in continuous["required_capabilities"]

    assert "tensor_parallelism" in tp["required_capabilities"]
    assert "paged_kv_cache" not in tp["required_capabilities"]
    assert "continuous_batching" not in tp["required_capabilities"]


def test_unselected_optional_capabilities_are_disabled():
    resolved = resolve_capabilities(_fixture("base_q8.json"))
    assert resolved["allowed_capabilities"] == []
    assert resolved["disabled_capabilities"] == [
        "paged_kv_cache",
        "continuous_batching",
        "tensor_parallelism",
        "speculative_decoding",
    ]


def test_allowed_capability_is_not_a_success_requirement():
    req = _fixture("base_q8.json")
    req["allowed_features"] = ["Paged KV cache"]
    resolved = resolve_capabilities(req)
    assert resolved["allowed_capabilities"] == ["paged_kv_cache"]
    assert "paged_kv_cache" not in resolved["required_capabilities"]
    assert "paged_kv_cache" not in resolved["disabled_capabilities"]


def test_full_feature_set_activates_only_matching_combinations():
    resolved = resolve_capabilities(_fixture("full_f16.json"))
    assert resolved["active_combination_contracts"] == [
        "paged_kv_with_continuous_batching",
        "tp_with_paged_kv",
        "tp_with_continuous_batching",
        "tp_with_paged_kv_and_continuous_batching",
    ]
    assert "integration.tp_paged_kv" in resolved["correctness_suites"]
    assert "integration.tp_continuous_batching" in resolved["correctness_suites"]
    assert "integration.tp_paged_kv_continuous_batching" in resolved["correctness_suites"]
    assert resolved["first_iteration_suites"] == [
        "base.build_and_lifecycle",
        "base.real_model_loading",
        "base.operator_numerics",
        "base.forward_numerics",
        "base.tokenizer_and_chat_template",
        "base.single_sequence_generation",
        "base.openai_http_api",
        "paged_kv.block_lifecycle",
        "continuous_batching.admission",
        "tensor_parallel.rank_startup",
        "tensor_parallel.sharding",
        "integration.paged_kv_continuous_batching",
        "integration.tp_paged_kv",
        "integration.tp_continuous_batching",
        "integration.tp_paged_kv_continuous_batching",
    ]
    assert resolved["capability_parameters"] == {
        "paged_kv_cache": {
            "block_size": 16,
            "capacity_policy": "full_context_per_request",
            "max_total_cached_tokens": 0,
        },
        "continuous_batching": {"max_concurrency": 4},
        "tensor_parallelism": {"tp_size": 2},
    }


def test_full_context_kv_resource_contract_is_compiled_per_tp_rank():
    resolved = resolve_capabilities(_fixture("full_f16.json"))
    resource = resolved["resource_contract"]

    assert resource["model_contract"] == {
        "num_layers": 36,
        "num_kv_heads": 8,
        "local_kv_heads_per_rank": 4,
        "head_dim": 128,
        "kv_dtype": "fp16",
        "kv_dtype_bytes": 2,
        "tp_size": 2,
    }
    assert resource["max_context_per_request"] == 4096
    assert resource["max_active_requests"] == 4
    assert resource["kv_capacity_policy"] == "full_context_per_request"
    assert resource["kv_blocks_per_full_context_request"] == 256
    assert resource["kv_total_blocks_per_rank"] == 1024
    assert resource["max_total_cached_tokens"] == 16384
    assert resource["guaranteed_full_context_requests"] == 4
    assert resource["kv_bytes_per_token_per_rank"] == 73728
    assert resource["kv_pool_bytes_per_rank"] == 1207959552
    assert resource["required_runtime_metadata"] == [
        "max_context_length",
        "max_concurrency",
        "kv_capacity_policy",
        "kv_block_size",
    ]
    assert resource["advisory_runtime_metadata"] == [
        "max_total_cached_tokens",
        "guaranteed_full_context_requests",
        "kv_bytes_per_token_per_rank",
        "kv_pool_bytes_per_rank",
        "kv_total_blocks",
    ]


def test_shared_kv_budget_exposes_reduced_full_context_guarantee():
    req = _fixture("full_f16.json")
    req["kv_capacity_policy"] = "Shared token budget"
    req["max_total_cached_tokens"] = 8192

    resolved = resolve_capabilities(req)
    resource = resolved["resource_contract"]

    assert resolved["capability_parameters"]["paged_kv_cache"] == {
        "block_size": 16,
        "capacity_policy": "shared_token_budget",
        "max_total_cached_tokens": 8192,
    }
    assert resource["kv_capacity_policy"] == "shared_token_budget"
    assert resource["kv_total_blocks_per_rank"] == 512
    assert resource["max_total_cached_tokens"] == 8192
    assert resource["max_active_requests"] == 4
    assert resource["guaranteed_full_context_requests"] == 2
    assert resource["kv_pool_bytes_per_rank"] == 603979776


def test_shared_kv_budget_cannot_undercut_single_request_context():
    req = _fixture("paged_continuous_q8.json")
    req["kv_capacity_policy"] = "shared_token_budget"
    req["max_context_length"] = 4096
    req["max_total_cached_tokens"] = 4095

    with pytest.raises(
        CapabilityResolutionError,
        match="at least max_context_length",
    ) as exc_info:
        resolve_capabilities(req)
    assert exc_info.value.field == "max_total_cached_tokens"


def test_kv_dtype_follows_dense_vs_scalable_runtime_path():
    base = resolve_capabilities(_fixture("base_q8.json"))["resource_contract"]
    tp_only = resolve_capabilities(_fixture("tp2_f16.json"))["resource_contract"]
    continuous = resolve_capabilities(
        _fixture("continuous_only_q8.json")
    )["resource_contract"]

    assert base["model_contract"]["kv_dtype"] == "fp32"
    assert base["model_contract"]["kv_dtype_bytes"] == 4
    assert tp_only["model_contract"]["kv_dtype"] == "fp32"
    assert continuous["model_contract"]["kv_dtype"] == "fp16"
    assert continuous["model_contract"]["kv_dtype_bytes"] == 2


def test_tp_q8_is_rejected_but_single_card_q8_is_supported():
    base = resolve_capabilities(_fixture("base_q8.json"))
    assert base["parameters"]["weight_format"] == "q8_0_gguf"

    with pytest.raises(CapabilityResolutionError, match="Q8_0 TP") as exc_info:
        resolve_capabilities(_fixture("invalid_tp2_q8.json"))
    assert exc_info.value.field == "weight_format"


@pytest.mark.parametrize("weight_format", ["BF16 GGUF", "Other GGUF"])
def test_unimplemented_weight_formats_are_rejected(weight_format):
    req = _fixture("base_q8.json")
    req["weight_format"] = weight_format
    with pytest.raises(
        CapabilityResolutionError, match="supports F16 or Q8_0 GGUF only"
    ) as exc_info:
        resolve_capabilities(req)
    assert exc_info.value.field == "weight_format"


def test_tp_size_is_limited_to_validated_tp2():
    req = _fixture("tp2_f16.json")
    req["tp_size"] = 4
    with pytest.raises(CapabilityResolutionError, match="tp_size must be <= 2"):
        resolve_capabilities(req)


def test_unavailable_capability_is_rejected_explicitly():
    req = _fixture("base_q8.json")
    req["features"] = ["Speculative decoding"]
    with pytest.raises(CapabilityResolutionError, match="currently unavailable"):
        resolve_capabilities(req)


def test_structured_performance_defaults_are_compiled():
    resolved = resolve_capabilities(_fixture("base_q8.json"))
    assert resolved["performance"] == {
        "objective": "throughput",
        "reference": "vllm",
        "minimum_reference_ratio": 0.5,
        "minimum_tokens_per_second": 0.0,
        "maximum_error_rate": 0.0,
        "concurrency": 4,
        "input_tokens": 128,
        "output_tokens": 128,
        "notes": "",
    }


def test_frozen_requirements_are_deterministic_and_immutable(tmp_path: Path):
    req = _fixture("full_f16.json")
    first = freeze_resolved_requirements(req, tmp_path)
    frozen_bytes = (tmp_path / "resolved_requirements.json").read_bytes()
    second = freeze_resolved_requirements(req, tmp_path)

    assert first == second
    assert (tmp_path / "resolved_requirements.json").read_bytes() == frozen_bytes
    assert "compiled_at" not in first

    changed = dict(req, max_context_length=2048)
    with pytest.raises(CapabilityResolutionError, match="changed after"):
        freeze_resolved_requirements(changed, tmp_path)


def test_legacy_nested_answers_are_supported():
    req = {
        "task_type": "gen-cpp-infer-framework",
        "answers": _fixture("tp2_f16.json"),
    }
    resolved = resolve_capabilities(req)
    assert "tensor_parallelism" in resolved["required_capabilities"]
    assert resolved["capability_parameters"]["tensor_parallelism"]["tp_size"] == 2


def test_orchestrator_records_rejected_requirements_without_starting_a(tmp_path: Path):
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.orchestrator import (
        run_with_requirements,
    )

    req = _fixture("invalid_tp2_q8.json")
    requirements_path = tmp_path / "input.json"
    requirements_path.write_text(json.dumps(req), encoding="utf-8")
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"

    assert run_with_requirements(
        requirements_path,
        state_dir=state_dir,
        workspace_dir=workspace_dir,
    ) == 2
    run = json.loads((state_dir / "run.json").read_text(encoding="utf-8"))
    assert run["finished"] is True
    assert run["final_status"] == "stopped"
    assert "Q8_0 TP" in run["notes"][-1]
    assert not (state_dir / "orchestrator.pid").exists()
    timeline = (state_dir / "timeline.jsonl").read_text(encoding="utf-8")
    assert "requirements_rejected" in timeline


def test_plan_and_artifact_contracts_gate_agent_deliverables(tmp_path: Path):
    req = _fixture("full_f16.json")
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.capabilities import (
        resolve_capabilities,
    )
    resolved = resolve_capabilities(req)
    req["resolved_requirements"] = resolved
    suites = " ".join(resolved["correctness_suites"])
    caps = " ".join(resolved["required_capabilities"])
    (tmp_path / "plan.md").write_text(
        f"# Overall architecture\n# Capability matrix\n{caps}\n"
        f"# Iteration roadmap\n# Test plan\n{suites}\n", encoding="utf-8"
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n"
            "GET /v1/models capabilities tp_size world_size rank device_ids "
            "weight_sharding collective_backend kv_block_size max_concurrency "
            "max_observed_batch_size max_context_length kv_capacity_policy\n",
        encoding="utf-8",
    )
    _write_plan_manifest(tmp_path, req)
    assert validate_plan_artifacts(tmp_path, req) == ()

    (tmp_path / "CMakeLists.txt").write_text("project(test)", encoding="utf-8")
    (tmp_path / "serve.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tmp_path / "serve.sh").chmod(0o755)
    (tmp_path / "include").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.cpp").write_text("int main(){}", encoding="utf-8")
    (tmp_path / "src/engine.cpp").write_text("", encoding="utf-8")
    errors = validate_implementation_artifacts(tmp_path, req)
    assert any("GGUF loader implementation" in error for error in errors)

    (tmp_path / "src/model_loader.cpp").write_text(
        "auto data_offset = align_up(tensor_info_end, general.alignment);\n"
        "auto file_offset = data_offset + tensor.offset;\n",
        encoding="utf-8",
    )
    assert validate_implementation_artifacts(tmp_path, req) == ()


def test_plan_contract_rejects_missing_selected_capability(tmp_path: Path):
    req = _fixture("tp2_f16.json")
    (tmp_path / "plan.md").write_text(
        "# Overall architecture\n# Iteration roadmap\n# Test plan\n",
        encoding="utf-8",
    )
    (tmp_path / "test_spec.md").write_text("# Tests\n", encoding="utf-8")
    errors = validate_plan_artifacts(tmp_path, req)
    assert any("tensor_parallelism" in error for error in errors)


def test_plan_contract_rejects_loader_only_first_iteration(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    caps = " ".join(req["resolved_requirements"]["required_capabilities"])
    suites = " ".join(req["resolved_requirements"]["correctness_suites"])
    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n",
        encoding="utf-8",
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n", encoding="utf-8"
    )
    _write_plan_manifest(tmp_path, req, defer_first_iteration=True)

    errors = validate_plan_artifacts(tmp_path, req, iteration=1)

    assert any("executable paths for every required capability" in error for error in errors)
    assert any("runnable vertical-slice suites" in error for error in errors)
    assert any("collective_backend" in error for error in errors)
    assert any("kv_block_size" in error for error in errors)
    assert any("max_concurrency" in error for error in errors)


def test_tp_plan_rejects_full_model_single_rank_e2e(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    caps = " ".join(req["resolved_requirements"]["required_capabilities"])
    suites = " ".join(req["resolved_requirements"]["correctness_suites"])
    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "Run the full 36-layer target model E2E in single-rank mode on "
        "device 0 only.\n",
        encoding="utf-8",
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n"
        "GET /v1/models capabilities tp_size world_size rank device_ids "
        "weight_sharding collective_backend kv_block_size max_concurrency\n",
        encoding="utf-8",
    )
    _write_plan_manifest(tmp_path, req)

    errors = validate_plan_artifacts(tmp_path, req)

    assert any("full-target-model single-rank/TP1" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "Use reduced synthetic rank-local operator checks. Full target model "
        "E2E validation always runs with TP2.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert not any("full-target-model single-rank/TP1" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "### Validator error: TP plan uses TP1 full-model E2E path\n"
        "Fixed. The plan now explicitly forbids loading complete real weights "
        "on a single device; real-model validation uses TP2.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert not any("full-target-model single-rank/TP1" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "No TP1/single-device full model reference is loaded. "
        "Real-model validation always uses TP2.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert not any("full-target-model single-rank/TP1" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "Test single-rank first on reduced synthetic data.** Before the full "
        "36-layer target model, switch to TP2 and keep all real-weight "
        "validation on TP2.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert not any("full-target-model single-rank/TP1" in error for error in errors)


def test_tp_plan_rejects_real_weights_tp1_but_allows_prohibition(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    resolved = req["resolved_requirements"]
    caps = " ".join(resolved["required_capabilities"])
    suites = " ".join(resolved["correctness_suites"])
    metadata = (
        "GET /v1/models capabilities tp_size world_size rank device_ids "
        "weight_sharding collective_backend kv_block_size max_concurrency\n"
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n{metadata}",
        encoding="utf-8",
    )
    _write_plan_manifest(tmp_path, req)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "Run TP1 on GPU 0, bypass sharding, and load the full weights on one "
        "device as the numeric parity baseline.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert any("full-target-model single-rank/TP1" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "Never load the full real weights in TP1 or single-rank mode. "
        "Reduced synthetic TP1 checks use bounded tensors; full target E2E "
        "always uses TP2.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert not any("full-target-model single-rank/TP1" in error for error in errors)


def test_tp_plan_rejects_single_gpu_forward_recovery(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    resolved = req["resolved_requirements"]
    caps = " ".join(resolved["required_capabilities"])
    suites = " ".join(resolved["correctness_suites"])
    metadata = (
        "GET /v1/models capabilities tp_size world_size rank device_ids "
        "weight_sharding collective_backend kv_block_size max_concurrency\n"
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n{metadata}",
        encoding="utf-8",
    )
    _write_plan_manifest(tmp_path, req)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "Get a single-GPU forward pass producing correct finite logits before "
        "adding tensor parallelism.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert any("full-target-model single-rank/TP1" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "Run a single-GPU synthetic operator forward pass with a bounded 2 MiB "
        "tensor, then run all real-model paths with TP2.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert not any("full-target-model single-rank/TP1" in error for error in errors)


def test_plan_rejects_unsubstantiated_hipblas_layout_rewrite(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    resolved = req["resolved_requirements"]
    caps = " ".join(resolved["required_capabilities"])
    suites = " ".join(resolved["correctness_suites"])
    metadata = (
        "GET /v1/models capabilities tp_size world_size rank device_ids "
        "weight_sharding collective_backend kv_block_size max_concurrency\n"
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n{metadata}",
        encoding="utf-8",
    )
    _write_plan_manifest(tmp_path, req)

    prefix = f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
    (tmp_path / "plan.md").write_text(
        prefix
        + "hipBLAS returns column-major output while kernels need row-major, so "
        "insert a transpose after every GEMM.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert any("hipBLAS layout rewrite" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        prefix
        + "Observed CPU-reference mismatch error after GEMM. Trace: "
        "transA=T transB=N, M=64,N=2,K=32, lda=32, ldb=32, ldc=64. "
        "The address trace proves a transpose is required.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req)
    assert not any("hipBLAS layout rewrite" in error for error in errors)


def test_plan_prose_cannot_defer_active_phase_but_can_defer_perf(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    resolved = req["resolved_requirements"]
    caps = " ".join(resolved["required_capabilities"])
    suites = " ".join(resolved["correctness_suites"])
    metadata = (
        "GET /v1/models capabilities tp_size world_size rank device_ids "
        "weight_sharding collective_backend kv_block_size max_concurrency\n"
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n{metadata}",
        encoding="utf-8",
    )
    _write_plan_manifest(tmp_path, req, iteration=2)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "If time runs short, defer Phases 3-4 to iteration 3.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req, iteration=2)
    assert any("plan prose may not defer" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "All active suites complete now. Defer performance profiling to iteration 3.\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req, iteration=2)
    assert not any("plan prose may not defer" in error for error in errors)

    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n"
        "All active suites complete now.\n"
        "```json\n"
        '{"suites":{"tensor_parallel.numeric_parity":true},'
        '"perf":{"notes":"performance deferred to iteration 3"}}\n'
        "```\n",
        encoding="utf-8",
    )
    errors = validate_plan_artifacts(tmp_path, req, iteration=2)
    assert not any("plan prose may not defer" in error for error in errors)


def test_plan_contract_cannot_postpone_a_prior_suite_deadline(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["max_iterations"] = 10
    req["resolved_requirements"] = resolve_capabilities(req)
    resolved = req["resolved_requirements"]
    first = tmp_path / "001"
    second = tmp_path / "002"
    first.mkdir()
    second.mkdir()
    _write_plan_manifest(first, req, iteration=1)
    _write_plan_manifest(second, req, iteration=2)

    caps = " ".join(resolved["required_capabilities"])
    suites = " ".join(resolved["correctness_suites"])
    (second / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n",
        encoding="utf-8",
    )
    (second / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n"
        "GET /v1/models capabilities tp_size world_size rank device_ids "
        "weight_sharding collective_backend kv_block_size max_concurrency\n",
        encoding="utf-8",
    )

    manifest_path = second / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    postponed = []
    for milestone in manifest["milestones"]:
        if milestone["iteration"] == 2:
            milestone["iteration"] = 3
            postponed.extend(milestone["suites"])
    manifest["current_suites"] = list(resolved["first_iteration_suites"])
    manifest["deferred_suites"] = postponed
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_plan_artifacts(second, req, iteration=2)

    assert postponed
    assert any(
        "cannot be postponed from iteration 2 to 3" in error
        for error in errors
    )


def test_plan_contract_requires_bounded_failure_evidence(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    resolved = req["resolved_requirements"]
    caps = " ".join(resolved["required_capabilities"])
    suites = " ".join(resolved["correctness_suites"])
    (tmp_path / "plan.md").write_text(
        f"# Architecture\n{caps}\n# Iteration roadmap\n# Test plan\n{suites}\n",
        encoding="utf-8",
    )
    (tmp_path / "test_spec.md").write_text(
        f"# Correctness tests\n{caps}\n{suites}\n"
        "GET /v1/models capabilities tp_size world_size rank device_ids "
        "weight_sharding collective_backend kv_block_size max_concurrency\n",
        encoding="utf-8",
    )
    _write_plan_manifest(tmp_path, req, iteration=1)

    errors = validate_plan_artifacts(
        tmp_path,
        req,
        prior_failure="B timed out after fixed-token output",
    )
    assert any("Failure evidence heading" in error for error in errors)
    assert any("diagnostic artifact" in error for error in errors)
    assert any("tensor offsets are relative" in error for error in errors)

    with (tmp_path / "plan.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "# Failure evidence\n"
            "A bounded search of iter1-implementer.attempt3.events.jsonl "
            "found non-finite logits; root cause remains a hypothesis.\n"
            "For GGUF, compute data_offset = align_up(tensor_info_end, "
            "general.alignment); tensor.offset is relative to the data blob, "
            "so read data_offset + tensor.offset.\n"
        )
    errors = validate_plan_artifacts(
        tmp_path,
        req,
        prior_failure="B timed out after fixed-token output",
    )
    assert not any("Failure evidence" in error for error in errors)
    assert not any("tensor offsets are relative" in error for error in errors)


def test_development_gate_allows_only_deferred_suite_failures(tmp_path: Path):
    req = _fixture("full_f16.json")
    req["max_iterations"] = 10
    req["resolved_requirements"] = resolve_capabilities(req)
    _write_plan_manifest(tmp_path, req)
    resolved = req["resolved_requirements"]
    current = set(resolved["first_iteration_suites"])
    suite_results = [
        {"suite": suite, "passed": suite in current, "evidence": []}
        for suite in resolved["correctness_suites"]
    ]
    report_path = tmp_path / "oracle-report.json"
    report_path.write_text(json.dumps({
        "passed": False,
        "acceptance": {
            "passed": False,
            "required_capabilities": resolved["required_capabilities"],
            "suite_results": suite_results,
        },
    }), encoding="utf-8")

    gate = annotate_development_gate(
        report_path, tmp_path, req, iteration=1
    )
    assert gate["passed"] is True
    assert gate["full_acceptance_passed"] is False
    assert gate["deferred_suites"]

    final_gate = annotate_development_gate(
        report_path, tmp_path, req, iteration=10
    )
    assert final_gate["passed"] is False
    assert final_gate["failed_suites"] == gate["deferred_suites"]


def test_runtime_metadata_is_required_for_selected_distributed_features():
    req = _fixture("tp2_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    good = {
        "data": [{
            "metainfer": {
                "capabilities": ["tensor_parallelism"],
                "tp_size": 2,
                "world_size": 2,
                "rank": 0,
                "device_ids": [0, 1],
                "weight_sharding": "tensor_parallel",
                "collective_backend": "rccl",
            }
        }]
    }
    assert validate_runtime_metadata(good, req) == ()
    good["data"][0]["metainfer"]["weight_sharding"] = "column_row"
    assert validate_runtime_metadata(good, req) == ()
    errors = validate_runtime_metadata(
        {"data": [{"metainfer": {"capabilities": ["tensor_parallelism"], "tp_size": 1}}]},
        req,
    )
    assert any("world_size" in error for error in errors)
    assert any("required TP size 2" in error for error in errors)


def test_runtime_metadata_gates_only_minimal_kv_concurrency_contract():
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    metadata = {
        "capabilities": [
            "tensor_parallelism", "paged_kv_cache", "continuous_batching",
        ],
        "tp_size": 2,
        "world_size": 2,
        "rank": 0,
        "device_ids": [0, 1],
        "weight_sharding": "tensor_parallel",
        "collective_backend": "rccl",
        "kv_block_size": 16,
        "max_context_length": 4096,
        "max_concurrency": 4,
        "kv_capacity_policy": "full_context_per_request",
        "max_observed_batch_size": 4,
    }

    assert validate_runtime_metadata({"data": [{"metainfer": metadata}]}, req) == ()

    diagnostics_are_advisory = dict(metadata)
    diagnostics_are_advisory.update({
        "max_total_cached_tokens": 1,
        "guaranteed_full_context_requests": 0,
        "kv_pool_bytes_per_rank": 1,
        "kv_total_blocks": 1,
    })
    assert validate_runtime_metadata(
        {"data": [{"metainfer": diagnostics_are_advisory}]}, req
    ) == ()

    insufficient = dict(metadata)
    insufficient["max_context_length"] = 2048
    insufficient["max_concurrency"] = 2
    insufficient["kv_capacity_policy"] = "shared_token_budget"
    errors = validate_runtime_metadata(
        {"data": [{"metainfer": insufficient}]}, req
    )
    assert any("max_context_length" in error and "below frozen contract" in error for error in errors)
    assert any("max_concurrency" in error and "below required" in error for error in errors)
    assert any("kv_capacity_policy" in error and "does not match" in error for error in errors)


def test_performance_is_observational_unless_explicitly_enabled():
    req = _fixture("base_q8.json")
    report = {"tokens_per_sec": 0, "errors_total": 4, "num_requests": 4}
    assert performance_gate(req, report)["passed"] is True
    req["enforce_performance"] = True
    gate = performance_gate(req, report)
    assert gate["required"] is True
    assert gate["passed"] is False


def test_review_verdict_parser_requires_explicit_pass(tmp_path: Path):
    review = tmp_path / "review.md"
    review.write_text("**Verdict**: PASS\n", encoding="utf-8")
    assert read_review_verdict(review) == "PASS"
    review.write_text("Verdict: NEEDS_FIX\n", encoding="utf-8")
    assert read_review_verdict(review) == "NEEDS_FIX"
    review.write_text("looks good\n", encoding="utf-8")
    assert read_review_verdict(review) is None


def test_final_audit_requires_oracle_acceptance_report(tmp_path: Path):
    req = _fixture("base_q8.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    iter_dir = tmp_path / "iter"
    logs_dir = tmp_path / "logs"
    iter_dir.mkdir()
    logs_dir.mkdir()
    for path in ("CMakeLists.txt", "serve.sh"):
        (iter_dir / path).write_text("", encoding="utf-8")
    (iter_dir / "serve.sh").chmod(0o755)
    (iter_dir / "include").mkdir()
    (iter_dir / "src").mkdir()
    (iter_dir / "src/main.cpp").write_text("", encoding="utf-8")
    (iter_dir / "src/engine.cpp").write_text("", encoding="utf-8")
    (iter_dir / "src/model_loader.cpp").write_text(
        "auto data_offset = align_up(tensor_info_end, general.alignment);\n"
        "auto file_offset = data_offset + tensor.offset;\n",
        encoding="utf-8",
    )
    record = {
        "iteration": 1,
        "status": "success",
        "phases": {"C_test": {"outcome": "ok"}, "D_review": {"outcome": "ok"}},
    }
    (logs_dir / "oracle-report.json").write_text(
        json.dumps({
            "passed": True,
            "acceptance": {
                "passed": True,
                "required_capabilities": req["resolved_requirements"]["required_capabilities"],
                "suite_results": [
                    {"suite": suite, "passed": True}
                    for suite in req["resolved_requirements"]["correctness_suites"]
                ],
            },
        }), encoding="utf-8"
    )
    audit = audit_iteration(req, iter_dir, logs_dir, record)
    assert audit["passed"] is True


def test_capability_knowledge_references_registered_documents():
    task_dir = Path(__file__).parents[1]
    registry = yaml.safe_load((task_dir / "capabilities.yaml").read_text(encoding="utf-8"))
    routing = yaml.safe_load((task_dir / "notebooks/routing.yaml").read_text(encoding="utf-8"))
    known = set(routing["documents"])
    referenced = set()
    for capability in registry["capabilities"].values():
        for document_ids in capability.get("knowledge", {}).values():
            referenced.update(document_ids)
    for combination in registry.get("combination_contracts", []):
        for document_ids in combination.get("knowledge", {}).values():
            referenced.update(document_ids)
    assert referenced <= known


def test_compiler_validates_only_active_capability_parameters():
    base = _fixture("base_q8.json")
    base["tp_size"] = 1
    assert "tensor_parallelism" not in resolve_capabilities(base)["required_capabilities"]

    tp = _fixture("tp2_f16.json")
    tp["tp_size"] = 1
    with pytest.raises(CapabilityResolutionError, match="tp_size must be >= 2") as exc_info:
        resolve_capabilities(tp)
    assert exc_info.value.field == "tp_size"


def test_dynamic_oracle_probes_follow_selected_capabilities(monkeypatch):
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.oracles import correctness

    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    models = {
        "data": [{
            "metainfer": {
                "capabilities": [
                    "tensor_parallelism", "paged_kv_cache", "continuous_batching",
                ],
                "tp_size": 2,
                "world_size": 2,
                "rank": 0,
                "device_ids": [0, 1],
                "weight_sharding": "tensor_parallel",
                "collective_backend": "rccl",
                "kv_block_size": 16,
                "max_concurrency": 4,
                "max_observed_batch_size": 4,
            }
        }]
    }

    def fake_send(_port, cfg, _timeout=None, **_kwargs):
        return "ok", 200, 0.01, None

    monkeypatch.setattr(correctness, "_send_request", fake_send)
    monkeypatch.setattr(correctness, "_fetch_models_payload", lambda _port: models)
    probes = correctness._run_capability_probes(
        1234, req, models_payload=models, metadata_errors=(),
    )
    ids = {probe.case_id for probe in probes}
    assert ids == {
        "capability-runtime-metadata",
        "capability-paged-kv-long-context",
        "capability-continuous-batching-concurrency",
        "capability-tp-paged-cb-integration",
    }
    assert all(probe.judge_verdict == "pass" for probe in probes)


def test_continuous_batching_probe_rejects_cross_sequence_output(monkeypatch):
    import threading

    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.oracles import correctness

    req = _fixture("continuous_only_q8.json")
    req["resolved_requirements"] = resolve_capabilities(req)

    def fake_send(_port, cfg, _timeout=None, **_kwargs):
        if str(cfg.get("id", "")).startswith("continuous-batching-sequence-"):
            if threading.current_thread() is threading.main_thread():
                return cfg["prompt"], 200, 0.01, None
            return "output leaked from another sequence", 200, 0.01, None
        return "ok", 200, 0.01, None

    monkeypatch.setattr(correctness, "_send_request", fake_send)
    monkeypatch.setattr(
        correctness,
        "_fetch_models_payload",
        lambda _port: {"metainfer": {"max_observed_batch_size": 4}},
    )
    probes = correctness._run_capability_probes(
        1234, req, models_payload={}, metadata_errors=(),
    )

    concurrency = next(
        probe for probe in probes
        if probe.case_id == "capability-continuous-batching-concurrency"
    )
    assert concurrency.judge_verdict == "fail"
    assert "changed under concurrency" in concurrency.judge_reason


def test_continuous_batching_probe_requires_observed_runner_batch(monkeypatch):
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.oracles import correctness

    req = _fixture("continuous_only_q8.json")
    req["resolved_requirements"] = resolve_capabilities(req)

    def fake_send(_port, cfg, _timeout=None, **_kwargs):
        return cfg["prompt"], 200, 0.01, None

    monkeypatch.setattr(correctness, "_send_request", fake_send)
    monkeypatch.setattr(
        correctness,
        "_fetch_models_payload",
        lambda _port: {"metainfer": {"max_observed_batch_size": 1}},
    )
    probes = correctness._run_capability_probes(
        1234, req, models_payload={}, metadata_errors=(),
    )

    concurrency = next(
        probe for probe in probes
        if probe.case_id == "capability-continuous-batching-concurrency"
    )
    assert concurrency.judge_verdict == "fail"
    assert "concurrent HTTP success alone" in concurrency.judge_reason


def test_capability_probe_failure_preserves_http_response_clue():
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.oracles.correctness import (
        _request_errors,
    )

    errors = _request_errors(
        [
            (
                '{"error":{"message":"Invalid JSON: Unterminated string"}}',
                400,
                0.01,
                "HTTP 400",
            )
        ],
        label="concurrent request",
    )
    assert errors == [
        "concurrent request 0: HTTP 400; response="
        "'{\"error\":{\"message\":\"Invalid JSON: Unterminated string\"}}'"
    ]


def test_perf_concurrency_ladder_is_compiled_from_continuous_batching():
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.oracles.perf import (
        _resolve_concurrency_ladder,
    )

    base = _fixture("base_q8.json")
    assert _resolve_concurrency_ladder(base, (1, 4, 16)) == (1,)
    continuous = _fixture("continuous_only_q8.json")
    continuous["resolved_requirements"] = resolve_capabilities(continuous)
    assert _resolve_concurrency_ladder(continuous, (1, 4, 16)) == (1, 4, 8)


def test_every_current_correctness_suite_has_immutable_evidence():
    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    contract = AcceptanceContract.from_request(req)
    results = compile_suite_results(
        contract,
        baseline_passed=True,
        probe_verdicts={
            "numeric-operator-contract": "pass",
            "capability-runtime-metadata": "pass",
            "capability-paged-kv-long-context": "pass",
            "capability-continuous-batching-concurrency": "pass",
            "capability-tp-paged-cb-integration": "pass",
        },
    )
    assert {result["suite"] for result in results} == set(contract.correctness_suites)
    assert all(result["evidence"] for result in results)
    assert all(result["passed"] for result in results)


def test_numeric_report_requires_selected_capability_cases_without_skip():
    from metainfer.tasks.gen_cpp_infer_framework.orchestrator.oracles.correctness import (
        _required_numeric_cases,
        _validate_numeric_report,
    )

    req = _fixture("full_f16.json")
    req["resolved_requirements"] = resolve_capabilities(req)
    required = _required_numeric_cases(req)
    assert "f16_linear" in required
    assert "paged_attention" in required
    assert "packed_sequence_isolation" in required
    assert "kv_capacity_contract" in required
    assert "tp_collective" in required
    report = {
        "passed": True,
        "cases": [{"id": case_id, "passed": True} for case_id in required],
    }
    assert _validate_numeric_report(report, req) == []

    target = next(
        case for case in report["cases"]
        if case["id"] == "packed_sequence_isolation"
    )
    target.update({"passed": False, "skipped": True})
    errors = _validate_numeric_report(report, req)
    assert "required numeric case packed_sequence_isolation skipped" in errors
