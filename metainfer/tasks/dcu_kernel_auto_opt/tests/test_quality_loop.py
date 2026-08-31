from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ..orchestrator.api_contracts import (
    OperatorAPIContract,
    stage_operator_references,
)
from ..orchestrator.config import WorkerAssignment
from ..orchestrator.experience_store import load_verified_experience
from ..orchestrator.pmc_profile import (
    add_unprofiled_bandwidth,
    parse_memory_traffic_csv,
    parse_pmc_csv,
)
from ..orchestrator.w8a8_pipeline import (
    RealW8A8OptimizationPipeline,
    _ISA_AGENT_ARGS,
    _SOURCE_ONLY_AGENT_ARGS,
    evaluate_candidate_acceptance,
    evaluate_final_target,
    isa_round_policy,
    is_infrastructure_failure,
    phase_extension_reason,
    pmc_profile_decision,
    validate_skill_draft,
)
from ..orchestrator.prompts import (
    split_k_candidate_set,
    w8a8_round_strategy,
)


def test_pmc_parser_selects_last_w8a8_dispatch(tmp_path):
    path = tmp_path / "pmc.csv"
    fields = [
        "KernelName", "gpu-id", "grd", "wgr", "lds", "scr",
        "arch_vgpr", "accum_vgpr", "sgpr", "wave_size",
        "GRBM_COUNT", "GRBM_GUI_ACTIVE", "SQ_ACTIVE_INST_VALU",
        "SQ_INSTS_FLAT_LDS_ONLY", "SQ_INSTS_LDS", "SQ_INSTS_VALU",
        "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR",
        "SQ_LDS_BANK_CONFLICT", "SQ_WAIT_INST_LDS",
        "TCC_HIT[0]", "TCC_MISS[0]", "BeginNs", "EndNs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "KernelName": "unrelated", "BeginNs": "1", "EndNs": "2",
        })
        writer.writerow({
            "KernelName": "w8a8_gemm_kernel",
            "gpu-id": "0",
            "grd": "3072",
            "wgr": "128",
            "lds": "1024",
            "scr": "0",
            "arch_vgpr": "40",
            "accum_vgpr": "0",
            "sgpr": "48",
            "wave_size": "64",
            "GRBM_COUNT": "200",
            "GRBM_GUI_ACTIVE": "100",
            "SQ_ACTIVE_INST_VALU": "12",
            "SQ_INSTS_FLAT_LDS_ONLY": "1",
            "SQ_INSTS_LDS": "2",
            "SQ_INSTS_VALU": "3",
            "SQ_INSTS_VMEM_RD": "4",
            "SQ_INSTS_VMEM_WR": "5",
            "SQ_LDS_BANK_CONFLICT": "6",
            "SQ_WAIT_INST_LDS": "7",
            "TCC_HIT[0]": "90",
            "TCC_MISS[0]": "10",
            "BeginNs": "1000",
            "EndNs": "51000",
        })

    evidence = parse_pmc_csv(path)

    assert evidence["kernel_name"] == "w8a8_gemm_kernel"
    assert evidence["grid_blocks"] == 24
    assert evidence["profiled_duration_us"] == 50.0
    assert evidence["l2_hit_rate_percent"] == 90.0
    assert evidence["gpu_active_percent"] == 50.0
    assert evidence["counters"]["lds_bank_conflicts"] == 6


def test_pmc_parser_keeps_partial_and_combine_in_one_replay(tmp_path):
    path = tmp_path / "pmc.csv"
    fields = [
        "KernelName", "grd", "wgr", "lds", "arch_vgpr", "sgpr",
        "TCC_HIT[0]", "TCC_MISS[0]", "BeginNs", "EndNs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"KernelName": "w8a8_old", "grd": 64, "wgr": 64})
        writer.writerow({"KernelName": "unrelated"})
        writer.writerow({
            "KernelName": "w8a8_splitk_partial", "grd": 30720,
            "wgr": 128, "lds": 10240, "arch_vgpr": 40, "sgpr": 27,
            "TCC_HIT[0]": 30, "TCC_MISS[0]": 10,
            "BeginNs": 1000, "EndNs": 12000,
        })
        writer.writerow({
            "KernelName": "w8a8_combine", "grd": 24576,
            "wgr": 256, "lds": 0, "arch_vgpr": 16, "sgpr": 38,
            "TCC_HIT[0]": 20, "TCC_MISS[0]": 20,
            "BeginNs": 12000, "EndNs": 17000,
        })

    evidence = parse_pmc_csv(path)

    assert evidence["primary_kernel_name"] == "w8a8_splitk_partial"
    assert evidence["grid_blocks"] == 240
    assert evidence["lds_bytes"] == 10240
    assert evidence["operator_aggregate"]["kernel_count"] == 2
    assert evidence["operator_aggregate"]["profiled_duration_us"] == 16.0
    assert evidence["operator_aggregate"]["l2_hit_rate_percent"] == 62.5


def test_memory_traffic_uses_dtk_request_size_formulas(tmp_path):
    fields = [
        "KernelName",
        "TCC_EA_RDREQ[0]", "TCC_EA_RDREQ_32B[0]",
        "TCC_EA1_RDREQ[0]", "TCC_EA1_RDREQ_32B[0]",
        "TCC_EA_WRREQ[0]", "TCC_EA_WRREQ_64B[0]",
        "TCC_EA1_WRREQ[0]", "TCC_EA1_WRREQ_64B[0]",
    ]
    read_path = tmp_path / "pmc-read.csv"
    write_path = tmp_path / "pmc-write.csv"
    for path, row in (
        (read_path, {
            "KernelName": "w8a8_gemm_kernel",
            "TCC_EA_RDREQ[0]": "10",
            "TCC_EA_RDREQ_32B[0]": "4",
            "TCC_EA1_RDREQ[0]": "3",
            "TCC_EA1_RDREQ_32B[0]": "1",
        }),
        (write_path, {
            "KernelName": "w8a8_gemm_kernel",
            "TCC_EA_WRREQ[0]": "5",
            "TCC_EA_WRREQ_64B[0]": "2",
            "TCC_EA1_WRREQ[0]": "4",
            "TCC_EA1_WRREQ_64B[0]": "1",
        }),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    evidence = {
        "memory_traffic": parse_memory_traffic_csv(
            read_path, write_path
        )
    }
    add_unprofiled_bandwidth(evidence, 2.0)

    traffic = evidence["memory_traffic"]
    assert traffic["read_bytes_per_dispatch"] == 672
    assert traffic["write_bytes_per_dispatch"] == 384
    assert traffic["total_bytes_per_dispatch"] == 1056
    assert traffic["counter_derived_hbm_bandwidth_gb_s"] == 0.528


def test_memory_traffic_aggregates_multi_kernel_operator(tmp_path):
    fields = [
        "KernelName", "TCC_EA_RDREQ[0]", "TCC_EA_RDREQ_32B[0]",
        "TCC_EA1_RDREQ[0]", "TCC_EA1_RDREQ_32B[0]",
        "TCC_EA_WRREQ[0]", "TCC_EA_WRREQ_64B[0]",
        "TCC_EA1_WRREQ[0]", "TCC_EA1_WRREQ_64B[0]",
    ]
    read_path = tmp_path / "read.csv"
    write_path = tmp_path / "write.csv"
    for path, counter in ((read_path, "TCC_EA_RDREQ[0]"),
                          (write_path, "TCC_EA_WRREQ[0]")):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"KernelName": "unrelated"})
            writer.writerow({"KernelName": "w8a8_partial", counter: 10})
            writer.writerow({"KernelName": "w8a8_combine", counter: 2})

    traffic = parse_memory_traffic_csv(read_path, write_path)

    assert traffic["kernel_names"] == ["w8a8_partial", "w8a8_combine"]
    assert traffic["kernel_count"] == 2
    assert traffic["total_bytes_per_operator_replay"] == 1152
    assert traffic["kernels"][0]["total_bytes"] == 960
    assert traffic["kernels"][1]["total_bytes"] == 192


def test_memory_traffic_marker_excludes_graph_validation_replay(tmp_path):
    fields = [
        "KernelName", "TCC_EA_RDREQ[0]", "TCC_EA_RDREQ_32B[0]",
        "TCC_EA1_RDREQ[0]", "TCC_EA1_RDREQ_32B[0]",
        "TCC_EA_WRREQ[0]", "TCC_EA_WRREQ_64B[0]",
        "TCC_EA1_WRREQ[0]", "TCC_EA1_WRREQ_64B[0]",
    ]
    read_path = tmp_path / "read.csv"
    write_path = tmp_path / "write.csv"
    for path, counter in ((read_path, "TCC_EA_RDREQ[0]"),
                          (write_path, "TCC_EA_WRREQ[0]")):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"KernelName": "w8a8_partial", counter: 10})
            writer.writerow({"KernelName": "w8a8_combine", counter: 2})
            writer.writerow({"KernelName": "vectorized_elementwise_marker"})
            writer.writerow({"KernelName": "w8a8_partial", counter: 10})
            writer.writerow({"KernelName": "w8a8_combine", counter: 2})

    traffic = parse_memory_traffic_csv(read_path, write_path)

    assert traffic["kernel_names"] == ["w8a8_partial", "w8a8_combine"]
    assert traffic["kernel_count"] == 2
    assert traffic["total_bytes_per_operator_replay"] == 1152


def test_memory_traffic_records_tiny_counter_replay_skew(tmp_path):
    fields = [
        "KernelName", "TCC_EA_RDREQ[0]", "TCC_EA_RDREQ_32B[0]",
        "TCC_EA_WRREQ[0]", "TCC_EA_WRREQ_64B[0]",
    ]
    read_path = tmp_path / "read.csv"
    write_path = tmp_path / "write.csv"
    for path, row in (
        (read_path, {
            "KernelName": "w8a8_gemm", "TCC_EA_RDREQ[0]": 100,
            "TCC_EA_RDREQ_32B[0]": 101,
        }),
        (write_path, {
            "KernelName": "w8a8_gemm", "TCC_EA_WRREQ[0]": 200,
            "TCC_EA_WRREQ_64B[0]": 201,
        }),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    traffic = parse_memory_traffic_csv(read_path, write_path)

    assert traffic["read_bytes_per_operator_replay"] == 101 * 32
    assert traffic["write_bytes_per_operator_replay"] == 201 * 64
    assert len(traffic["counter_reconciliation"]) == 2


def test_memory_traffic_rejects_large_counter_mismatch(tmp_path):
    fields = [
        "KernelName", "TCC_EA_RDREQ[0]", "TCC_EA_RDREQ_32B[0]",
        "TCC_EA_WRREQ[0]", "TCC_EA_WRREQ_64B[0]",
    ]
    read_path = tmp_path / "read.csv"
    write_path = tmp_path / "write.csv"
    for path, row in (
        (read_path, {
            "KernelName": "w8a8_gemm", "TCC_EA_RDREQ[0]": 100,
            "TCC_EA_RDREQ_32B[0]": 120,
        }),
        (write_path, {"KernelName": "w8a8_gemm"}),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    with pytest.raises(ValueError, match="exceeds TCC_EA_RDREQ"):
        parse_memory_traffic_csv(read_path, write_path)


def test_repair_prompt_exposes_four_repairs_and_mismatch():
    prompt = RealW8A8OptimizationPipeline._repair_prompt(
        assignment=WorkerAssignment("worker_0", 0, ["m2"]),
        shape_id="m2",
        shape={"M": 2, "N": 16, "K": 32},
        root=Path("/tmp/worker"),
        iteration=1,
        repair=4,
        error="exact correctness failed",
        metrics={
            "mismatch_count": 1,
            "first_mismatch": {
                "m": 0, "n": 3, "actual": 2.0, "expected": 3.0,
            },
        },
        pmc_evidence={"available": True, "arch_vgpr": 40},
    )

    assert "repair 4/4" in prompt
    assert '"mismatch_count": 1' in prompt
    assert "Preserve the proposed" in prompt


def test_worker_prompt_receives_pmc_evidence():
    prompt = RealW8A8OptimizationPipeline._worker_prompt(
        WorkerAssignment("worker_0", 0, ["m2"]),
        "m2",
        {"M": 2, "N": 16, "K": 32},
        {"median_us": 10.0},
        Path("/tmp/worker"),
        1,
        None,
        [],
        {
            "available": True,
            "kernel_name": "w8a8_gemm_kernel",
            "arch_vgpr": 40,
            "isa": {
                "available": True,
                "instruction_counts": {"mmac": 8, "waitcnt": 4},
            },
        },
    )

    assert "Trusted PMC evidence" in prompt
    assert '"arch_vgpr": 40' in prompt
    assert '"mmac": 8' not in prompt
    assert "hygon-gfx928-memory-isa" not in prompt
    assert "hygon-gfx928-compute-isa" not in prompt
    assert "Skill tool is disabled" in prompt
    assert "not measured HBM traffic" in prompt


def test_pmc_policy_skips_scalar_and_unchanged_best():
    hip_policy = {"skill_allowed": False}
    initial = pmc_profile_decision(
        iteration=1,
        history=[],
        source_uses_dumma=False,
        isa_policy=hip_policy,
    )
    unchanged = pmc_profile_decision(
        iteration=3,
        history=[{"accepted": False}],
        source_uses_dumma=True,
        isa_policy=hip_policy,
    )

    assert initial["profile"] is False
    assert "scalar bootstrap" in initial["reason"]
    assert unchanged["profile"] is False
    assert "unchanged" in unchanged["reason"]


def test_pmc_policy_profiles_new_best_and_late_isa():
    new_best = pmc_profile_decision(
        iteration=2,
        history=[{"accepted": True}],
        source_uses_dumma=True,
        isa_policy={"skill_allowed": False},
    )
    late_isa = pmc_profile_decision(
        iteration=9,
        history=[{"accepted": False}],
        source_uses_dumma=True,
        isa_policy={"skill_allowed": True},
    )

    assert new_best["profile"] is True
    assert late_isa["profile"] is True


def test_continuation_prompt_is_incremental_and_compact():
    prompt = RealW8A8OptimizationPipeline._worker_prompt(
        WorkerAssignment("worker_0", 0, ["m16"]),
        "m16",
        {"M": 16, "N": 1024, "K": 4096},
        {
            "median_us": 10.0,
            "p90_us": 10.2,
            "latency_samples_us": list(range(30)),
        },
        Path("/tmp/worker"),
        4,
        None,
        [{
            "iteration": 3,
            "accepted": False,
            "metrics": {
                "median_us": 10.1,
                "latency_samples_us": list(range(30)),
            },
        }],
        {"available": False, "skipped": True},
        continuation=True,
    )

    assert "Continue the existing shape-specialized" in prompt
    assert "immutable API" in prompt
    assert "latency_samples_us" not in prompt
    assert "do not\nreread unchanged scaffold files" in prompt.lower()
    assert "experiments.jsonl" in prompt


def test_m16_rounds_keep_architecture_search_after_iteration_five():
    shape = {"M": 16, "N": 1024, "K": 4096}

    split_k = w8a8_round_strategy(shape, 3, [], {
        "grid_blocks": 64,
        "device_cu_count": 120,
    })
    packed = w8a8_round_strategy(shape, 5, [])
    pipeline = w8a8_round_strategy(shape, 6, [])
    isa = w8a8_round_strategy(shape, 9, [])

    assert "split-K=2" in split_k
    assert "64 blocks for 120 CUs" in split_k
    assert "HIP-only packed-weight" in packed
    assert "A-only LDS" in pipeline
    assert "Late ISA-diagnosis" in isa


def test_m16_near_one_block_per_cu_triggers_parallelism_warning():
    strategy = w8a8_round_strategy(
        {"M": 16, "N": 4096, "K": 2048},
        3,
        [],
        {"grid_blocks": 128, "device_cu_count": 120},
    )

    assert "two-blocks-per-CU" in strategy
    assert "split-K=2" in strategy


def test_split_k_candidates_include_non_power_cu_aligned_probe():
    candidates = split_k_candidate_set(
        {"M": 16, "N": 1536, "K": 4096},
        cu_count=120,
        max_split=16,
    )

    assert 10 in candidates
    assert 2 in candidates
    assert all(2 <= split <= 16 for split in candidates)


def test_optional_variant_is_staged_read_only_under_references(tmp_path):
    api = tmp_path / "int8_w8a8_gemm_api.py"
    variant = tmp_path / "w8a8_gemm_variants.hip"
    api.write_text("# api\n", encoding="utf-8")
    variant.write_text("// evidence, not policy\n", encoding="utf-8")
    contract = OperatorAPIContract(
        operator="Quantized GEMM",
        dtype="INT8 W8A8",
        source=api,
        destination_name=api.name,
        reference_sources=(variant,),
    )

    staged = stage_operator_references(contract, tmp_path / "repo")

    assert [path.relative_to(tmp_path / "repo").as_posix() for path in staged] == [
        "references/w8a8_gemm_variants.hip"
    ]
    assert staged[0].read_text(encoding="utf-8") == "// evidence, not policy\n"
    assert staged[0].stat().st_mode & 0o222 == 0


def _valid_history(iteration, improvement, **extra):
    record = {
        "iteration": iteration,
        "build_success": True,
        "correctness_passed": True,
        "metrics": {"graph_capture_passed": True},
        "acceptance": {"improvement_percent": improvement},
    }
    record.update(extra)
    return record


def test_isa_policy_requires_eight_hip_rounds_and_plateau():
    history = [
        _valid_history(index, improvement, isa_policy={"phase": "hip_only"})
        for index, improvement in enumerate(
            [5.0, 3.0, 1.5, 1.0, 0.8, 0.8, -0.2, 1.4],
            start=1,
        )
    ]

    early = isa_round_policy(
        iteration=8, max_iterations=10, history=history[:7]
    )
    late = isa_round_policy(
        iteration=9, max_iterations=10, history=history
    )

    assert early["skill_allowed"] is False
    assert late["phase"] == "isa_guided_hip"
    assert late["skill_allowed"] is True
    assert late["raw_inline_asm_allowed"] is False
    assert "Skill" in _SOURCE_ONLY_AGENT_ARGS[1].split(",")
    assert "Skill" not in _ISA_AGENT_ARGS[1].split(",")


def test_isa_policy_does_not_count_failed_attempts_as_hip_rounds():
    history = [
        _valid_history(index, 0.5, isa_policy={"phase": "hip_only"})
        for index in range(1, 8)
    ]
    history.append({
        "iteration": 8,
        "build_success": False,
        "correctness_passed": False,
        "failure_reason": "killed after timeout",
        "isa_policy": {"phase": "hip_only"},
    })

    policy = isa_round_policy(
        iteration=9, max_iterations=10, history=history
    )

    assert policy["skill_allowed"] is False
    assert policy["valid_hip_rounds"] == 7


def test_timeout_is_an_infrastructure_failure():
    assert is_infrastructure_failure("worker failed: killed") is True
    assert is_infrastructure_failure("nonzero exit 143") is True
    assert is_infrastructure_failure("exact correctness failed") is False


def test_timeout_strategy_does_not_repair_partial_candidate():
    strategy = w8a8_round_strategy(
        {"M": 16, "N": 4096, "K": 2048},
        4,
        [{
            "iteration": 3,
            "build_success": False,
            "failure_reason": "worker failed: killed after timeout",
            "artifact_dir": "iterations/m16/iteration3",
        }],
        {},
    )

    assert "agent infrastructure" in strategy
    assert "do not repair or replay" in strategy


def test_late_isa_prompt_exposes_only_selected_isa_skills():
    policy = {
        "phase": "isa_guided_hip",
        "skill_allowed": True,
        "raw_inline_asm_allowed": False,
        "plateau": True,
        "reason": "test plateau",
    }
    prompt = RealW8A8OptimizationPipeline._worker_prompt(
        WorkerAssignment("worker_0", 0, ["m16"]),
        "m16",
        {"M": 16, "N": 1024, "K": 4096},
        {"median_us": 10.0},
        Path("/tmp/worker"),
        9,
        None,
        [],
        {"available": True, "isa": {"available": True}},
        isa_policy=policy,
    )

    assert "hygon-gfx928-memory-isa" in prompt
    assert "hygon-gfx928-compute-isa" in prompt
    assert "Raw inline asm remains forbidden" in prompt
    assert '"isa_optimization"' in prompt


def test_raw_asm_needs_two_isa_rounds_and_prior_confirmed_limitation():
    history = [
        *[
            _valid_history(
                index,
                0.4,
                isa_policy={"phase": "hip_only"},
            )
            for index in range(1, 9)
        ],
        _valid_history(
            9,
            0.2,
            isa_policy={"phase": "isa_guided_hip"},
            isa_optimization={
                "compiler_limitation_confirmed": False,
                "target_instructions": [],
            },
            candidate_isa={"available": True},
        ),
        _valid_history(
            10,
            0.1,
            isa_policy={"phase": "isa_guided_hip"},
            isa_optimization={
                "compiler_limitation_confirmed": True,
                "target_instructions": ["s_waitcnt vmcnt(0)"],
            },
            candidate_isa={"available": True},
        ),
    ]

    policy = isa_round_policy(
        iteration=11, max_iterations=10, history=history
    )

    assert policy["phase"] == "conditional_inline_asm"
    assert policy["raw_inline_asm_allowed"] is True
    assert policy["verified_target_instructions"] == ["s_waitcnt vmcnt(0)"]


def test_one_isa_round_does_not_open_raw_asm():
    history = [
        *[
            _valid_history(
                index, 0.4, isa_policy={"phase": "hip_only"}
            )
            for index in range(1, 9)
        ],
        _valid_history(
            9,
            0.2,
            isa_policy={"phase": "isa_guided_hip"},
            isa_optimization={
                "compiler_limitation_confirmed": True,
                "target_instructions": ["s_waitcnt vmcnt(0)"],
            },
            candidate_isa={"available": True},
        ),
    ]

    policy = isa_round_policy(
        iteration=10, max_iterations=10, history=history
    )

    assert policy["phase"] == "isa_guided_hip"
    assert policy["raw_inline_asm_allowed"] is False
    assert policy["valid_isa_guided_rounds"] == 1


def test_large_regressions_do_not_prove_plateau():
    history = [
        _valid_history(
            index, improvement, isa_policy={"phase": "hip_only"}
        )
        for index, improvement in enumerate(
            [4.0, 3.0, 1.0, 0.5, 0.2, -21.0, -19.0, -22.0],
            start=1,
        )
    ]

    policy = isa_round_policy(
        iteration=9, max_iterations=10, history=history
    )

    assert policy["phase"] == "hip_only"
    assert policy["plateau"] is False
    assert "Large regressions" in policy["reason"]


def test_phase_extension_reserves_two_valid_isa_rounds():
    hip_history = [
        _valid_history(index, 0.4, isa_policy={"phase": "hip_only"})
        for index in range(1, 9)
    ]
    one_isa = _valid_history(
        9,
        0.2,
        isa_policy={"phase": "isa_guided_hip"},
        isa_optimization={"compiler_limitation_confirmed": False},
        candidate_isa={"available": True},
    )

    reason = phase_extension_reason(
        max_iterations=10, history=[*hip_history, one_isa]
    )

    assert reason == "need 1 more valid ISA-guided HIP experiment(s)"


def test_final_target_is_measured_against_fixed_baseline():
    result = evaluate_final_target(
        baseline={"m16": {"median_us": 100.0}},
        validation={
            "m16": {"passed": True, "metrics": {"median_us": 80.0}}
        },
        target_improvement_percent=20.0,
    )

    assert result["shapes"]["m16"]["improvement_percent"] == 25.0
    assert result["all_shapes_met"] is True


def test_acceptance_rejects_median_gain_with_p90_regression():
    result = evaluate_candidate_acceptance(
        passed=True,
        metrics={"median_us": 90.0, "p90_us": 130.0},
        best_metrics={"median_us": 100.0, "p90_us": 120.0},
        minimum_improvement_percent=2.0,
    )

    assert result["improvement_percent"] > 2.0
    assert result["p90_guard_passed"] is False
    assert result["accepted"] is False


def test_acceptance_requires_both_median_and_p90():
    result = evaluate_candidate_acceptance(
        passed=True,
        metrics={"median_us": 90.0, "p90_us": 110.0},
        best_metrics={"median_us": 100.0, "p90_us": 120.0},
        minimum_improvement_percent=2.0,
    )

    assert result["accepted"] is True


def test_subthreshold_gain_can_become_shadow_without_updating_best():
    result = evaluate_candidate_acceptance(
        passed=True,
        metrics={"median_us": 99.4, "p90_us": 118.0},
        best_metrics={"median_us": 100.0, "p90_us": 120.0},
        minimum_improvement_percent=1.0,
    )

    assert result["accepted"] is False
    assert result["shadow_eligible"] is True


def test_shadow_candidate_must_improve_existing_shadow():
    result = evaluate_candidate_acceptance(
        passed=True,
        metrics={"median_us": 99.5, "p90_us": 118.0},
        best_metrics={"median_us": 100.0, "p90_us": 120.0},
        shadow_metrics={"median_us": 99.4, "p90_us": 118.0},
        minimum_improvement_percent=1.0,
    )

    assert result["accepted"] is False
    assert result["shadow_eligible"] is False
    assert result["improves_shadow"] is False


def test_verified_experience_ignores_skill_prose_and_filters_shape(tmp_path):
    repo = tmp_path / "old-task"
    record_path = (
        repo / "candidates" / "worker_0" / "iteration1"
        / "iteration.json"
    )
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps({
        "shape": {"M": 2, "N": 16, "K": 32},
        "shape_id": "m2",
        "iteration": 1,
        "hypothesis": "vector load",
        "build_success": True,
        "correctness_passed": False,
        "accepted": False,
        "speedup": 1.5,
        "metrics": {"median_us": 5.0, "mismatch_count": 1},
    }), encoding="utf-8")

    evidence = load_verified_experience(
        tmp_path, {"M": 2, "N": 16, "K": 32}
    )

    assert len(evidence) == 1
    assert evidence[0]["classification"] == "faster_incorrect_repairable"
    assert evidence[0]["speedup"] == 1.5


def test_skill_validator_rejects_hardware_claim_from_compile_failure():
    facts = [{
        "accepted": False,
        "build_success": False,
        "correctness_passed": False,
    }]

    with pytest.raises(ValueError, match="unsupported conclusions"):
        validate_skill_draft(
            "No candidate was accepted. DUMMA is unavailable on this GPU.",
            facts,
        )
