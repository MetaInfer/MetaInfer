"""Compile and execute task-owned C++ reference templates."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def test_reference_headers_compile_and_enforce_contracts(tmp_path: Path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("no host C++ compiler is available")

    source = tmp_path / "reference_contract_test.cpp"
    binary = tmp_path / "reference_contract_test"
    source.write_text(
        r'''
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "reference/framework_wiring_template.hpp"
#include "reference/gguf_loader_template.hpp"
#include "reference/numeric_harness_template.hpp"
#include "reference/scheduler_block_manager_template.hpp"
#include "reference/tp_sharding_template.hpp"

int main() {
    namespace ref = metainfer::reference;

    const auto range = ref::tensor_file_range(65, 32, 4, 20, 128);
    assert(range && range->offset == 100 && range->size == 20);
    assert(!ref::tensor_file_range(65, 32, 40, 20, 128));

    ref::BlockManager blocks(8);
    assert(blocks.total_blocks() == 8 && blocks.free_blocks() == 8);
    std::vector<ref::BlockReservation> reservations;
    assert(blocks.reserve_batch({5, 4}, &reservations) ==
           ref::ReserveStatus::kExhausted);
    assert(blocks.free_blocks() == 8 && reservations.empty());
    assert(blocks.reserve_batch({3, 4}, &reservations) ==
           ref::ReserveStatus::kOk);
    assert(blocks.free_blocks() == 1 && reservations.size() == 2);
    ref::BlockManager::commit_batch(&reservations);
    blocks.release_batch(&reservations);
    assert(blocks.free_blocks() == 8);

    assert(blocks.reserve_batch({2, 2}, &reservations) ==
           ref::ReserveStatus::kOk);
    reservations[0].committed = true;
    blocks.rollback_batch(&reservations);
    assert(blocks.free_blocks() == 6 && reservations.size() == 1);
    blocks.release_batch(&reservations);
    assert(blocks.free_blocks() == 8);

    ref::CapacityContract capacity{
        16, 2, 4, 8, ref::CapacityPolicy::kFullContextPerRequest
    };
    assert(ref::valid_capacity_contract(capacity));
    assert(ref::can_admit(capacity, 1, 8, 8, 4));
    assert(!ref::can_admit(capacity, 1, 16, 1, 8));
    ref::CapacityContract invalid_shared{
        16, 4, 4, 3, ref::CapacityPolicy::kSharedTokenBudget
    };
    assert(!ref::valid_capacity_contract(invalid_shared));

    ref::TpShape tp{2, 1, 32, 8, 4096, 12288};
    assert(ref::valid_tp_shape(tp));
    const auto q_heads = ref::local_attention_heads(tp);
    const auto kv_heads = ref::local_kv_heads(tp);
    const auto mlp = ref::column_parallel_output(12288, tp);
    assert(q_heads && q_heads->begin == 16 && q_heads->count == 16);
    assert(kv_heads && kv_heads->begin == 4 && kv_heads->count == 4);
    assert(mlp && mlp->begin == 6144 && mlp->count == 6144);

    ref::FrameworkConfig baseline;
    baseline.model_path = "/models/baseline.gguf";
    baseline.device_ordinals = {3};
    baseline.tp_size = 1;
    baseline.max_context = 4096;
    baseline.max_active_requests = 1;
    baseline.max_batched_tokens = 256;
    baseline.prefill_chunk_tokens = 128;
    assert(ref::validate_framework_config(baseline));
    baseline.kv_block_size = 16;
    assert(!ref::validate_framework_config(baseline));
    baseline.kv_block_size = 0;

    ref::FrameworkConfig tp4;
    tp4.model_path = "/models/tp4.gguf";
    tp4.device_ordinals = {0, 2, 4, 6};
    tp4.tp_size = 4;
    tp4.max_context = 8192;
    tp4.max_active_requests = 8;
    tp4.max_batched_tokens = 1024;
    tp4.prefill_chunk_tokens = 256;
    tp4.paged_kv = true;
    tp4.continuous_batching = true;
    tp4.tensor_parallel = true;
    tp4.kv_block_size = 32;
    tp4.kv_total_blocks_per_rank = 2048;
    tp4.kv_capacity_policy = ref::KvCapacityPolicy::kFullContextPerRequest;
    assert(ref::validate_framework_config(tp4));
    tp4.device_ordinals.pop_back();
    assert(!ref::validate_framework_config(tp4));
    tp4.device_ordinals = {0, 2, 4, 4};
    assert(!ref::validate_framework_config(tp4));

    ref::LogicalStepPlan plan;
    plan.plan_id = 7;
    plan.max_context = 32;
    plan.sequence_rows = 2;
    plan.token_ids = {10, 11, 12};
    plan.positions = {0, 1, 4};
    plan.token_rows = {0, 0, 1};
    plan.sample_rows = {1, 2};
    plan.sequences = {
        ref::SequenceSlice{100, 0, 2, 0, true},
        ref::SequenceSlice{200, 2, 1, 4, false},
    };
    assert(ref::validate_logical_step_plan(plan));
    plan.token_rows = {0, 1, 0};
    assert(!ref::validate_logical_step_plan(plan));
    plan.token_rows = {0, 0, 1};

    ref::RankBatchSnapshot snapshot;
    snapshot.plan_id = 7;
    snapshot.rank = 3;
    snapshot.world_size = 4;
    snapshot.block_table_stride = 2;
    snapshot.block_tables = {9, 10, 40, 41};
    snapshot.past_lengths = {0, 4};
    assert(ref::validate_rank_batch_snapshot(plan, snapshot, true));

    ref::InitializationJournal journal;
    assert(journal.acquire(ref::RuntimeResource::kModelMetadata));
    assert(journal.acquire(ref::RuntimeResource::kRankStreams));
    assert(!journal.acquire(ref::RuntimeResource::kRankStreams));
    const auto release_order = journal.reverse_release_order();
    assert(release_order.size() == 2);
    assert(release_order[0] == ref::RuntimeResource::kRankStreams);

    int prepare_calls = 0;
    int execute_calls = 0;
    int apply_calls = 0;
    int rollback_calls = 0;
    ref::TickHooks hooks;
    hooks.prepare = [&](const ref::LogicalStepPlan&, std::string*) {
        ++prepare_calls;
        return true;
    };
    hooks.execute = [&execution_calls = execute_calls](
        const ref::LogicalStepPlan&,
        std::vector<std::int32_t>* sampled,
        std::string*) {
        ++execution_calls;
        *sampled = {21, 22};
        return true;
    };
    hooks.apply = [&](const ref::LogicalStepPlan&,
                      const std::vector<std::int32_t>& sampled,
                      std::string*) {
        ++apply_calls;
        return sampled.size() == 2;
    };
    hooks.rollback = [&](const ref::LogicalStepPlan&) { ++rollback_calls; };
    assert(ref::run_transactional_tick(plan, hooks) ==
           ref::TickOutcome::kApplied);
    assert(prepare_calls == 1 && execute_calls == 1 && apply_calls == 1);
    assert(rollback_calls == 0);

    hooks.execute = [&](const ref::LogicalStepPlan&,
                        std::vector<std::int32_t>*, std::string*) {
        ++execute_calls;
        return false;
    };
    assert(ref::run_transactional_tick(plan, hooks) ==
           ref::TickOutcome::kExecuteFailed);
    assert(apply_calls == 1 && rollback_calls == 1);

    ref::NumericFeatures numeric_features;
    numeric_features.weight_format = ref::NumericWeightFormat::kQ8_0;
    numeric_features.paged_kv = true;
    numeric_features.continuous_batching = true;
    numeric_features.tensor_parallel = true;
    const auto required_cases = ref::required_numeric_case_ids(numeric_features);
    assert(std::find(required_cases.begin(), required_cases.end(),
                     "dequant_q8_0") != required_cases.end());
    assert(std::find(required_cases.begin(), required_cases.end(),
                     "kv_capacity_contract") != required_cases.end());
    ref::NumericFeatures f16_features;
    const auto f16_cases = ref::required_numeric_case_ids(f16_features);
    assert(std::find(f16_cases.begin(), f16_cases.end(), "f16_linear") !=
           f16_cases.end());
    assert(std::find(f16_cases.begin(), f16_cases.end(), "dequant_q8_0") ==
           f16_cases.end());

    ref::NumericHarness harness;
    for (const std::string& id : required_cases) {
        assert(harness.add(id, [id] {
            return ref::NumericCaseResult{id, true, "ok"};
        }));
    }
    const auto numeric_report = harness.run_required(numeric_features);
    assert(numeric_report.passed);
    assert(numeric_report.cases.size() == required_cases.size());
    assert(ref::numeric_report_json(numeric_report).find(
        "\"passed\":true") != std::string::npos);

    ref::NumericHarness incomplete_harness;
    const auto incomplete = incomplete_harness.run_required(numeric_features);
    assert(!incomplete.passed);
    assert(incomplete.cases.size() == required_cases.size());
    return 0;
}
''',
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(NOTEBOOKS),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr

    executed = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert executed.returncode == 0, executed.stderr
