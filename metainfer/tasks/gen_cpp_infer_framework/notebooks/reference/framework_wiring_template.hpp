#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace metainfer::reference {

enum class KvCapacityPolicy {
    kPerSequenceAllocation,
    kFullContextPerRequest,
    kSharedTokenBudget,
};

struct FrameworkConfig {
    std::string model_path;
    std::vector<std::uint32_t> device_ordinals;
    std::uint32_t tp_size = 1;
    std::uint32_t max_context = 0;
    std::uint32_t max_active_requests = 1;
    std::uint32_t max_batched_tokens = 1;
    std::uint32_t prefill_chunk_tokens = 1;
    bool paged_kv = false;
    bool continuous_batching = false;
    bool tensor_parallel = false;
    std::uint32_t kv_block_size = 0;
    std::uint32_t kv_total_blocks_per_rank = 0;
    KvCapacityPolicy kv_capacity_policy =
        KvCapacityPolicy::kPerSequenceAllocation;
};

inline std::uint64_t wiring_blocks_for_tokens(
    std::uint64_t tokens, std::uint32_t block_size) {
    if (block_size == 0) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return tokens / block_size + (tokens % block_size != 0 ? 1U : 0U);
}

inline bool validate_framework_config(
    const FrameworkConfig& config, std::string* error = nullptr) {
    const auto fail = [error](const char* message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (config.model_path.empty()) {
        return fail("model_path is required");
    }
    if (config.tp_size == 0 ||
        config.device_ordinals.size() != config.tp_size) {
        return fail("device count must equal tp_size");
    }
    const std::set<std::uint32_t> unique_devices(
        config.device_ordinals.begin(), config.device_ordinals.end());
    if (unique_devices.size() != config.device_ordinals.size()) {
        return fail("device ordinals must be unique");
    }
    if (config.tensor_parallel != (config.tp_size > 1)) {
        return fail("tensor_parallel must agree with tp_size");
    }
    if (config.max_context == 0 || config.max_active_requests == 0 ||
        config.max_batched_tokens == 0 || config.prefill_chunk_tokens == 0 ||
        config.prefill_chunk_tokens > config.max_batched_tokens) {
        return fail("invalid context, request, or token budget");
    }
    if (!config.continuous_batching && config.max_active_requests != 1) {
        return fail("disabled continuous batching requires one active request");
    }
    if (!config.paged_kv) {
        if (config.kv_capacity_policy !=
            KvCapacityPolicy::kPerSequenceAllocation) {
            return fail("non-paged KV must use per-sequence allocation");
        }
        if (config.kv_block_size != 0 ||
            config.kv_total_blocks_per_rank != 0) {
            return fail("disabled paged KV cannot retain block-pool settings");
        }
        return true;
    }
    if (config.kv_capacity_policy ==
        KvCapacityPolicy::kPerSequenceAllocation) {
        return fail("paged KV requires an explicit block capacity policy");
    }
    if (config.kv_block_size == 0 ||
        config.kv_total_blocks_per_rank == 0) {
        return fail("paged KV requires block size and rank-local capacity");
    }
    const std::uint64_t full_context_blocks = wiring_blocks_for_tokens(
        config.max_context, config.kv_block_size);
    if (full_context_blocks > config.kv_total_blocks_per_rank) {
        return fail("rank-local KV cannot hold one full context");
    }
    if (config.kv_capacity_policy ==
            KvCapacityPolicy::kFullContextPerRequest &&
        full_context_blocks * config.max_active_requests >
            config.kv_total_blocks_per_rank) {
        return fail("rank-local KV violates full-context request guarantee");
    }
    return true;
}

struct SequenceSlice {
    std::int64_t sequence_id = -1;
    std::uint32_t token_begin = 0;
    std::uint32_t token_count = 0;
    std::uint32_t past_length = 0;
    bool is_prefill = false;
};

struct LogicalStepPlan {
    std::uint64_t plan_id = 0;
    std::uint32_t max_context = 0;
    std::uint32_t sequence_rows = 0;
    std::vector<std::int32_t> token_ids;
    std::vector<std::uint32_t> positions;
    std::vector<std::uint32_t> token_rows;
    std::vector<std::uint32_t> sample_rows;
    std::vector<SequenceSlice> sequences;
};

inline bool validate_logical_step_plan(
    const LogicalStepPlan& plan, std::string* error = nullptr) {
    const auto fail = [error](const char* message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    const std::size_t tokens = plan.token_ids.size();
    if (plan.plan_id == 0 || plan.max_context == 0 || tokens == 0 ||
        plan.sequence_rows == 0) {
        return fail("plan identity, context, tokens, and rows are required");
    }
    if (plan.positions.size() != tokens || plan.token_rows.size() != tokens) {
        return fail("token, position, and row arrays must have equal length");
    }
    if (plan.sequences.size() != plan.sequence_rows ||
        !std::is_sorted(plan.token_rows.begin(), plan.token_rows.end())) {
        return fail("sequence rows and ordered packed slices must agree");
    }
    for (const std::uint32_t row : plan.token_rows) {
        if (row >= plan.sequence_rows) {
            return fail("token row is out of range");
        }
    }
    std::set<std::uint32_t> sampled;
    for (const std::uint32_t row : plan.sample_rows) {
        if (row >= tokens || !sampled.insert(row).second) {
            return fail("sample row is out of range or duplicated");
        }
    }
    std::set<std::int64_t> sequence_ids;
    std::vector<bool> covered(tokens, false);
    for (std::size_t sequence_row = 0;
         sequence_row < plan.sequences.size(); ++sequence_row) {
        const SequenceSlice& slice = plan.sequences[sequence_row];
        if (slice.sequence_id < 0 || slice.token_count == 0 ||
            !sequence_ids.insert(slice.sequence_id).second) {
            return fail("sequence slice identity is invalid or duplicated");
        }
        if (slice.token_begin > tokens ||
            slice.token_count > tokens - slice.token_begin) {
            return fail("sequence slice exceeds packed token range");
        }
        if (slice.past_length > plan.max_context ||
            slice.token_count > plan.max_context - slice.past_length) {
            return fail("sequence slice exceeds maximum context");
        }
        for (std::uint32_t offset = 0; offset < slice.token_count; ++offset) {
            const std::size_t token_index = slice.token_begin + offset;
            if (covered[token_index] ||
                plan.token_rows[token_index] != sequence_row ||
                plan.positions[token_index] != slice.past_length + offset) {
                return fail("position does not match past length and offset");
            }
            covered[token_index] = true;
        }
    }
    if (std::find(covered.begin(), covered.end(), false) != covered.end()) {
        return fail("sequence slices do not cover every packed token exactly once");
    }
    return true;
}

struct RankBatchSnapshot {
    std::uint64_t plan_id = 0;
    std::uint32_t rank = 0;
    std::uint32_t world_size = 1;
    std::uint32_t block_table_stride = 0;
    std::vector<std::uint32_t> block_tables;
    std::vector<std::uint32_t> past_lengths;
};

inline bool validate_rank_batch_snapshot(
    const LogicalStepPlan& plan,
    const RankBatchSnapshot& snapshot,
    bool paged_kv,
    std::string* error = nullptr) {
    const auto fail = [error](const char* message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };
    if (snapshot.plan_id != plan.plan_id || snapshot.world_size == 0 ||
        snapshot.rank >= snapshot.world_size) {
        return fail("rank snapshot identity does not match logical plan");
    }
    if (!paged_kv) {
        return snapshot.block_tables.empty() &&
               snapshot.block_table_stride == 0;
    }
    if (snapshot.block_table_stride == 0 ||
        snapshot.past_lengths.size() != plan.sequences.size()) {
        return fail("paged snapshot is missing table stride or past lengths");
    }
    const std::size_t rows = plan.sequences.size();
    if (rows > std::numeric_limits<std::size_t>::max() /
            snapshot.block_table_stride ||
        snapshot.block_tables.size() !=
            rows * snapshot.block_table_stride) {
        return fail("paged block table shape is invalid");
    }
    return true;
}

enum class RuntimeResource {
    kModelMetadata,
    kTokenizer,
    kRankWeights,
    kRankStreams,
    kRankKv,
    kCollectives,
    kScheduler,
    kEngineWorker,
    kHttpListener,
};

class InitializationJournal {
public:
    bool acquire(RuntimeResource resource) {
        if (std::find(acquired_.begin(), acquired_.end(), resource) !=
            acquired_.end()) {
            return false;
        }
        acquired_.push_back(resource);
        return true;
    }

    std::vector<RuntimeResource> reverse_release_order() const {
        return std::vector<RuntimeResource>(
            acquired_.rbegin(), acquired_.rend());
    }

    void clear() { acquired_.clear(); }
    std::size_t size() const { return acquired_.size(); }

private:
    std::vector<RuntimeResource> acquired_;
};

enum class TickOutcome {
    kIdle,
    kApplied,
    kPrepareFailed,
    kExecuteFailed,
    kApplyFailed,
};

struct TickHooks {
    std::function<bool(const LogicalStepPlan&, std::string*)> prepare;
    std::function<bool(
        const LogicalStepPlan&,
        std::vector<std::int32_t>*,
        std::string*)> execute;
    std::function<bool(
        const LogicalStepPlan&,
        const std::vector<std::int32_t>&,
        std::string*)> apply;
    std::function<void(const LogicalStepPlan&)> rollback;
};

inline TickOutcome run_transactional_tick(
    const LogicalStepPlan& plan,
    const TickHooks& hooks,
    std::string* error = nullptr) {
    if (plan.token_ids.empty()) {
        return TickOutcome::kIdle;
    }
    if (!validate_logical_step_plan(plan, error) || !hooks.prepare ||
        !hooks.execute || !hooks.apply || !hooks.rollback) {
        return TickOutcome::kPrepareFailed;
    }
    if (!hooks.prepare(plan, error)) {
        hooks.rollback(plan);
        return TickOutcome::kPrepareFailed;
    }
    std::vector<std::int32_t> sampled_tokens;
    if (!hooks.execute(plan, &sampled_tokens, error)) {
        hooks.rollback(plan);
        return TickOutcome::kExecuteFailed;
    }
    if (!hooks.apply(plan, sampled_tokens, error)) {
        hooks.rollback(plan);
        return TickOutcome::kApplyFailed;
    }
    return TickOutcome::kApplied;
}

}  // namespace metainfer::reference
