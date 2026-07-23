#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace metainfer::reference {

enum class ReserveStatus {
    kOk,
    kExhausted,
    kInvalid,
};

struct BlockReservation {
    std::vector<std::uint32_t> blocks;
    bool committed = false;
};

// Reference ownership model: Reserve removes blocks, Commit attaches them to
// a sequence, and Rollback/Release returns them exactly once.
class BlockManager {
public:
    explicit BlockManager(std::uint32_t total_blocks) : total_(total_blocks) {
        free_.reserve(total_blocks);
        for (std::uint32_t index = total_blocks; index > 0; --index) {
            free_.push_back(index - 1);
        }
    }

    std::uint32_t total_blocks() const {
        return total_;
    }

    std::uint32_t free_blocks() const {
        return static_cast<std::uint32_t>(free_.size());
    }

    ReserveStatus reserve_batch(
        const std::vector<std::uint32_t>& blocks_per_sequence,
        std::vector<BlockReservation>* output) {
        if (output == nullptr) {
            return ReserveStatus::kInvalid;
        }
        std::uint64_t needed = 0;
        for (const std::uint32_t blocks : blocks_per_sequence) {
            needed += blocks;
            if (needed > free_.size()) {
                return ReserveStatus::kExhausted;
            }
        }

        std::vector<BlockReservation> staged;
        staged.reserve(blocks_per_sequence.size());
        for (const std::uint32_t count : blocks_per_sequence) {
            BlockReservation reservation;
            reservation.blocks.reserve(count);
            for (std::uint32_t i = 0; i < count; ++i) {
                reservation.blocks.push_back(free_.back());
                free_.pop_back();
            }
            staged.push_back(std::move(reservation));
        }
        *output = std::move(staged);
        return ReserveStatus::kOk;
    }

    static void commit_batch(std::vector<BlockReservation>* reservations) {
        if (reservations == nullptr) {
            return;
        }
        for (auto& reservation : *reservations) {
            reservation.committed = true;
        }
    }

    void rollback_batch(std::vector<BlockReservation>* reservations) {
        release_batch(reservations, false);
    }

    void release_batch(std::vector<BlockReservation>* reservations) {
        release_batch(reservations, true);
    }

private:
    void release_batch(
        std::vector<BlockReservation>* reservations, bool include_committed) {
        if (reservations == nullptr) {
            return;
        }
        std::vector<BlockReservation> retained;
        for (auto& reservation : *reservations) {
            if (!include_committed && reservation.committed) {
                retained.push_back(std::move(reservation));
                continue;
            }
            for (const std::uint32_t block : reservation.blocks) {
                free_.push_back(block);
            }
            reservation.blocks.clear();
            reservation.committed = false;
        }
        *reservations = std::move(retained);
    }

    std::uint32_t total_ = 0;
    std::vector<std::uint32_t> free_;
};

enum class CapacityPolicy {
    kFullContextPerRequest,
    kSharedTokenBudget,
};

struct CapacityContract {
    std::uint32_t max_context = 0;
    std::uint32_t max_active_requests = 0;
    std::uint32_t block_size = 0;
    std::uint32_t total_blocks = 0;
    CapacityPolicy policy = CapacityPolicy::kFullContextPerRequest;
};

inline std::uint64_t blocks_for_tokens(
    std::uint64_t tokens, std::uint32_t block_size) {
    if (block_size == 0) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return tokens / block_size + (tokens % block_size != 0 ? 1 : 0);
}

inline bool valid_capacity_contract(const CapacityContract& contract) {
    if (contract.max_context == 0 || contract.max_active_requests == 0 ||
        contract.block_size == 0 || contract.total_blocks == 0) {
        return false;
    }
    const std::uint64_t per_request = blocks_for_tokens(
        contract.max_context, contract.block_size
    );
    if (per_request > contract.total_blocks) {
        return false;
    }
    if (contract.policy == CapacityPolicy::kSharedTokenBudget) {
        return true;
    }
    return per_request * contract.max_active_requests <= contract.total_blocks;
}

inline bool can_admit(
    const CapacityContract& contract,
    std::uint32_t active_requests,
    std::uint64_t prompt_tokens,
    std::uint64_t maximum_new_tokens,
    std::uint32_t free_blocks) {
    if (!valid_capacity_contract(contract) ||
        active_requests >= contract.max_active_requests ||
        maximum_new_tokens >
            std::numeric_limits<std::uint64_t>::max() - prompt_tokens) {
        return false;
    }
    const std::uint64_t total_tokens = prompt_tokens + maximum_new_tokens;
    if (total_tokens > contract.max_context) {
        return false;
    }
    return blocks_for_tokens(total_tokens, contract.block_size) <= free_blocks;
}

}  // namespace metainfer::reference
