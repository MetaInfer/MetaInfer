#pragma once

#include <cstdint>
#include <optional>

namespace metainfer::reference {

struct ShardRange {
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
};

inline std::optional<ShardRange> even_shard(
    std::uint64_t global_count,
    std::uint32_t world_size,
    std::uint32_t rank) {
    if (world_size == 0 || rank >= world_size ||
        global_count % world_size != 0) {
        return std::nullopt;
    }
    const std::uint64_t local = global_count / world_size;
    return ShardRange{local * rank, local};
}

struct TpShape {
    std::uint32_t world_size = 0;
    std::uint32_t rank = 0;
    std::uint32_t attention_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint64_t hidden_size = 0;
    std::uint64_t intermediate_size = 0;
};

inline bool valid_tp_shape(const TpShape& shape) {
    return shape.world_size > 0 && shape.rank < shape.world_size &&
           shape.attention_heads % shape.world_size == 0 &&
           shape.kv_heads % shape.world_size == 0 &&
           shape.hidden_size % shape.world_size == 0 &&
           shape.intermediate_size % shape.world_size == 0;
}

inline std::optional<ShardRange> local_attention_heads(const TpShape& shape) {
    if (!valid_tp_shape(shape)) {
        return std::nullopt;
    }
    return even_shard(shape.attention_heads, shape.world_size, shape.rank);
}

inline std::optional<ShardRange> local_kv_heads(const TpShape& shape) {
    if (!valid_tp_shape(shape)) {
        return std::nullopt;
    }
    return even_shard(shape.kv_heads, shape.world_size, shape.rank);
}

// Column-parallel Q/K/V/Gate/Up slice their output dimension.
inline std::optional<ShardRange> column_parallel_output(
    std::uint64_t output_features, const TpShape& shape) {
    if (!valid_tp_shape(shape)) {
        return std::nullopt;
    }
    return even_shard(output_features, shape.world_size, shape.rank);
}

// Row-parallel O/Down slice their input dimension, then AllReduce outputs.
inline std::optional<ShardRange> row_parallel_input(
    std::uint64_t input_features, const TpShape& shape) {
    if (!valid_tp_shape(shape)) {
        return std::nullopt;
    }
    return even_shard(input_features, shape.world_size, shape.rank);
}

}  // namespace metainfer::reference
