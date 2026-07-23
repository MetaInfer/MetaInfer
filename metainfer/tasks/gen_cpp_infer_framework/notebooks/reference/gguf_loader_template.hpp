#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace metainfer::reference {

struct FileRange {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

inline std::optional<std::uint64_t> checked_add(
    std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return std::nullopt;
    }
    return left + right;
}

inline std::optional<std::uint64_t> align_up(
    std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0) {
        return std::nullopt;
    }
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return checked_add(value, alignment - remainder);
}

// GGUF tensor offsets are relative to the aligned tensor-data blob.
inline std::optional<FileRange> tensor_file_range(
    std::uint64_t tensor_info_end,
    std::uint64_t general_alignment,
    std::uint64_t relative_tensor_offset,
    std::uint64_t tensor_bytes,
    std::uint64_t file_bytes) {
    const auto data_base = align_up(tensor_info_end, general_alignment);
    if (!data_base) {
        return std::nullopt;
    }
    const auto absolute = checked_add(*data_base, relative_tensor_offset);
    if (!absolute || *absolute > file_bytes ||
        tensor_bytes > file_bytes - *absolute) {
        return std::nullopt;
    }
    return FileRange{*absolute, tensor_bytes};
}

}  // namespace metainfer::reference
