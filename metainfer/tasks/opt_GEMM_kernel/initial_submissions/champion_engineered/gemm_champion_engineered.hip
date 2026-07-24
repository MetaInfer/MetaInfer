#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// =========================================================================
// Split-K SDOT4 kernel — small-M, large-K W8A8 GEMM with cross-K parallelism
// =========================================================================
// BM=8, BN=64, BK=32, 512 threads (8 Wave64).
// Each workgroup processes one K-slice and writes INT32 partial output.
// A second reduction kernel sums across splits and applies BF16 scale+convert.
//
// Constraints: M ≤ 16, N % 16 == 0, K % 16 == 0, K % (split_k * BK) == 0.

struct alignas(16) GlobalVec128 { int32_t dwords[4]; };


// Register-only K100/gfx928 path for (M<=8, N=4096, K=256).
// A Wave64 computes four adjacent columns. Each lane owns four adjacent K
// values. W is loaded as four int32 vectors along N, transposed in registers,
// accumulated with SDOT4, and reduced with native AMD DPP instructions.
__device__ __forceinline__ int32_t pack_weight_column(
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, int byte_index) {
    const int shift = byte_index * 8;
    return static_cast<int32_t>(
        ((w0 >> shift) & 0xffu) |
        (((w1 >> shift) & 0xffu) << 8) |
        (((w2 >> shift) & 0xffu) << 16) |
        (((w3 >> shift) & 0xffu) << 24));
}

__device__ __forceinline__ int32_t wave64_reduce_sum_dpp(int32_t value) {
    value += __builtin_amdgcn_mov_dpp(value, 0x111, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x112, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x114, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x118, 0xf, 0xf, true);
    value += __builtin_amdgcn_mov_dpp(value, 0x142, 0xf, 0xf, false);
    value += __builtin_amdgcn_mov_dpp(value, 0x143, 0xf, 0xf, false);
    return value; // full Wave64 sum is in lane 63
}

__global__ __launch_bounds__(256) void wave_dpp_sdot4_k256_n4096_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ w,
    const float* __restrict__ a_scale,
    const float* __restrict__ w_scale,
    hip_bfloat16* __restrict__ output,
    int M, int N, int K) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int col0 = static_cast<int>(blockIdx.x) * 16 + wave * 4;
    const int k0 = lane * 4;

    // W does not depend on M: load its 4x4 byte fragment once per wave lane
    // and reuse the register-transposed SDOT4 operands for every input row.
    const uint32_t w0 = static_cast<uint32_t>(*reinterpret_cast<const int32_t*>(
        w + static_cast<int64_t>(k0 + 0) * N + col0));
    const uint32_t w1 = static_cast<uint32_t>(*reinterpret_cast<const int32_t*>(
        w + static_cast<int64_t>(k0 + 1) * N + col0));
    const uint32_t w2 = static_cast<uint32_t>(*reinterpret_cast<const int32_t*>(
        w + static_cast<int64_t>(k0 + 2) * N + col0));
    const uint32_t w3 = static_cast<uint32_t>(*reinterpret_cast<const int32_t*>(
        w + static_cast<int64_t>(k0 + 3) * N + col0));
    const int32_t wp0 = pack_weight_column(w0, w1, w2, w3, 0);
    const int32_t wp1 = pack_weight_column(w0, w1, w2, w3, 1);
    const int32_t wp2 = pack_weight_column(w0, w1, w2, w3, 2);
    const int32_t wp3 = pack_weight_column(w0, w1, w2, w3, 3);

    // M is at most eight. Keeping the row loop inside the wave avoids
    // reloading W M times while retaining 1024 independent waves over N.
#pragma unroll
    for (int row = 0; row < M; ++row) {
        const int32_t a_pack = *reinterpret_cast<const int32_t*>(
            a + static_cast<int64_t>(row) * K + k0);
        int32_t acc0 = __builtin_amdgcn_sdot4(a_pack, wp0, 0, false);
        int32_t acc1 = __builtin_amdgcn_sdot4(a_pack, wp1, 0, false);
        int32_t acc2 = __builtin_amdgcn_sdot4(a_pack, wp2, 0, false);
        int32_t acc3 = __builtin_amdgcn_sdot4(a_pack, wp3, 0, false);
        acc0 = wave64_reduce_sum_dpp(acc0);
        acc1 = wave64_reduce_sum_dpp(acc1);
        acc2 = wave64_reduce_sum_dpp(acc2);
        acc3 = wave64_reduce_sum_dpp(acc3);

        if (lane == 63) {
            const float row_scale = a_scale[row];
            const int64_t out = static_cast<int64_t>(row) * N + col0;
            output[out + 0] = hip_bfloat16(
                static_cast<float>(acc0) * row_scale * w_scale[col0 + 0]);
            output[out + 1] = hip_bfloat16(
                static_cast<float>(acc1) * row_scale * w_scale[col0 + 1]);
            output[out + 2] = hip_bfloat16(
                static_cast<float>(acc2) * row_scale * w_scale[col0 + 2]);
            output[out + 3] = hip_bfloat16(
                static_cast<float>(acc3) * row_scale * w_scale[col0 + 3]);
        }
    }
}

template <int BM, int BN, int BK>
__global__ __launch_bounds__(512) void small_m_splitk_dot4_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ w,
    int32_t* __restrict__ partial,
    int M, int N, int K, int split_k) {

    constexpr int BK_PAD = BK + 4;
    const int split = static_cast<int>(blockIdx.z);
    const int span = K / split_k;
    const int k_begin = split * span;
    const int k_end = k_begin + span;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * BN + tx;
    constexpr int threads = BM * BN;

    const int row = static_cast<int>(blockIdx.y) * BM + ty;
    const int col = static_cast<int>(blockIdx.x) * BN + tx;

    __shared__ __align__(16) int8_t a_tile[BM][BK];
    __shared__ __align__(16) int8_t b_tile[BN][BK_PAD];

    int32_t acc = 0;

    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
        // A: 128-bit loads along K dimension (coalesced, contiguous)
        constexpr int A_VECS = BM * BK / 16;  // 8*32/16 = 16 vecs
        for (int vi = tid; vi < A_VECS; vi += threads) {
            const int lm = vi / (BK / 16);
            const int lk = (vi % (BK / 16)) * 16;
            const int gm = static_cast<int>(blockIdx.y) * BM + lm;
            const int gk = k0 + lk;
            GlobalVec128 value{};
            if (gm < M && gk + 16 <= K &&
                (reinterpret_cast<uintptr_t>(
                    a + static_cast<int64_t>(gm) * K + gk) & 15u) == 0) {
                value = *reinterpret_cast<const GlobalVec128*>(
                    a + static_cast<int64_t>(gm) * K + gk);
            } else if (gm < M && gk < K) {
                const int valid = (K - gk) < 16 ? (K - gk) : 16;
#pragma unroll
                for (int j = 0; j < valid; ++j)
                    reinterpret_cast<int8_t*>(&value)[j] =
                        a[static_cast<int64_t>(gm) * K + gk + j];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j)
                a_tile[lm][lk + j] = reinterpret_cast<const int8_t*>(&value)[j];
        }

        // B: 128-bit loads along N dimension, stored transposed into LDS
        // Global W[K,N] → LDS b_tile[N][K_PAD] so SDOT4 consumes K-contiguous bytes
        constexpr int B_VECS = BK * BN / 16;  // 32*64/16 = 128 vecs
        for (int vi = tid; vi < B_VECS; vi += threads) {
            const int lk = vi / (BN / 16);
            const int ln = (vi % (BN / 16)) * 16;
            const int gk = k0 + lk;
            const int gn = static_cast<int>(blockIdx.x) * BN + ln;
            GlobalVec128 value{};
            if (gk < K && gn + 16 <= N &&
                (reinterpret_cast<uintptr_t>(
                    w + static_cast<int64_t>(gk) * N + gn) & 15u) == 0) {
                value = *reinterpret_cast<const GlobalVec128*>(
                    w + static_cast<int64_t>(gk) * N + gn);
            } else if (gk < K && gn < N) {
                const int valid = (N - gn) < 16 ? (N - gn) : 16;
#pragma unroll
                for (int j = 0; j < valid; ++j)
                    reinterpret_cast<int8_t*>(&value)[j] =
                        w[static_cast<int64_t>(gk) * N + gn + j];
            }
            // Transpose: B's N elements scatter across LDS rows
#pragma unroll
            for (int j = 0; j < 16; ++j)
                b_tile[ln + j][lk] = reinterpret_cast<const int8_t*>(&value)[j];
        }
        __syncthreads();

        if (row < M && col < N) {
#pragma unroll
            for (int kk = 0; kk < BK; kk += 4) {
                const int a_pack =
                    *reinterpret_cast<const int32_t*>(&a_tile[ty][kk]);
                const int b_pack =
                    *reinterpret_cast<const int32_t*>(&b_tile[tx][kk]);
                acc = __builtin_amdgcn_sdot4(a_pack, b_pack, acc, false);
            }
        }
        __syncthreads();
    }

    if (row < M && col < N)
        partial[(static_cast<int64_t>(split) * M + row) * N + col] = acc;
}

// =========================================================================
// Reduction kernel — INT32 sum across splits, then FP32 scale → BF16 write
// =========================================================================

__global__ void reduce_splitk_scale_kernel(
    const int32_t* __restrict__ partial,
    const float* __restrict__ a_scale,
    const float* __restrict__ w_scale,
    hip_bfloat16* __restrict__ output,
    int M, int N, int split_k) {

    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(M) * N;
    if (idx >= elements) return;

    int32_t sum = 0;
#pragma unroll 1
    for (int s = 0; s < split_k; ++s)
        sum += partial[static_cast<int64_t>(s) * elements + idx];

    const int row = static_cast<int>(idx / N);
    const int col = static_cast<int>(idx - static_cast<int64_t>(row) * N);
    output[idx] =
        hip_bfloat16(static_cast<float>(sum) * a_scale[row] * w_scale[col]);
}

// =========================================================================
// Legacy kernels — correctness-first baselines, preserved as fallbacks
// =========================================================================

// Decode/small-M baseline. One block computes one row and BLOCK_N columns.
template <int BLOCK_N, int BLOCK_K>
__global__ void w8a8_scaled_gemv_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight_kn,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ output,
    int M, int N, int K) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int col = static_cast<int>(blockIdx.x) * BLOCK_N + tid;
    if (row >= M) return;

    __shared__ __align__(16) int8_t a_tile[BLOCK_K];
    const int64_t a_row = static_cast<int64_t>(row) * K;
    int32_t acc = 0;
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        const int global_k = k0 + tid;
        a_tile[tid] = global_k < K ? x_q[a_row + global_k] : int8_t{0};
        __syncthreads();
        const int valid_k = (K - k0) < BLOCK_K ? (K - k0) : BLOCK_K;
        if (col < N) {
            for (int kk = 0; kk < valid_k; ++kk) {
                acc += static_cast<int>(a_tile[kk]) * static_cast<int>(
                    weight_kn[static_cast<int64_t>(k0 + kk) * N + col]);
            }
        }
        __syncthreads();
    }
    if (col < N) {
        const float y = static_cast<float>(acc) * x_scale[row] * weight_scale[col];
        output[static_cast<int64_t>(row) * N + col] = hip_bfloat16(y);
    }
}

// General-M correctness-first baseline. Each thread computes one output.
template <int BM, int BN, int BK>
__global__ void w8a8_scaled_gemm_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight_kn,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ output,
    int M, int N, int K) {
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * BN + tx;
    const int threads = BM * BN;
    const int row = static_cast<int>(blockIdx.y) * BM + ty;
    const int col = static_cast<int>(blockIdx.x) * BN + tx;

    __shared__ __align__(16) int8_t a_tile[BM][BK];
    __shared__ __align__(16) int8_t b_tile[BK][BN];
    int32_t acc = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += threads) {
            const int lm = i / BK;
            const int lk = i % BK;
            const int gm = static_cast<int>(blockIdx.y) * BM + lm;
            const int gk = k0 + lk;
            a_tile[lm][lk] = (gm < M && gk < K)
                ? x_q[static_cast<int64_t>(gm) * K + gk]
                : int8_t{0};
        }
        for (int i = tid; i < BK * BN; i += threads) {
            const int lk = i / BN;
            const int ln = i % BN;
            const int gk = k0 + lk;
            const int gn = static_cast<int>(blockIdx.x) * BN + ln;
            b_tile[lk][ln] = (gk < K && gn < N)
                ? weight_kn[static_cast<int64_t>(gk) * N + gn]
                : int8_t{0};
        }
        __syncthreads();
        if (row < M && col < N) {
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                acc += static_cast<int>(a_tile[ty][kk]) *
                       static_cast<int>(b_tile[kk][tx]);
            }
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        const float y = static_cast<float>(acc) * x_scale[row] * weight_scale[col];
        output[static_cast<int64_t>(row) * N + col] = hip_bfloat16(y);
    }
}

// =========================================================================
// Eight-wave MMAC kernel — general W8A8 GEMM using v_mmac_i32_16x16x32_i8
// =========================================================================
// BM=32, BN=64, BK=128, 512 threads (8 Wave64).
// Each wave computes 16×16 output elements. Wave-to-tile mapping:
//   row_group = wave >> 2, col_group = wave & 3
//   waves 0-3 → rows 0..15, waves 4-7 → rows 16..31
//   col_group * 16 → column slice start within the 64-column tile
// A tile ([2][32][132]) and B tile ([2][64][132]) are double-buffered.
// 128-bit global loads, B LDS transposition, MMAC compute.

typedef int int4_t __attribute__((ext_vector_type(4)));
typedef int int2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int2_t load_lds_int8x8(const int8_t* ptr) {
    const int32_t* p = reinterpret_cast<const int32_t*>(ptr);
    int2_t result;
    result.x = p[0];
    result.y = p[1];
    return result;
}

__device__ __forceinline__ int4_t mmac_i32_16x16x32_i8(
    int2_t a, int2_t b, int4_t c) {
    __asm__ __volatile__(
        "v_mmac_i32_16x16x32_i8 %0, %1, %2, %0"
        : "+v"(c)
        : "v"(a), "v"(b));
    return c;
}

template <int BM, int BN, int BK>
__global__ __launch_bounds__(512) void w8a8_scaled_gemm_mmac_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ w,
    const float* __restrict__ a_scale,
    const float* __restrict__ w_scale,
    hip_bfloat16* __restrict__ y,
    int M, int N, int K) {

    constexpr int BK_PAD = BK + 4;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int row16 = lane & 15;
    const int kg = lane >> 4;
    const int row_group = wave >> 2;
    const int col_group = wave & 3;
    const int row = row_group * 16 + row16;

    // Double-buffered LDS: two banks so global loads for tile k+1 overlap
    // MMAC compute on tile k. Pipeline depth = 2.
    __shared__ __align__(16) int8_t a_tile[2][BM][BK_PAD];
    __shared__ __align__(16) int8_t b_tile[2][BN][BK_PAD];

    int4_t acc{};
    acc.x = 0; acc.y = 0; acc.z = 0; acc.w = 0;

    // ── Prefetch: load tile 0 into buffer 0 ─────────────────────────
    {
        constexpr int A_VECS = BM * BK / 16;
        for (int vi = tid; vi < A_VECS; vi += 512) {
            const int lm = vi / (BK / 16);
            const int lk = (vi % (BK / 16)) * 16;
            const int gm = static_cast<int>(blockIdx.y) * BM + lm;
            GlobalVec128 value{};
            if (gm < M && lk + 16 <= K &&
                (reinterpret_cast<uintptr_t>(
                    a + static_cast<int64_t>(gm) * K + lk) & 15u) == 0) {
                value = *reinterpret_cast<const GlobalVec128*>(
                    a + static_cast<int64_t>(gm) * K + lk);
            } else if (gm < M && lk < K) {
                const int valid = (K - lk) < 16 ? (K - lk) : 16;
#pragma unroll
                for (int j = 0; j < valid; ++j)
                    reinterpret_cast<int8_t*>(&value)[j] =
                        a[static_cast<int64_t>(gm) * K + lk + j];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j)
                a_tile[0][lm][lk + j] = reinterpret_cast<const int8_t*>(&value)[j];
        }

        constexpr int B_VECS = BK * BN / 16;
        for (int vi = tid; vi < B_VECS; vi += 512) {
            const int lk = vi / (BN / 16);
            const int ln = (vi % (BN / 16)) * 16;
            const int gn = static_cast<int>(blockIdx.x) * BN + ln;
            GlobalVec128 value{};
            if (lk < K && gn + 16 <= N &&
                (reinterpret_cast<uintptr_t>(
                    w + static_cast<int64_t>(lk) * N + gn) & 15u) == 0) {
                value = *reinterpret_cast<const GlobalVec128*>(
                    w + static_cast<int64_t>(lk) * N + gn);
            } else if (lk < K && gn < N) {
                const int valid = (N - gn) < 16 ? (N - gn) : 16;
#pragma unroll
                for (int j = 0; j < valid; ++j)
                    reinterpret_cast<int8_t*>(&value)[j] =
                        w[static_cast<int64_t>(lk) * N + gn + j];
            }
#pragma unroll
            for (int j = 0; j < 16; ++j)
                b_tile[0][ln + j][lk] = reinterpret_cast<const int8_t*>(&value)[j];
        }
    }
    __syncthreads();

    // ── Pipelined K-loop: overlap loads for tile k+1 with MMAC on tile k
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int buf_comp = (k0 / BK) & 1;
        const int buf_load = 1 - buf_comp;

        // Issue global loads for the next K-tile (if not the last iteration)
        if (k0 + BK < K) {
            constexpr int A_VECS = BM * BK / 16;
            for (int vi = tid; vi < A_VECS; vi += 512) {
                const int lm = vi / (BK / 16);
                const int lk = (vi % (BK / 16)) * 16;
                const int gm = static_cast<int>(blockIdx.y) * BM + lm;
                const int gk = k0 + BK + lk;
                GlobalVec128 value{};
                if (gm < M && gk + 16 <= K &&
                (reinterpret_cast<uintptr_t>(
                    a + static_cast<int64_t>(gm) * K + gk) & 15u) == 0) {
                    value = *reinterpret_cast<const GlobalVec128*>(
                        a + static_cast<int64_t>(gm) * K + gk);
                } else if (gm < M && gk < K) {
                    const int valid = (K - gk) < 16 ? (K - gk) : 16;
#pragma unroll
                    for (int j = 0; j < valid; ++j)
                        reinterpret_cast<int8_t*>(&value)[j] =
                            a[static_cast<int64_t>(gm) * K + gk + j];
                }
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    a_tile[buf_load][lm][lk + j] =
                        reinterpret_cast<const int8_t*>(&value)[j];
            }

            constexpr int B_VECS = BK * BN / 16;
            for (int vi = tid; vi < B_VECS; vi += 512) {
                const int lk = vi / (BN / 16);
                const int ln = (vi % (BN / 16)) * 16;
                const int gk = k0 + BK + lk;
                const int gn = static_cast<int>(blockIdx.x) * BN + ln;
                GlobalVec128 value{};
                if (gk < K && gn + 16 <= N &&
                (reinterpret_cast<uintptr_t>(
                    w + static_cast<int64_t>(gk) * N + gn) & 15u) == 0) {
                    value = *reinterpret_cast<const GlobalVec128*>(
                        w + static_cast<int64_t>(gk) * N + gn);
                } else if (gk < K && gn < N) {
                    const int valid = (N - gn) < 16 ? (N - gn) : 16;
#pragma unroll
                    for (int j = 0; j < valid; ++j)
                        reinterpret_cast<int8_t*>(&value)[j] =
                            w[static_cast<int64_t>(gk) * N + gn + j];
                }
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    b_tile[buf_load][ln + j][lk] =
                        reinterpret_cast<const int8_t*>(&value)[j];
            }
        }

        // MMAC compute on current buffer
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            const int fragment_k = kk + kg * 8;
            int2_t af = load_lds_int8x8(
                &a_tile[buf_comp][row][fragment_k]);
            int2_t bf = load_lds_int8x8(
                &b_tile[buf_comp][col_group * 16 + row16][fragment_k]);
            acc = mmac_i32_16x16x32_i8(af, bf, acc);
        }
        __syncthreads();
    }

    // Epilogue: FP32 scale → BF16 write, 4 outputs per thread
    const int global_row = static_cast<int>(blockIdx.y) * 32 + row;
    if (global_row < M) {
        const float a_s = a_scale[global_row];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = static_cast<int>(blockIdx.x) * 64
                          + col_group * 16 + kg + i * 4;
            if (col < N) {
                y[static_cast<int64_t>(global_row) * N + col] =
                    hip_bfloat16(static_cast<float>(acc[i]) * a_s * w_scale[col]);
            }
        }
    }
}

// =========================================================================
// Split-K dispatch table
// =========================================================================

__host__ inline int choose_split_k(int M, int N, int K) {
    // Exact table entries — M-aware ternary extends iteration 2's M ≤ 8 table
    // to also cover M=16 for K ≥ 2048 shapes.
    if (K == 4096 && N == 1536) return M <= 8 ? 16 : 8;   // wqkv-a TP=4, TP=8
    if (K == 4096 && N == 1024) return M <= 8 ? 32 : 16;  // shared-gate-up TP=4
    if (K == 4096 && N == 512)  return M <= 8 ? 32 : 16;  // shared-gate-up TP=8
    if (K == 2048 && N == 4096) return M <= 8 ? 16 : 8;   // wo-b TP=4

    // M ≤ 8 shapes from iteration 2 (unchanged), not covered by exact table
    if (M <= 8) {
        if (K == 1024) return 4;   // wq-b TP=8, wo-b TP=8
        if (K == 512)  return 4;   // shared-down-proj TP=4
        if (K == 256)  return 2;   // shared-down-proj TP=8
    }
    return 1;  // no split
}

// =========================================================================
// Static workspace for split-K partial outputs (persists across calls)
// =========================================================================

static int32_t* g_workspace = nullptr;
static size_t g_workspace_capacity = 0;

__host__ static int ensure_workspace(size_t bytes) {
    if (bytes <= g_workspace_capacity) return 0;
    if (g_workspace) {
        hipFree(g_workspace);
        g_workspace = nullptr;
        g_workspace_capacity = 0;
    }
    hipError_t err = hipMalloc(&g_workspace, bytes);
    if (err != hipSuccess) return static_cast<int>(err);
    g_workspace_capacity = bytes;
    return 0;
}

// =========================================================================
// Public ABI — the only symbol called by the frozen evaluator
// =========================================================================

extern "C" int launch_w8a8_gemm(
    const int8_t* x_q,
    const int8_t* weight_kn,
    const float* x_scale,
    const float* weight_scale,
    void* output_bf16,
    int M, int N, int K,
    void* stream_handle) {
    if (!x_q || !weight_kn || !x_scale || !weight_scale || !output_bf16 ||
        M <= 0 || N <= 0 || K <= 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    auto* output = static_cast<hip_bfloat16*>(output_bf16);
    auto stream = static_cast<hipStream_t>(stream_handle);

    // Native DPP Wave64 SDOT4 path for shared-down-proj TP8.
    if (M <= 4 && N == 4096 && K == 256 &&
        (reinterpret_cast<uintptr_t>(x_q) & 3u) == 0 &&
        (reinterpret_cast<uintptr_t>(weight_kn) & 3u) == 0) {
        constexpr int threads = 256;
        constexpr int columns_per_block = 16;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(wave_dpp_sdot4_k256_n4096_kernel),
            dim3(N / columns_per_block), dim3(threads), 0, stream,
            x_q, weight_kn, x_scale, weight_scale, output, M, N, K);
        const hipError_t error = hipGetLastError();
        return error == hipSuccess ? 0 : static_cast<int>(error);
    }

    // --- Split-K SDOT4 path: M ≤ 16, K ≥ 2048, and a valid split value ---
    const bool use_split_k = (M <= 16 && K >= 2048) ||
                             (M <= 8);  // M ≤ 8 always tries split-K (iteration 2 behavior)

    if (use_split_k) {
        const int split_k = choose_split_k(M, N, K);
        if (split_k > 1) {
            const size_t workspace_bytes =
                static_cast<size_t>(split_k) * M * N * sizeof(int32_t);
            int err = ensure_workspace(workspace_bytes);
            if (err != 0) return err;

            // Split-K GEMM: each workgroup computes one K-slice
            constexpr int BM = 8;
            constexpr int BN = 64;
            constexpr int BK = 32;
            dim3 block(BN, BM);  // 64 × 8 = 512 threads
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, split_k);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(small_m_splitk_dot4_kernel<BM, BN, BK>),
                grid, block, 0, stream,
                x_q, weight_kn, g_workspace, M, N, K, split_k);

            // Reduction: INT32 sum across splits → FP32 scale → BF16 write
            const int64_t elements = static_cast<int64_t>(M) * N;
            const int reduce_threads = 256;
            const int reduce_blocks =
                static_cast<int>((elements + reduce_threads - 1) / reduce_threads);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(reduce_splitk_scale_kernel),
                dim3(reduce_blocks), dim3(reduce_threads), 0, stream,
                g_workspace, x_scale, weight_scale, output, M, N, split_k);

            const hipError_t error = hipGetLastError();
            return error == hipSuccess ? 0 : static_cast<int>(error);
        }
        // split_k == 1: fall through to appropriate fallback below
    }

    // --- Small-M fallback (scalar GEMV): M ≤ 8 with no valid split-K ---
    if (M <= 8) {
        constexpr int BN = 256;
        constexpr int BK = 256;
        dim3 block(BN);
        dim3 grid((N + BN - 1) / BN, M);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_scaled_gemv_kernel<BN, BK>),
            grid, block, 0, stream,
            x_q, weight_kn, x_scale, weight_scale, output, M, N, K);
    }
    // --- General MMAC kernel: everything not routed to split-K ---
    else {
        constexpr int BM = 32;
        constexpr int BN = 64;
        constexpr int BK = 128;
        dim3 block(512);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_scaled_gemm_mmac_kernel<BM, BN, BK>),
            grid, block, 0, stream,
            x_q, weight_kn, x_scale, weight_scale, output, M, N, K);
    }

    const hipError_t error = hipGetLastError();
    return error == hipSuccess ? 0 : static_cast<int>(error);
}
