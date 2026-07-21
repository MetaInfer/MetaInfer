#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// Decode/small-M baseline. One block computes one row and BLOCK_N columns.
template <int BLOCK_N, int BLOCK_K>
__global__ void w8a8_scaled_gemv_kernel(
    const int8_t* __restrict__ x_q,
    const int8_t* __restrict__ weight_kn,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ output,
    int M,
    int N,
    int K) {
    const int tid = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int col = static_cast<int>(blockIdx.x) * BLOCK_N + tid;
    if (row >= M) return;

    __shared__ int8_t a_tile[BLOCK_K];
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
    int M,
    int N,
    int K) {
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * BN + tx;
    const int threads = BM * BN;
    const int row = static_cast<int>(blockIdx.y) * BM + ty;
    const int col = static_cast<int>(blockIdx.x) * BN + tx;

    __shared__ int8_t a_tile[BM][BK];
    __shared__ int8_t b_tile[BK][BN];
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

// The only symbol called by the frozen evaluator. It enqueues work on the
// supplied stream and returns immediately; synchronization belongs to Harness.
extern "C" int launch_w8a8_gemm(
    const int8_t* x_q,
    const int8_t* weight_kn,
    const float* x_scale,
    const float* weight_scale,
    void* output_bf16,
    int M,
    int N,
    int K,
    void* stream_handle) {
    if (!x_q || !weight_kn || !x_scale || !weight_scale || !output_bf16 ||
        M <= 0 || N <= 0 || K <= 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    auto* output = static_cast<hip_bfloat16*>(output_bf16);
    auto stream = static_cast<hipStream_t>(stream_handle);
    if (M <= 8) {
        constexpr int BN = 256;
        constexpr int BK = 256;
        dim3 block(BN);
        dim3 grid((N + BN - 1) / BN, M);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_scaled_gemv_kernel<BN, BK>),
            grid, block, 0, stream,
            x_q, weight_kn, x_scale, weight_scale, output, M, N, K);
    } else {
        constexpr int BM = 16;
        constexpr int BN = 16;
        constexpr int BK = 32;
        dim3 block(BN, BM);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(w8a8_scaled_gemm_kernel<BM, BN, BK>),
            grid, block, 0, stream,
            x_q, weight_kn, x_scale, weight_scale, output, M, N, K);
    }
    const hipError_t error = hipGetLastError();
    return error == hipSuccess ? 0 : static_cast<int>(error);
}
