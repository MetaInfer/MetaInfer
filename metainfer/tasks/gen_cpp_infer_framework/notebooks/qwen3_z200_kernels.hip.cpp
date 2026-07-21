#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cmath>

// HIP kernels for the dense Qwen3 text path on Hygon DCU/Z200-style targets.
//
// Scope:
// - Keep only operators needed by Qwen3 forward.
// - Leave weighted linear layers to hipBLAS/rocBLAS.
// - Avoid CUDA-only CUB, cuda_pipeline, cp.async, and hard-coded warp size 32.
//
// Expected tensor layouts:
// - hidden states: [tokens, hidden_dim], contiguous row-major.
// - Q:            [tokens, n_heads,    head_dim], contiguous row-major.
// - K/V:          [tokens, n_kv_heads, head_dim], contiguous row-major.
// - KV cache:     [max_seq_len, n_kv_heads, head_dim], contiguous row-major
//                 for a single sequence. Add a batch stride in the caller if
//                 you later support multi-sequence batching.

namespace qwen3_z200 {

constexpr int kBlockSize = 256;
constexpr int kQ8BlockSize = 32;

// GGML/GGUF Q8_0 stores 32 signed quants behind one FP16 scale.  Keep this
// layout byte-for-byte compatible with block_q8_0 in ggml-quants.h.
struct alignas(2) BlockQ8_0 {
    __half d;
    int8_t qs[kQ8BlockSize];
};

static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be exactly 34 bytes");

enum RopeMode {
    ROPE_INTERLEAVED = 0,
    ROPE_NEOX        = 1,
};

static inline int div_up(int x, int y) {
    return (x + y - 1) / y;
}

static inline size_t div_up_size(size_t x, size_t y) {
    return (x + y - 1) / y;
}

__global__ void cast_fp32_to_fp16_kernel(
        __half * __restrict__ out,
        const float * __restrict__ x,
        size_t n_elements) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_elements) {
        out[tid] = __float2half(x[tid]);
    }
    return;
}

__global__ void dequant_q8_0_to_fp16_kernel(
        __half * __restrict__ out,
        const BlockQ8_0 * __restrict__ weight,
        size_t n_elements) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) {
        return;
    }

    const BlockQ8_0 * block = &weight[tid / kQ8BlockSize];
    const int offset = (int) (tid % kQ8BlockSize);
    const float value = __half2float(block->d) * (float) block->qs[offset];
    out[tid] = __float2half(value);
}
//tokenid 到 dim的映射
__global__ void embedding_lookup_q8_0_kernel(
        float * __restrict__ out,
        const BlockQ8_0 * __restrict__ token_embedding,
        const int * __restrict__ token_ids,
        int n_tokens,
        int vocab_size,
        int hidden_dim) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) n_tokens * hidden_dim;
    if (tid >= total) {
        return;
    }

    const int d = (int) (tid % hidden_dim);
    const int t = (int) (tid / hidden_dim);
    const int token_id = token_ids[t];
    if (token_id < 0 || token_id >= vocab_size) {
        out[tid] = 0.0f;
        return;
    }

    const size_t blocks_per_row = (size_t) hidden_dim / kQ8BlockSize;
    const size_t block_idx = (size_t) token_id * blocks_per_row
            + (size_t) d / kQ8BlockSize;
    const int offset = d % kQ8BlockSize;
    const BlockQ8_0 * block = &token_embedding[block_idx];
    out[tid] = __half2float(block->d) * (float) block->qs[offset];
}

__device__ float block_reduce_sum(float value, float * smem) {
    const int tid = threadIdx.x;
    smem[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    return smem[0];
}

__device__ float block_reduce_max(float value, float * smem) {
    const int tid = threadIdx.x;
    smem[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    return smem[0];
}

__global__ void embedding_lookup_kernel(
        float * __restrict__ out,
        const float * __restrict__ token_embedding,
        const int * __restrict__ token_ids,
        int n_tokens,
        int hidden_dim) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * hidden_dim;
    if (tid >= total) {
        return;
    }

    const int d = tid % hidden_dim;
    const int t = tid / hidden_dim;
    const int token_id = token_ids[t];
    out[tid] = token_embedding[(long long) token_id * hidden_dim + d];
}

__global__ void rms_norm_kernel(
        float * __restrict__ out,
        const float * __restrict__ x,
        const float * __restrict__ weight,
        int rows,
        int dim,
        float eps) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const long long base = (long long) row * dim;
    float sum = 0.0f;

    for (int d = tid; d < dim; d += blockDim.x) {
        const float v = x[base + d];
        sum += v * v;
    }

    const float total = block_reduce_sum(sum, smem);
    const float inv_rms = rsqrtf(total / (float) dim + eps);

    for (int d = tid; d < dim; d += blockDim.x) {
        out[base + d] = x[base + d] * inv_rms * weight[d];
    }
}

__global__ void per_head_rms_norm_kernel(
        float * __restrict__ out,
        const float * __restrict__ x,
        const float * __restrict__ weight,
        int n_tokens,
        int n_heads,
        int head_dim,
        int row_stride,
        float eps) {
    extern __shared__ float smem[];

    const int block = blockIdx.x;
    const int tid = threadIdx.x;
    const int t = block / n_heads;
    const int h = block - t * n_heads;
    if (t >= n_tokens) {
        return;
    }

    const long long base = (long long) t * row_stride + (long long) h * head_dim;
    float sum = 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = x[base + d];
        sum += v * v;
    }

    const float total = block_reduce_sum(sum, smem);
    const float inv_rms = rsqrtf(total / (float) head_dim + eps);

    for (int d = tid; d < head_dim; d += blockDim.x) {
        out[base + d] = x[base + d] * inv_rms * weight[d];
    }
}

__global__ void rope_kernel(
        float * __restrict__ x,
        const float * __restrict__ cos_table,
        const float * __restrict__ sin_table,
        int n_tokens,
        int n_heads,
        int head_dim,
        int row_stride,
        int start_pos,
        int max_position,
        int rope_mode) {
    const int half = head_dim >> 1;
    const int pair = threadIdx.x;
    const int block = blockIdx.x;
    const int t = block / n_heads;
    const int h = block - t * n_heads;

    if (t >= n_tokens || pair >= half) {
        return;
    }

    const int pos = start_pos + t;
    if (pos >= max_position) {
        return;
    }

    const float c = cos_table[(long long) pos * half + pair];
    const float s = sin_table[(long long) pos * half + pair];

    const int i0 = rope_mode == ROPE_NEOX ? pair : 2 * pair;    //对半索引
    const int i1 = rope_mode == ROPE_NEOX ? pair + half : 2 * pair + 1;

    const long long base = (long long) t * row_stride + (long long) h * head_dim;
    const float x0 = x[base + i0];
    const float x1 = x[base + i1];

    x[base + i0] = x0 * c - x1 * s;
    x[base + i1] = x0 * s + x1 * c;
}

__global__ void kv_cache_write_kernel(
        const float * __restrict__ k_src,
        const float * __restrict__ v_src,
        float * __restrict__ k_cache,
        float * __restrict__ v_cache,
        int n_tokens,
        int n_kv_heads,
        int head_dim,
        int max_seq_len,
        int start_pos) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_kv_heads * head_dim;
    if (tid >= total) {
        return;
    }

    const int d = tid % head_dim;
    const int tmp = tid / head_dim;
    const int h = tmp % n_kv_heads;
    const int t = tmp / n_kv_heads;
    const int pos = start_pos + t;

    if (pos >= max_seq_len) {
        return;
    }

    const long long src = ((long long) t * n_kv_heads + h) * head_dim + d;
    const long long dst = ((long long) pos * n_kv_heads + h) * head_dim + d;

    k_cache[dst] = k_src[src];
    v_cache[dst] = v_src[src];
}

__global__ void prefill_gqa_attention_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k_cache,
        const float * __restrict__ v_cache,
        float * __restrict__ out,
        int n_tokens,
        int start_pos,
        int max_seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float scale) {
    extern __shared__ float smem[];

    const int token_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= n_tokens || q_head >= n_heads) {
        return;
    }

    const int current_pos = start_pos + token_idx;
    if (current_pos >= max_seq_len) {
        return;
    }

    const int q_per_kv = n_heads / n_kv_heads;
    const int kv_head = q_head / q_per_kv;

    const float * q_vec = q + ((long long) token_idx * n_heads + q_head) * head_dim;

    float local_max = -FLT_MAX;
    for (int t = 0; t <= current_pos; ++t) {
        const float * k_vec = k_cache + ((long long) t * n_kv_heads + kv_head) * head_dim;
        float partial = 0.0f;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            partial += q_vec[d] * k_vec[d];
        }

        const float score = block_reduce_sum(partial, smem) * scale;
        local_max = fmaxf(local_max, score);
    }

    const float max_score = block_reduce_max(local_max, smem);

    float sum_score = 0.0f;
    float acc = 0.0f;

    for (int t = 0; t <= current_pos; ++t) {
        const float * k_vec = k_cache + ((long long) t * n_kv_heads + kv_head) * head_dim;
        float partial = 0.0f;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            partial += q_vec[d] * k_vec[d];
        }

        const float score = block_reduce_sum(partial, smem) * scale;
        const float prob = expf(score - max_score);
        sum_score += prob;

        if (tid < head_dim) {
            const float * v_vec = v_cache + ((long long) t * n_kv_heads + kv_head) * head_dim;
            acc += prob * v_vec[tid];
        }
    }

    if (tid < head_dim) {
        const long long out_idx = ((long long) token_idx * n_heads + q_head) * head_dim + tid;
        out[out_idx] = acc / sum_score;
    }
}

__global__ void decode_gqa_attention_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k_cache,
        const float * __restrict__ v_cache,
        float * __restrict__ out,
        float * __restrict__ scores,
        int current_pos,
        int max_seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float scale) {
    extern __shared__ float smem[];

    const int q_head = blockIdx.x;
    const int tid = threadIdx.x;
    if (q_head >= n_heads) {
        return;
    }

    const int seq_len = current_pos + 1;
    const int q_per_kv = n_heads / n_kv_heads;
    const int kv_head = q_head / q_per_kv;

    const float * q_vec = q + (long long) q_head * head_dim;
    float * head_scores = scores + (long long) q_head * max_seq_len;

    for (int t = tid; t < seq_len; t += blockDim.x) {
        const float * k_vec = k_cache + ((long long) t * n_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }

        head_scores[t] = dot * scale;
    }
    __syncthreads();

    float local_max = -FLT_MAX;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        local_max = fmaxf(local_max, head_scores[t]);
    }
    const float max_score = block_reduce_max(local_max, smem);

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        const float p = expf(head_scores[t] - max_score);
        head_scores[t] = p;
        local_sum += p;
    }
    const float sum_score = block_reduce_sum(local_sum, smem);
    const float inv_sum = 1.0f / sum_score;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (int t = 0; t < seq_len; ++t) {
            const float p = head_scores[t] * inv_sum;
            const float * v_vec = v_cache + ((long long) t * n_kv_heads + kv_head) * head_dim;
            acc += p * v_vec[d];
        }

        out[(long long) q_head * head_dim + d] = acc;
    }
}

__global__ void swiglu_kernel(
        float * __restrict__ out,
        const float * __restrict__ gate,
        const float * __restrict__ up,
        int n_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elements) {
        return;
    }

    const float g = gate[tid];
    out[tid] = (g / (1.0f + expf(-g))) * up[tid];
}

__global__ void greedy_argmax_kernel(
        const float * __restrict__ logits,
        int vocab_size,
        int * __restrict__ out_token) {
    const int tid = threadIdx.x;
    float best_val = -FLT_MAX;
    int best_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float v = logits[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    __shared__ float s_vals[kBlockSize];
    __shared__ int s_idxs[kBlockSize];

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_val = s_vals[tid + stride];
            const int other_idx = s_idxs[tid + stride];
            if (other_val > s_vals[tid]
                    || (other_val == s_vals[tid] && other_idx < s_idxs[tid])) {
                s_vals[tid] = other_val;
                s_idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token = s_idxs[0];
    }
    return;
}

__global__ void add_kernel(
        float * __restrict__ out,
        const float * __restrict__ a,
        const float * __restrict__ b,
        int n_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_elements) {
        out[tid] = a[tid] + b[tid];
    }
    return;
}

__global__ void add_inplace_kernel(
        float * __restrict__ dst,
        const float * __restrict__ src,
        int n_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_elements) {
        dst[tid] += src[tid];
    }
    return;
}

} // namespace qwen3_z200

extern "C" hipError_t qwen3_z200_launch_cast_fp32_to_fp16(
        __half * out,
        const float * x,
        size_t n_elements,
        hipStream_t stream) {
    if (out == nullptr || x == nullptr || n_elements == 0) {
        return hipErrorInvalidValue;
    }

    hipLaunchKernelGGL(
            qwen3_z200::cast_fp32_to_fp16_kernel,
            dim3((unsigned int) qwen3_z200::div_up_size(
                    n_elements, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            out,
            x,
            n_elements);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_dequant_q8_0_to_fp16(
        __half * out,
        const qwen3_z200::BlockQ8_0 * weight,
        size_t n_elements,
        hipStream_t stream) {
    if (out == nullptr || weight == nullptr || n_elements == 0
            || n_elements % qwen3_z200::kQ8BlockSize != 0) {
        return hipErrorInvalidValue;
    }

    hipLaunchKernelGGL(
            qwen3_z200::dequant_q8_0_to_fp16_kernel,
            dim3((unsigned int) qwen3_z200::div_up_size(
                    n_elements, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            out,
            weight,
            n_elements);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_embedding_lookup_q8_0(
        float * out,
        const qwen3_z200::BlockQ8_0 * token_embedding,
        const int * token_ids,
        int n_tokens,
        int vocab_size,
        int hidden_dim,
        hipStream_t stream) {
    if (out == nullptr || token_embedding == nullptr || token_ids == nullptr
            || n_tokens <= 0 || vocab_size <= 0 || hidden_dim <= 0
            || hidden_dim % qwen3_z200::kQ8BlockSize != 0) {
        return hipErrorInvalidValue;
    }

    const size_t total = (size_t) n_tokens * hidden_dim;
    hipLaunchKernelGGL(
            qwen3_z200::embedding_lookup_q8_0_kernel,
            dim3((unsigned int) qwen3_z200::div_up_size(
                    total, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            out,
            token_embedding,
            token_ids,
            n_tokens,
            vocab_size,
            hidden_dim);
    return hipGetLastError();
}

// Compute row-major Y[M, N] = X[M, K] * W[N, K]^T.
//
// X and the existing non-linear kernels remain FP32.  W stays Q8_0 in device
// memory.  The caller owns and reuses both FP16 workspaces; this function only
// enqueues cast -> dequant -> GEMM on one stream.
//
// hipBLAS is column-major, so row-major W[N, K] is viewed as column-major
// [K, N], transposed to [N, K], and row-major X[M, K] is viewed as
// column-major [K, M].  The column-major result [N, M] has the same bytes as
// row-major Y[M, N].
extern "C" hipblasStatus_t qwen3_z200_q8_linear_fp32(
        hipblasHandle_t handle,
        float * out,
        const float * x,
        const qwen3_z200::BlockQ8_0 * weight,
        __half * x_fp16_workspace,
        size_t x_workspace_elements,
        __half * weight_fp16_workspace,
        size_t weight_workspace_elements,
        int m,
        int n,
        int k,
        hipStream_t stream) {
    if (handle == nullptr || out == nullptr || x == nullptr || weight == nullptr
            || x_fp16_workspace == nullptr || weight_fp16_workspace == nullptr
            || m <= 0 || n <= 0 || k <= 0
            || k % qwen3_z200::kQ8BlockSize != 0) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    const size_t x_elements = (size_t) m * k;
    const size_t weight_elements = (size_t) n * k;
    if (x_workspace_elements < x_elements
            || weight_workspace_elements < weight_elements) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasStatus_t status = hipblasSetStream(handle, stream);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return status;
    }

    hipError_t hip_status = qwen3_z200_launch_cast_fp32_to_fp16(
            x_fp16_workspace, x, x_elements, stream);
    if (hip_status != hipSuccess) {
        return HIPBLAS_STATUS_EXECUTION_FAILED;
    }

    hip_status = qwen3_z200_launch_dequant_q8_0_to_fp16(
            weight_fp16_workspace, weight, weight_elements, stream);
    if (hip_status != hipSuccess) {
        return HIPBLAS_STATUS_EXECUTION_FAILED;
    }

    hipblasPointerMode_t old_pointer_mode;
    status = hipblasGetPointerMode(handle, &old_pointer_mode);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        return status;
    }

    const bool restore_pointer_mode = old_pointer_mode != HIPBLAS_POINTER_MODE_HOST;
    if (restore_pointer_mode) {
        status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            return status;
        }
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // DTK 5.7 exposes the legacy hipBLAS Ex API.  Its matrix and compute
    // types are all hipblasDatatype_t, so use the HIPBLAS_R_* values rather
    // than the hipDataType HIP_R_* values used by newer v2 headers.
    status = hipblasGemmEx(
            handle,
            HIPBLAS_OP_T,
            HIPBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            weight_fp16_workspace,
            HIPBLAS_R_16F,
            k,
            x_fp16_workspace,
            HIPBLAS_R_16F,
            k,
            &beta,
            out,
            HIPBLAS_R_32F,
            n,
            HIPBLAS_R_32F,
            HIPBLAS_GEMM_DEFAULT);

    if (restore_pointer_mode) {
        const hipblasStatus_t restore_status = hipblasSetPointerMode(
                handle, old_pointer_mode);
        if (status == HIPBLAS_STATUS_SUCCESS) {
            status = restore_status;
        }
    }
    return status;
}

extern "C" hipError_t qwen3_z200_launch_embedding_lookup(
        float * out,
        const float * token_embedding,
        const int * token_ids,
        int n_tokens,
        int hidden_dim,
        hipStream_t stream) {
    const int total = n_tokens * hidden_dim;
    hipLaunchKernelGGL(
            qwen3_z200::embedding_lookup_kernel,
            dim3(qwen3_z200::div_up(total, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            out,
            token_embedding,
            token_ids,
            n_tokens,
            hidden_dim);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_rms_norm(
        float * out,
        const float * x,
        const float * weight,
        int rows,
        int dim,
        float eps,
        hipStream_t stream) {
    hipLaunchKernelGGL(
            qwen3_z200::rms_norm_kernel,
            dim3(rows),
            dim3(qwen3_z200::kBlockSize),
            qwen3_z200::kBlockSize * sizeof(float),
            stream,
            out,
            x,
            weight,
            rows,
            dim,
            eps);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_per_head_rms_norm(
        float * out,
        const float * x,
        const float * weight,
        int n_tokens,
        int n_heads,
        int head_dim,
        int row_stride,
        float eps,
        hipStream_t stream) {
    hipLaunchKernelGGL(
            qwen3_z200::per_head_rms_norm_kernel,
            dim3(n_tokens * n_heads),
            dim3(qwen3_z200::kBlockSize),
            qwen3_z200::kBlockSize * sizeof(float),
            stream,
            out,
            x,
            weight,
            n_tokens,
            n_heads,
            head_dim,
            row_stride,
            eps);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_rope(
        float * x,
        const float * cos_table,
        const float * sin_table,
        int n_tokens,
        int n_heads,
        int head_dim,
        int row_stride,
        int start_pos,
        int max_position,
        int rope_mode,
        hipStream_t stream) {
    hipLaunchKernelGGL(
            qwen3_z200::rope_kernel,
            dim3(n_tokens * n_heads),
            dim3((head_dim + 1) / 2),
            0,
            stream,
            x,
            cos_table,
            sin_table,
            n_tokens,
            n_heads,
            head_dim,
            row_stride,
            start_pos,
            max_position,
            rope_mode);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_kv_cache_write(
        const float * k_src,
        const float * v_src,
        float * k_cache,
        float * v_cache,
        int n_tokens,
        int n_kv_heads,
        int head_dim,
        int max_seq_len,
        int start_pos,
        hipStream_t stream) {
    const int total = n_tokens * n_kv_heads * head_dim;
    hipLaunchKernelGGL(
            qwen3_z200::kv_cache_write_kernel,
            dim3(qwen3_z200::div_up(total, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            k_src,
            v_src,
            k_cache,
            v_cache,
            n_tokens,
            n_kv_heads,
            head_dim,
            max_seq_len,
            start_pos);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_prefill_gqa_attention(
        const float * q,
        const float * k_cache,
        const float * v_cache,
        float * out,
        int n_tokens,
        int start_pos,
        int max_seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float scale,
        hipStream_t stream) {
    if (head_dim > qwen3_z200::kBlockSize || n_heads % n_kv_heads != 0) {
        return hipErrorInvalidValue;
    }

    hipLaunchKernelGGL(
            qwen3_z200::prefill_gqa_attention_kernel,
            dim3(n_tokens, n_heads),
            dim3(qwen3_z200::kBlockSize),
            qwen3_z200::kBlockSize * sizeof(float),
            stream,
            q,
            k_cache,
            v_cache,
            out,
            n_tokens,
            start_pos,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            scale);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_decode_gqa_attention(
        const float * q,
        const float * k_cache,
        const float * v_cache,
        float * out,
        float * scores,
        int current_pos,
        int max_seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float scale,
        hipStream_t stream) {
    hipLaunchKernelGGL(
            qwen3_z200::decode_gqa_attention_kernel,
            dim3(n_heads),
            dim3(qwen3_z200::kBlockSize),
            qwen3_z200::kBlockSize * sizeof(float),
            stream,
            q,
            k_cache,
            v_cache,
            out,
            scores,
            current_pos,
            max_seq_len,
            n_heads,
            n_kv_heads,
            head_dim,
            scale);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_swiglu(
        float * out,
        const float * gate,
        const float * up,
        int n_elements,
        hipStream_t stream) {
    hipLaunchKernelGGL(
            qwen3_z200::swiglu_kernel,
            dim3(qwen3_z200::div_up(n_elements, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            out,
            gate,
            up,
            n_elements);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_add(
        float * out,
        const float * a,
        const float * b,
        int n_elements,
        hipStream_t stream) {
    hipLaunchKernelGGL(
            qwen3_z200::add_kernel,
            dim3(qwen3_z200::div_up(n_elements, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            out,
            a,
            b,
            n_elements);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_add_inplace(
        float * dst,
        const float * src,
        int n_elements,
        hipStream_t stream) {
    hipLaunchKernelGGL(
            qwen3_z200::add_inplace_kernel,
            dim3(qwen3_z200::div_up(n_elements, qwen3_z200::kBlockSize)),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            dst,
            src,
            n_elements);
    return hipGetLastError();
}

extern "C" hipError_t qwen3_z200_launch_greedy_sample(
        const float * logits,
        int vocab_size,
        int * out_token,
        hipStream_t stream) {
    if (logits == nullptr || out_token == nullptr || vocab_size <= 0) {
        return hipErrorInvalidValue;
    }

    hipLaunchKernelGGL(
            qwen3_z200::greedy_argmax_kernel,
            dim3(1),
            dim3(qwen3_z200::kBlockSize),
            0,
            stream,
            logits,
            vocab_size,
            out_token);
    return hipGetLastError();
}
