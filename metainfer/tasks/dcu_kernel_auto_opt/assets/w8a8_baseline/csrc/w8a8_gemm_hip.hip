#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>


namespace {

constexpr int kBlockThreads = 256;

__global__ void w8a8_gemm_scalar(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    hip_bfloat16* __restrict__ out,
    int m,
    int n,
    int k) {
  const int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t output_elements = static_cast<int64_t>(m) * n;
  if (output_index >= output_elements) {
    return;
  }

  const int row = static_cast<int>(output_index / n);
  const int col = static_cast<int>(output_index - static_cast<int64_t>(row) * n);
  const int8_t* a_row = a + static_cast<int64_t>(row) * k;

  int32_t accumulator = 0;
  for (int inner = 0; inner < k; ++inner) {
    accumulator += static_cast<int32_t>(a_row[inner]) *
                   static_cast<int32_t>(
                       b[static_cast<int64_t>(inner) * n + col]);
  }
  const float scaled = static_cast<float>(accumulator) * x_scale[row] *
                       weight_scale[col];
  out[output_index] = hip_bfloat16(scaled);
}

}  // namespace


extern "C" void launch_w8a8_gemm(
    const int8_t* a,
    const int8_t* b,
    const float* x_scale,
    const float* weight_scale,
    void* out,
    void* workspace,
    int64_t workspace_bytes,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  (void)workspace;
  (void)workspace_bytes;
  const int64_t output_elements = static_cast<int64_t>(m) * n;
  const dim3 block(kBlockThreads);
  const dim3 grid(
      static_cast<unsigned int>(
          (output_elements + kBlockThreads - 1) / kBlockThreads));
  hipLaunchKernelGGL(
      w8a8_gemm_scalar,
      grid,
      block,
      0,
      stream,
      a,
      b,
      x_scale,
      weight_scale,
      static_cast<hip_bfloat16*>(out),
      m,
      n,
      k);
}

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream) {
  hipMemcpyAsync(
      packed_weight,
      raw_weight,
      static_cast<size_t>(k) * n * sizeof(int8_t),
      hipMemcpyDeviceToDevice,
      stream);
  hipMemcpyAsync(
      packed_weight_scale,
      weight_scale,
      static_cast<size_t>(n) * sizeof(float),
      hipMemcpyDeviceToDevice,
      stream);
}
