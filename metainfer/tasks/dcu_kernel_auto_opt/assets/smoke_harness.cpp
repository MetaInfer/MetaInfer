#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define HIP_CHECK(call) do { \
  hipError_t error = (call); \
  if (error != hipSuccess) { \
    std::fprintf(stderr, "%s\n", hipGetErrorString(error)); \
    return 2; \
  } \
} while (0)

__global__ void scalar_kernel(const float* input, float* output, size_t count) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index] * 2.0f + 1.0f;
}

__global__ void vector4_kernel(
    const float4* input, float4* output, size_t count4) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count4) {
    float4 value = input[index];
    output[index] = make_float4(
        value.x * 2.0f + 1.0f, value.y * 2.0f + 1.0f,
        value.z * 2.0f + 1.0f, value.w * 2.0f + 1.0f);
  }
}

static void launch(
    const std::string& variant, const float* input, float* output,
    size_t count, hipStream_t stream) {
  constexpr int threads = 256;
  if (variant == "vector4") {
    size_t count4 = count / 4;
    hipLaunchKernelGGL(
        vector4_kernel, dim3((count4 + threads - 1) / threads),
        dim3(threads), 0, stream, reinterpret_cast<const float4*>(input),
        reinterpret_cast<float4*>(output), count4);
  } else {
    hipLaunchKernelGGL(
        scalar_kernel, dim3((count + threads - 1) / threads),
        dim3(threads), 0, stream, input, output, count);
  }
}

int main(int argc, char** argv) {
  int count_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&count_devices));
  hipDeviceProp_t properties{};
  if (count_devices > 0) HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  if (argc == 2 && std::string(argv[1]) == "--probe") {
    std::printf(
        "{\"visible_devices\":%d,\"logical_device\":0,"
        "\"device_name\":\"%s\",\"warp_size\":%d}\n",
        count_devices, count_devices ? properties.name : "",
        count_devices ? properties.warpSize : 0);
    return count_devices == 1 ? 0 : 3;
  }
  if (argc != 3) {
    std::fprintf(stderr, "usage: smoke_harness ELEMENTS scalar|vector4\n");
    return 2;
  }
  size_t count = std::strtoull(argv[1], nullptr, 10);
  std::string variant = argv[2];
  if (count < 4 || count % 4 != 0 ||
      (variant != "scalar" && variant != "vector4")) {
    std::fprintf(stderr, "invalid arguments\n");
    return 2;
  }

  std::vector<float> host_input(count);
  std::vector<float> host_output(count);
  for (size_t i = 0; i < count; ++i) {
    host_input[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 17;
  }
  float *input = nullptr, *output = nullptr;
  HIP_CHECK(hipMalloc(&input, count * sizeof(float)));
  HIP_CHECK(hipMalloc(&output, count * sizeof(float)));
  HIP_CHECK(hipMemcpy(
      input, host_input.data(), count * sizeof(float), hipMemcpyHostToDevice));

  for (int i = 0; i < 10; ++i) launch(variant, input, output, count, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> samples;
  samples.reserve(50);
  hipEvent_t begin, end;
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  for (int i = 0; i < 50; ++i) {
    HIP_CHECK(hipEventRecord(begin));
    launch(variant, input, output, count, nullptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
    samples.push_back(milliseconds * 1000.0f);
  }
  HIP_CHECK(hipMemcpy(
      host_output.data(), output, count * sizeof(float), hipMemcpyDeviceToHost));
  bool passed = true;
  for (size_t i = 0; i < count; ++i) {
    float expected = host_input[i] * 2.0f + 1.0f;
    if (std::fabs(host_output[i] - expected) > 1e-5f) {
      passed = false;
      break;
    }
  }
  std::sort(samples.begin(), samples.end());
  float median = samples[samples.size() / 2];
  float p90 = samples[static_cast<size_t>(samples.size() * 0.9)];
  double seconds = median * 1e-6;
  double tflops = (2.0 * static_cast<double>(count)) / seconds / 1e12;
  double bandwidth = (8.0 * static_cast<double>(count)) / seconds / 1e9;
  std::printf(
      "{\"passed\":%s,\"variant\":\"%s\",\"elements\":%zu,"
      "\"visible_devices\":%d,\"device_name\":\"%s\","
      "\"median_us\":%.6f,\"p90_us\":%.6f,\"min_us\":%.6f,"
      "\"max_us\":%.6f,\"tflops\":%.9f,\"bandwidth_gb_s\":%.6f,"
      "\"warmup\":10,\"samples\":50}\n",
      passed ? "true" : "false", variant.c_str(), count, count_devices,
      properties.name, median, p90, samples.front(), samples.back(),
      tflops, bandwidth);
  hipEventDestroy(begin);
  hipEventDestroy(end);
  hipFree(input);
  hipFree(output);
  return passed ? 0 : 4;
}
