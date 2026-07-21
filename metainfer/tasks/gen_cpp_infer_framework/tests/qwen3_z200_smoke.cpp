#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" hipError_t qwen3_z200_launch_add(
        float*, const float*, const float*, int, hipStream_t);
extern "C" hipError_t qwen3_z200_launch_rms_norm(
        float*, const float*, const float*, int, int, float, hipStream_t);
extern "C" hipError_t qwen3_z200_launch_greedy_sample(
        const float*, int, int*, hipStream_t);

static void check(hipError_t status, const char* where) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, hipGetErrorString(status));
        std::exit(1);
    }
}

int main() {
    constexpr int n = 8;
    const float a[n] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float b[n] = {8, 7, 6, 5, 4, 3, 2, 1};
    const float weight[n] = {1, 1, 1, 1, 1, 1, 1, 1};

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_weight = nullptr;
    float* device_out = nullptr;
    int* device_token = nullptr;
    check(hipMalloc(&device_a, sizeof(a)), "hipMalloc(a)");
    check(hipMalloc(&device_b, sizeof(b)), "hipMalloc(b)");
    check(hipMalloc(&device_weight, sizeof(weight)), "hipMalloc(weight)");
    check(hipMalloc(&device_out, sizeof(a)), "hipMalloc(out)");
    check(hipMalloc(&device_token, sizeof(int)), "hipMalloc(token)");
    check(hipMemcpy(device_a, a, sizeof(a), hipMemcpyHostToDevice), "copy(a)");
    check(hipMemcpy(device_b, b, sizeof(b), hipMemcpyHostToDevice), "copy(b)");
    check(hipMemcpy(
            device_weight, weight, sizeof(weight), hipMemcpyHostToDevice),
            "copy(weight)");

    check(qwen3_z200_launch_add(
            device_out, device_a, device_b, n, nullptr), "launch(add)");
    float out[n];
    check(hipMemcpy(out, device_out, sizeof(out), hipMemcpyDeviceToHost),
          "copy(add result)");
    for (float value : out) {
        if (std::fabs(value - 9.0f) > 1e-6f) return 2;
    }

    check(qwen3_z200_launch_rms_norm(
            device_out, device_a, device_weight, 1, n, 1e-6f, nullptr),
            "launch(rms_norm)");
    check(hipMemcpy(out, device_out, sizeof(out), hipMemcpyDeviceToHost),
          "copy(rms_norm result)");
    float sum_squares = 0.0f;
    for (float value : a) sum_squares += value * value;
    const float inverse_rms = 1.0f / std::sqrt(sum_squares / n + 1e-6f);
    for (int i = 0; i < n; ++i) {
        if (std::fabs(out[i] - a[i] * inverse_rms) > 1e-5f) return 3;
    }

    check(qwen3_z200_launch_greedy_sample(
            device_a, n, device_token, nullptr), "launch(greedy)");
    int token = -1;
    check(hipMemcpy(
            &token, device_token, sizeof(token), hipMemcpyDeviceToHost),
            "copy(greedy result)");
    if (token != 7) return 4;

    int device_count = 0;
    check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    std::printf("PASS devices=%d add=9 rms0=%.6f greedy=%d\n",
                device_count, out[0], token);

    check(hipFree(device_a), "hipFree(a)");
    check(hipFree(device_b), "hipFree(b)");
    check(hipFree(device_weight), "hipFree(weight)");
    check(hipFree(device_out), "hipFree(out)");
    check(hipFree(device_token), "hipFree(token)");
    return 0;
}
