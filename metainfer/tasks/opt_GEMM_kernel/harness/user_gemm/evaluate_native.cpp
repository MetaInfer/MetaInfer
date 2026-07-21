#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(METAINFER_USE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
using GpuEvent = hipEvent_t;
using GpuStream = hipStream_t;
using DeviceBFloat16 = hip_bfloat16;
#define GPU_CHECK(call) gpu_check((call), #call)
static void gpu_check(hipError_t status, const char* call) {
  if (status != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(status));
}
#define GPU_MALLOC hipMalloc
#define GPU_FREE hipFree
#define GPU_MEMCPY hipMemcpy
#define GPU_H2D hipMemcpyHostToDevice
#define GPU_D2H hipMemcpyDeviceToHost
#define GPU_SYNC hipDeviceSynchronize
#define GPU_EVENT_CREATE hipEventCreate
#define GPU_EVENT_RECORD hipEventRecord
#define GPU_EVENT_SYNC hipEventSynchronize
#define GPU_EVENT_ELAPSED hipEventElapsedTime
#define GPU_EVENT_DESTROY hipEventDestroy
#define GPU_LAST_ERROR hipGetLastError
#define GPU_LAUNCH(kernel, grid, block, stream, ...) hipLaunchKernelGGL(kernel, grid, block, 0, stream, __VA_ARGS__)
__device__ static DeviceBFloat16 make_bf16(float value) { return DeviceBFloat16(value); }
#elif defined(METAINFER_USE_CUDA)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using GpuEvent = cudaEvent_t;
using GpuStream = cudaStream_t;
using DeviceBFloat16 = __nv_bfloat16;
#define GPU_CHECK(call) gpu_check((call), #call)
static void gpu_check(cudaError_t status, const char* call) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
#define GPU_MALLOC cudaMalloc
#define GPU_FREE cudaFree
#define GPU_MEMCPY cudaMemcpy
#define GPU_H2D cudaMemcpyHostToDevice
#define GPU_D2H cudaMemcpyDeviceToHost
#define GPU_SYNC cudaDeviceSynchronize
#define GPU_EVENT_CREATE cudaEventCreate
#define GPU_EVENT_RECORD cudaEventRecord
#define GPU_EVENT_SYNC cudaEventSynchronize
#define GPU_EVENT_ELAPSED cudaEventElapsedTime
#define GPU_EVENT_DESTROY cudaEventDestroy
#define GPU_LAST_ERROR cudaGetLastError
#define GPU_LAUNCH(kernel, grid, block, stream, ...) kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)
__device__ static DeviceBFloat16 make_bf16(float value) { return __float2bfloat16(value); }
#else
#error "A GPU backend definition is required"
#endif

namespace fs = std::filesystem;

struct Case {
  std::string id, op;
  int tp, m, n, k;
};

struct DeviceBuffers {
  int8_t *a = nullptr, *w = nullptr;
  float *a_scale = nullptr, *w_scale = nullptr;
  DeviceBFloat16 *y = nullptr, *reference = nullptr;
  DeviceBuffers() = default;
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  DeviceBuffers(DeviceBuffers&& other) noexcept
      : a(other.a), w(other.w), a_scale(other.a_scale), w_scale(other.w_scale),
        y(other.y), reference(other.reference) {
    other.a = nullptr; other.w = nullptr; other.a_scale = nullptr;
    other.w_scale = nullptr; other.y = nullptr; other.reference = nullptr;
  }
  ~DeviceBuffers() {
    if (a) GPU_FREE(a); if (w) GPU_FREE(w); if (a_scale) GPU_FREE(a_scale);
    if (w_scale) GPU_FREE(w_scale); if (y) GPU_FREE(y); if (reference) GPU_FREE(reference);
  }
};

static std::string env(const char* name) {
  const char* value = std::getenv(name);
  if (!value || !*value) throw std::runtime_error(std::string("missing environment variable: ") + name);
  return value;
}

template <typename T>
static std::vector<T> read_binary(const fs::path& path, size_t count) {
  if (!fs::is_regular_file(path) || fs::file_size(path) != count * sizeof(T)) {
    throw std::runtime_error("binary size mismatch: " + path.string());
  }
  std::vector<T> data(count);
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
  if (!in) throw std::runtime_error("cannot load " + path.string());
  return data;
}

static void validate_metadata(const std::string& json, const std::string& name,
                              const std::vector<int>& shape, const std::string& dtype,
                              size_t nbytes) {
  size_t key = json.find("\"" + name + "\"");
  if (key == std::string::npos) throw std::runtime_error("info.json is missing " + name);
  size_t begin = json.find('{', key), end = json.find('}', begin);
  if (begin == std::string::npos || end == std::string::npos) throw std::runtime_error("invalid metadata for " + name);
  std::string object = json.substr(begin, end - begin + 1); std::smatch match;
  if (!std::regex_search(object, match, std::regex("\\\"dtype\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"")) || match[1] != dtype)
    throw std::runtime_error("dtype mismatch for " + name);
  if (!std::regex_search(object, match, std::regex("\\\"nbytes\\\"\\s*:\\s*([0-9]+)")) || std::stoull(match[1]) != nbytes)
    throw std::runtime_error("nbytes mismatch for " + name);
  if (!std::regex_search(object, match, std::regex("\\\"shape\\\"\\s*:\\s*\\[([^\\]]*)\\]")))
    throw std::runtime_error("shape is missing for " + name);
  std::vector<int> actual; std::regex number("[0-9]+");
  for (std::sregex_iterator it(match[1].first, match[1].second, number), last; it != last; ++it) actual.push_back(std::stoi(it->str()));
  if (actual != shape) throw std::runtime_error("shape mismatch for " + name);
}

struct WeightStore {
  std::vector<int8_t> qa, qb, kv, o, w1, w2, w3;
  std::vector<float> qas, qbs, kvs, os, w1s, w2s, w3s;
  explicit WeightStore(const fs::path& root) {
    if (!fs::is_regular_file(root / "info.json")) throw std::runtime_error("weight directory has no info.json");
    std::ifstream info_stream(root / "info.json", std::ios::binary);
    std::string info(std::istreambuf_iterator<char>(info_stream), {});
    validate_metadata(info, "q_proj_a", {4096, 1024}, "int8", 4194304);
    validate_metadata(info, "q_proj_a_scale", {1024}, "float32", 4096);
    validate_metadata(info, "q_proj_b", {1024, 32768}, "int8", 33554432);
    validate_metadata(info, "q_proj_b_scale", {32768}, "float32", 131072);
    validate_metadata(info, "kv_proj", {4096, 512}, "int8", 2097152);
    validate_metadata(info, "kv_proj_scale", {512}, "float32", 2048);
    validate_metadata(info, "o_proj", {8192, 4096}, "int8", 33554432);
    validate_metadata(info, "o_proj_scale", {4096}, "float32", 16384);
    validate_metadata(info, "moe_w1", {4096, 2048}, "int8", 8388608);
    validate_metadata(info, "moe_w1_scale", {2048}, "float32", 8192);
    validate_metadata(info, "moe_w2", {2048, 4096}, "int8", 8388608);
    validate_metadata(info, "moe_w2_scale", {4096}, "float32", 16384);
    validate_metadata(info, "moe_w3", {4096, 2048}, "int8", 8388608);
    validate_metadata(info, "moe_w3_scale", {2048}, "float32", 8192);
    qa = read_binary<int8_t>(root / "q_proj_a.bin", 4096ull * 1024);
    qas = read_binary<float>(root / "q_proj_a_scale.bin", 1024);
    qb = read_binary<int8_t>(root / "q_proj_b.bin", 1024ull * 32768);
    qbs = read_binary<float>(root / "q_proj_b_scale.bin", 32768);
    kv = read_binary<int8_t>(root / "kv_proj.bin", 4096ull * 512);
    kvs = read_binary<float>(root / "kv_proj_scale.bin", 512);
    o = read_binary<int8_t>(root / "o_proj.bin", 8192ull * 4096);
    os = read_binary<float>(root / "o_proj_scale.bin", 4096);
    w1 = read_binary<int8_t>(root / "moe_w1.bin", 4096ull * 2048);
    w1s = read_binary<float>(root / "moe_w1_scale.bin", 2048);
    w2 = read_binary<int8_t>(root / "moe_w2.bin", 2048ull * 4096);
    w2s = read_binary<float>(root / "moe_w2_scale.bin", 4096);
    w3 = read_binary<int8_t>(root / "moe_w3.bin", 4096ull * 2048);
    w3s = read_binary<float>(root / "moe_w3_scale.bin", 2048);
  }

  void derive(const Case& c, std::vector<int8_t>& weight, std::vector<float>& scale) const {
    weight.clear(); scale.clear(); weight.reserve(static_cast<size_t>(c.k) * c.n); scale.reserve(c.n);
    if (c.op == "wqkv_a") {
      for (int row = 0; row < 4096; ++row) {
        weight.insert(weight.end(), qa.begin() + row * 1024, qa.begin() + (row + 1) * 1024);
        weight.insert(weight.end(), kv.begin() + row * 512, kv.begin() + (row + 1) * 512);
      }
      scale = qas; scale.insert(scale.end(), kvs.begin(), kvs.end());
    } else if (c.op == "wq_b") {
      int width = 32768 / c.tp;
      for (int row = 0; row < 1024; ++row)
        weight.insert(weight.end(), qb.begin() + row * 32768, qb.begin() + row * 32768 + width);
      scale.assign(qbs.begin(), qbs.begin() + width);
    } else if (c.op == "wo_b") {
      int depth = 8192 / c.tp;
      weight.assign(o.begin(), o.begin() + static_cast<size_t>(depth) * 4096); scale = os;
    } else if (c.op == "shared_gate_up_proj") {
      int width = 2048 / c.tp;
      for (int row = 0; row < 4096; ++row) {
        weight.insert(weight.end(), w1.begin() + row * 2048, w1.begin() + row * 2048 + width);
        weight.insert(weight.end(), w3.begin() + row * 2048, w3.begin() + row * 2048 + width);
      }
      scale.assign(w1s.begin(), w1s.begin() + width);
      scale.insert(scale.end(), w3s.begin(), w3s.begin() + width);
    } else if (c.op == "shared_down_proj") {
      int depth = 2048 / c.tp;
      weight.assign(w2.begin(), w2.begin() + static_cast<size_t>(depth) * 4096); scale = w2s;
    } else throw std::runtime_error("unknown workload " + c.op);
    if (weight.size() != static_cast<size_t>(c.k) * c.n || scale.size() != static_cast<size_t>(c.n))
      throw std::runtime_error("derived tensor shape mismatch for " + c.id);
  }
};

static std::vector<Case> public_cases() {
  struct Work { const char* id; const char* op; int tp, k, n; };
  const Work work[] = {
    {"wqkv-a-tp4", "wqkv_a", 4, 4096, 1536}, {"wq-b-tp4", "wq_b", 4, 1024, 8192},
    {"wo-b-tp4", "wo_b", 4, 2048, 4096},
    {"shared-gate-up-proj-tp4", "shared_gate_up_proj", 4, 4096, 1024},
    {"shared-down-proj-tp4", "shared_down_proj", 4, 512, 4096},
    {"wqkv-a-tp8", "wqkv_a", 8, 4096, 1536}, {"wq-b-tp8", "wq_b", 8, 1024, 4096},
    {"wo-b-tp8", "wo_b", 8, 1024, 4096},
    {"shared-gate-up-proj-tp8", "shared_gate_up_proj", 8, 4096, 512},
    {"shared-down-proj-tp8", "shared_down_proj", 8, 256, 4096},
  };
  const int ms[] = {1, 2, 4, 8, 16, 4096};
  std::vector<Case> result;
  for (const auto& w : work) for (int m : ms)
    result.push_back({std::string(w.id) + "-m" + std::to_string(m), w.op, w.tp, m, w.n, w.k});
  return result;
}

static std::vector<Case> correctness_cases() {
  auto result = public_cases();
  result.push_back({"heldout-wq-b-tp4-m7", "wq_b", 4, 7, 8192, 1024});
  result.push_back({"heldout-wo-b-tp8-m13", "wo_b", 8, 13, 4096, 1024});
  result.push_back({"heldout-shared-gate-up-proj-tp4-m3", "shared_gate_up_proj", 4, 3, 1024, 4096});
  result.push_back({"heldout-shared-down-proj-tp8-m7", "shared_down_proj", 8, 7, 4096, 256});
  return result;
}

static uint16_t float_to_bf16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}
static float bf16_to_float(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16; float out; std::memcpy(&out, &bits, sizeof(out)); return out;
}

static void prepare_activation(const Case& c, std::vector<int8_t>& q, std::vector<float>& scales) {
  std::seed_seq seq(c.id.begin(), c.id.end()); std::mt19937 rng(seq); std::normal_distribution<float> dist(0.f, 1.5f);
  q.resize(static_cast<size_t>(c.m) * c.k); scales.resize(c.m);
  std::vector<float> row(c.k);
  for (int m = 0; m < c.m; ++m) {
    float maximum = 0.f;
    for (int k = 0; k < c.k; ++k) { row[k] = bf16_to_float(float_to_bf16(dist(rng))); maximum = std::max(maximum, std::abs(row[k])); }
    float scale = maximum / 127.f; scales[m] = scale;
    for (int k = 0; k < c.k; ++k) q[static_cast<size_t>(m) * c.k + k] = static_cast<int8_t>(
      std::max(-127.f, std::min(127.f, std::nearbyint(scale == 0.f ? 0.f : row[k] / scale))));
  }
}

__global__ void reference_kernel(const int8_t* a, const int8_t* w, const float* as,
                                 const float* ws, DeviceBFloat16* y, int m, int n, int k) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(m) * n;
  if (index >= total) return;
  int row = static_cast<int>(index / n), col = static_cast<int>(index % n), acc = 0;
  for (int inner = 0; inner < k; ++inner) acc += static_cast<int>(a[static_cast<size_t>(row) * k + inner]) * static_cast<int>(w[static_cast<size_t>(inner) * n + col]);
  y[index] = make_bf16(static_cast<float>(acc) * as[row] * ws[col]);
}

using Launch = int (*)(const int8_t*, const int8_t*, const float*, const float*, void*, int, int, int, void*);
struct Candidate {
  void* handle = nullptr; Launch launch = nullptr;
  explicit Candidate(const fs::path& artifact) {
    fs::path library;
    for (const auto& entry : fs::recursive_directory_iterator(artifact))
      if (entry.is_regular_file() && entry.path().filename().string().find("libmetainfer_gemm_candidate") == 0) { library = entry.path(); break; }
    if (library.empty()) throw std::runtime_error("candidate shared library is missing");
    handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) throw std::runtime_error(dlerror());
    launch = reinterpret_cast<Launch>(dlsym(handle, "launch_w8a8_gemm"));
    if (!launch) throw std::runtime_error("candidate has no launch_w8a8_gemm symbol");
  }
  ~Candidate() { if (handle) dlclose(handle); }
};

static DeviceBuffers upload(const Case& c, const std::vector<int8_t>& a, const std::vector<int8_t>& w,
                            const std::vector<float>& as, const std::vector<float>& ws) {
  DeviceBuffers d;
  GPU_CHECK(GPU_MALLOC(reinterpret_cast<void**>(&d.a), a.size()));
  GPU_CHECK(GPU_MALLOC(reinterpret_cast<void**>(&d.w), w.size()));
  GPU_CHECK(GPU_MALLOC(reinterpret_cast<void**>(&d.a_scale), as.size() * sizeof(float)));
  GPU_CHECK(GPU_MALLOC(reinterpret_cast<void**>(&d.w_scale), ws.size() * sizeof(float)));
  size_t output_bytes = static_cast<size_t>(c.m) * c.n * sizeof(DeviceBFloat16);
  GPU_CHECK(GPU_MALLOC(reinterpret_cast<void**>(&d.y), output_bytes));
  GPU_CHECK(GPU_MALLOC(reinterpret_cast<void**>(&d.reference), output_bytes));
  GPU_CHECK(GPU_MEMCPY(d.a, a.data(), a.size(), GPU_H2D)); GPU_CHECK(GPU_MEMCPY(d.w, w.data(), w.size(), GPU_H2D));
  GPU_CHECK(GPU_MEMCPY(d.a_scale, as.data(), as.size() * sizeof(float), GPU_H2D));
  GPU_CHECK(GPU_MEMCPY(d.w_scale, ws.data(), ws.size() * sizeof(float), GPU_H2D));
  return d;
}

static std::string correctness_case(Candidate& candidate, const WeightStore& store, const Case& c) {
  std::vector<int8_t> a, w; std::vector<float> as, ws; prepare_activation(c, a, as); store.derive(c, w, ws);
  auto d = upload(c, a, w, as, ws); GpuStream stream = nullptr;
  size_t total = static_cast<size_t>(c.m) * c.n;
  GPU_LAUNCH(reference_kernel, dim3((total + 255) / 256), dim3(256), stream, d.a, d.w, d.a_scale, d.w_scale, d.reference, c.m, c.n, c.k);
  GPU_CHECK(GPU_LAST_ERROR());
  if (candidate.launch(d.a, d.w, d.a_scale, d.w_scale, d.y, c.m, c.n, c.k, stream) != 0) throw std::runtime_error("candidate returned non-zero");
  GPU_CHECK(GPU_SYNC());
  std::vector<uint16_t> got(total), expected(total);
  GPU_CHECK(GPU_MEMCPY(got.data(), d.y, total * 2, GPU_D2H)); GPU_CHECK(GPU_MEMCPY(expected.data(), d.reference, total * 2, GPU_D2H));
  size_t mismatches = 0; float max_abs = 0.f;
  for (size_t i = 0; i < total; ++i) {
    float x = bf16_to_float(got[i]), y = bf16_to_float(expected[i]), error = std::abs(x - y);
    if (!std::isfinite(x) || error > 1.0e-3f) ++mismatches;
    max_abs = std::max(max_abs, error);
  }
  size_t cpu_mismatches = 0; float cpu_max_abs = 0.f;
  std::vector<int> rows;
  if (c.m <= 16) for (int row = 0; row < c.m; ++row) rows.push_back(row);
  else rows = {0, c.m / 3, (2 * c.m) / 3, c.m - 1};
  int column_count = std::min(c.n, 64);
  for (int row : rows) for (int sample = 0; sample < column_count; ++sample) {
    int col = column_count == 1 ? 0 : static_cast<int>(std::llround(1.0 * sample * (c.n - 1) / (column_count - 1)));
    int64_t accumulator = 0;
    for (int inner = 0; inner < c.k; ++inner)
      accumulator += static_cast<int64_t>(a[static_cast<size_t>(row) * c.k + inner]) * w[static_cast<size_t>(inner) * c.n + col];
    float expected_cpu = bf16_to_float(float_to_bf16(static_cast<float>(accumulator) * as[row] * ws[col]));
    float measured = bf16_to_float(got[static_cast<size_t>(row) * c.n + col]);
    float error = std::abs(measured - expected_cpu);
    if (!std::isfinite(measured) || error > 1.0e-3f) ++cpu_mismatches;
    cpu_max_abs = std::max(cpu_max_abs, error);
  }
  mismatches += cpu_mismatches;
  std::ostringstream out; out << "{\"id\":\"" << c.id << "\",\"passed\":" << (mismatches == 0 ? "true" : "false")
    << ",\"mismatches\":" << mismatches << ",\"elements\":" << total << ",\"max_abs_error\":" << max_abs
    << ",\"cpu_int64_sentinel_passed\":" << (cpu_mismatches == 0 ? "true" : "false")
    << ",\"cpu_int64_sentinel_mismatches\":" << cpu_mismatches
    << ",\"cpu_int64_sentinel_max_abs_error\":" << cpu_max_abs << "}";
  return out.str();
}

static int json_integer(const std::string& json, const std::string& key) {
  std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([0-9]+)"); std::smatch match;
  if (!std::regex_search(json, match, pattern)) throw std::runtime_error("protocol has no " + key);
  return std::stoi(match[1]);
}

static std::string benchmark_case(Candidate& candidate, const WeightStore& store, const Case& c, int warmup, int samples) {
  std::vector<int8_t> a, w; std::vector<float> as, ws; prepare_activation(c, a, as); store.derive(c, w, ws);
  auto d = upload(c, a, w, as, ws); GpuStream stream = nullptr;
  for (int i = 0; i < warmup; ++i) if (candidate.launch(d.a, d.w, d.a_scale, d.w_scale, d.y, c.m, c.n, c.k, stream) != 0) throw std::runtime_error("candidate returned non-zero");
  GPU_CHECK(GPU_SYNC()); std::vector<float> values; values.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    GpuEvent start, stop; GPU_CHECK(GPU_EVENT_CREATE(&start)); GPU_CHECK(GPU_EVENT_CREATE(&stop));
    GPU_CHECK(GPU_EVENT_RECORD(start, stream));
    if (candidate.launch(d.a, d.w, d.a_scale, d.w_scale, d.y, c.m, c.n, c.k, stream) != 0) throw std::runtime_error("candidate returned non-zero");
    GPU_CHECK(GPU_EVENT_RECORD(stop, stream)); GPU_CHECK(GPU_EVENT_SYNC(stop)); float ms = 0.f;
    GPU_CHECK(GPU_EVENT_ELAPSED(&ms, start, stop)); GPU_CHECK(GPU_EVENT_DESTROY(start)); GPU_CHECK(GPU_EVENT_DESTROY(stop)); values.push_back(ms);
  }
  std::sort(values.begin(), values.end()); float latency = values[values.size() / 2];
  double flops = 2.0 * c.m * c.n * c.k;
  double bytes = 1.0 * c.m * c.k + 1.0 * c.k * c.n + 4.0 * c.m + 4.0 * c.n + 2.0 * c.m * c.n;
  std::ostringstream out; out << "{\"id\":\"" << c.id << "\",\"latency_ms\":" << std::setprecision(9) << latency
    << ",\"min_ms\":" << values.front() << ",\"max_ms\":" << values.back()
    << ",\"tops\":" << flops / (latency * 1e9) << ",\"bandwidth_gbps\":" << bytes / (latency * 1e6) << "}";
  return out.str();
}

// Hardware-profiler entrypoint.  Setup, activation generation/quantization,
// weight loading and all H2D copies happen before the one candidate launch,
// so the profiler can filter and attribute only the W8A8 GEMM kernel.
static void profile_case(Candidate& candidate, const WeightStore& store, const Case& c) {
  std::vector<int8_t> a, w; std::vector<float> as, ws;
  prepare_activation(c, a, as); store.derive(c, w, ws);
  auto d = upload(c, a, w, as, ws); GpuStream stream = nullptr;
  GPU_CHECK(GPU_SYNC());
  if (candidate.launch(d.a, d.w, d.a_scale, d.w_scale, d.y,
                       c.m, c.n, c.k, stream) != 0) {
    throw std::runtime_error("candidate returned non-zero");
  }
  GPU_CHECK(GPU_SYNC());
}

static void write_report(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path()); std::ofstream out(path); if (!out) throw std::runtime_error("cannot write report"); out << text;
}

int main(int argc, char** argv) {
  fs::path report;
  try {
    const bool is_profile = argc == 3 && std::string(argv[1]) == "profile";
    const bool is_evaluation = argc == 2 &&
      (std::string(argv[1]) == "correctness" || std::string(argv[1]) == "benchmark");
    if (!is_profile && !is_evaluation) throw std::runtime_error("usage: metainfer_gemm_harness correctness|benchmark|profile CASE_ID");
    std::string phase = argv[1]; report = env("METAINFER_REPORT_PATH");
    std::string profile_case_id = is_profile ? std::string(argv[2]) : "";
    if (phase != env("METAINFER_EVALUATION_PHASE")) throw std::runtime_error("phase mismatch");
    fs::path weight_root = fs::absolute(env("METAINFER_WEIGHT_BUNDLE"));
    fs::path artifact_root = fs::absolute(env("METAINFER_BUILD_ARTIFACT_DIR"));
    std::string protocol = phase == "benchmark" ? env("METAINFER_BENCHMARK_PROTOCOL") : "";
    WeightStore weights(weight_root);
    // Candidate code receives pointers and the public ABI only. Remove direct
    // evaluator paths/phase hints from its environment and working directory
    // before dlopen. Bundle digests are verified again by the parent process.
    const char* private_names[] = {
      "METAINFER_EVALUATOR_BUNDLE", "METAINFER_WEIGHT_BUNDLE", "METAINFER_WEIGHT_SHA256",
      "METAINFER_REPORT_PATH", "METAINFER_EVALUATION_PHASE", "METAINFER_EVALUATION_ROLE",
      "METAINFER_SUBMISSION_DIR", "METAINFER_BUILD_ARTIFACT_DIR",
      "METAINFER_BUILD_FINGERPRINT", "METAINFER_BENCHMARK_PROTOCOL"
    };
    for (const char* name : private_names) unsetenv(name);
    std::fill(argv[1], argv[1] + std::strlen(argv[1]), 'x');
    fs::current_path(artifact_root);
    Candidate candidate(artifact_root);
    std::ostringstream json; bool passed = true;
    if (phase == "profile") {
      auto cases = public_cases();
      auto found = std::find_if(cases.begin(), cases.end(), [&](const Case& c) {
        return c.id == profile_case_id;
      });
      if (found == cases.end()) throw std::runtime_error("unknown public profile case");
      profile_case(candidate, weights, *found);
      write_report(report, std::string("{\"passed\":true,\"case_id\":\"") + found->id +
        "\",\"timed_scope\":\"launch_w8a8_gemm_only\"}");
      return 0;
    }
    if (phase == "correctness") {
      json << "{\"passed\":true,\"reference\":\"frozen full GPU INT32 GEMM plus independent CPU INT64 sentinel points\",\"activation_quantization_timed\":false,\"cases\":[";
      auto cases = correctness_cases();
      for (size_t i = 0; i < cases.size(); ++i) { if (i) json << ','; auto item = correctness_case(candidate, weights, cases[i]); if (item.find("\"passed\":false") != std::string::npos) passed = false; json << item; }
      std::string text = json.str(); text.replace(text.find("\"passed\":true"), 13, passed ? "\"passed\":true" : "\"passed\":false"); text += "]}"; write_report(report, text); return 0;
    }
    int warmup = json_integer(protocol, "warmup"), samples = json_integer(protocol, "samples");
    json << "{\"passed\":true,\"methodology\":" << protocol << ",\"timed_scope\":\"launch_w8a8_gemm_only\",\"activation_quantization_timed\":false,\"weight_loading_or_preprocessing_timed\":false,\"cases\":[";
    auto cases = public_cases(); for (size_t i = 0; i < cases.size(); ++i) { if (i) json << ','; json << benchmark_case(candidate, weights, cases[i], warmup, samples); }
    json << "]}"; write_report(report, json.str()); return 0;
  } catch (const std::exception& error) {
    std::cerr << "GEMM harness failed: " << error.what() << '\n';
    if (!report.empty()) try { write_report(report, std::string("{\"passed\":false,\"reason\":\"") + error.what() + "\",\"cases\":[]}"); } catch (...) {}
    return 2;
  }
}
