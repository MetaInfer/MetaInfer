#include <torch/extension.h>
#include <torch/library.h>

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <tuple>


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
    hipStream_t stream);

extern "C" void launch_pack_w8a8_weight(
    const int8_t* raw_weight,
    const float* weight_scale,
    int8_t* packed_weight,
    float* packed_weight_scale,
    int k,
    int n,
    hipStream_t stream);


namespace {

std::tuple<at::Tensor, at::Tensor> pack_weight_impl(
    const at::Tensor& raw_weight,
    const at::Tensor& weight_scale) {
  auto packed_weight = at::empty_like(raw_weight);
  auto packed_weight_scale = at::empty_like(weight_scale);
  const int k = static_cast<int>(raw_weight.size(0));
  const int n = static_cast<int>(raw_weight.size(1));
  const auto device_index = raw_weight.device().index();
  const hipStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  launch_pack_w8a8_weight(
      raw_weight.data_ptr<int8_t>(),
      weight_scale.data_ptr<float>(),
      packed_weight.data_ptr<int8_t>(),
      packed_weight_scale.data_ptr<float>(),
      k,
      n,
      stream);
  return std::make_tuple(packed_weight, packed_weight_scale);
}

at::Tensor gemm_out_impl(
    const at::Tensor& x_q,
    const at::Tensor& packed_weight,
    const at::Tensor& x_scale,
    const at::Tensor& packed_weight_scale,
    at::Tensor out,
    const at::Tensor& workspace) {
  const int m = static_cast<int>(x_q.size(0));
  const int k = static_cast<int>(x_q.size(1));
  const int n = static_cast<int>(out.size(1));
  const auto device_index = x_q.device().index();
  const hipStream_t stream =
      c10::cuda::getCurrentCUDAStream(device_index).stream();
  launch_w8a8_gemm(
      x_q.data_ptr<int8_t>(),
      packed_weight.data_ptr<int8_t>(),
      x_scale.data_ptr<float>(),
      packed_weight_scale.data_ptr<float>(),
      out.data_ptr(),
      workspace.data_ptr<uint8_t>(),
      static_cast<int64_t>(workspace.numel()),
      m,
      n,
      k,
      stream);
  return out;
}

}  // namespace


TORCH_LIBRARY(zth_w8a8, m) {
  m.def(
      "pack_weight(Tensor raw_weight, Tensor weight_scale) "
      "-> (Tensor, Tensor)");
  m.def(
      "gemm_out(Tensor x_q, Tensor packed_weight, Tensor x_scale, "
      "Tensor packed_weight_scale, Tensor(a!) out, Tensor(b!) workspace) "
      "-> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(zth_w8a8, CUDA, m) {
  m.impl("pack_weight", pack_weight_impl);
  m.impl("gemm_out", gemm_out_impl);
}

// Keep setup.py builds importable as well as torch.ops.load_library compatible.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
