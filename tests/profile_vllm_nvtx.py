"""vllm profiling with built-in NVTX tracing + nsys."""
import os, torch, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams

MODEL = '/home/honglin/models/qwen/Qwen3-0.6B'

print("[1] Loading vllm with NVTX tracing...", flush=True)
llm = LLM(
    model=MODEL,
    enforce_eager=True,
    enable_layerwise_nvtx_tracing=True,
    max_model_len=512,
    gpu_memory_utilization=0.5,
)
sampling = SamplingParams(temperature=0.0, max_tokens=8)

print("[2] Warmup...", flush=True)
out = llm.generate(['苏州园林的特点是'], sampling)
print(f"  warmup: {out[0].outputs[0].text!r}", flush=True)
torch.cuda.synchronize()

print("[3] Profiled run...", flush=True)
torch.cuda.nvtx.range_push("vllm_generate")
t0 = time.perf_counter()
out = llm.generate(['苏州园林的特点是'], sampling)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
torch.cuda.nvtx.range_pop()

print(f"[4] Done in {elapsed:.3f}s", flush=True)
print(f"  output: {out[0].outputs[0].text!r}", flush=True)
