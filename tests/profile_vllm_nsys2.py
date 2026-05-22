"""vllm profiling with manual cudaProfilerStart/Stop for subprocess injection."""
import os, torch, time

# --- CRITICAL: Start CUDA profiler in this process BEFORE vllm imports ---
# This ensures CUPTI is loaded before any CUDA context is created
torch.cuda.cudart().cudaProfilerStart()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams

MODEL = '/home/honglin/models/qwen/Qwen3-0.6B'
PROMPT = '苏州园林的特点是'

print("[1] Loading vllm for nsys profiling...", flush=True)
torch.cuda.nvtx.range_push("vllm_init")
llm = LLM(model=MODEL, enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.5)
torch.cuda.nvtx.range_pop()
sampling = SamplingParams(temperature=0.0, max_tokens=8)

# Warmup to compile any JIT
print("[2] Warmup...", flush=True)
torch.cuda.nvtx.range_push("vllm_warmup")
out = llm.generate([PROMPT], sampling)
torch.cuda.nvtx.range_pop()
print(f"  warmup: {out[0].outputs[0].text!r}", flush=True)
torch.cuda.synchronize()

# This is the region we want to profile
print("[3] Profiled generate...", flush=True)
torch.cuda.nvtx.range_push("vllm_generate")
t0 = time.perf_counter()
out = llm.generate([PROMPT], sampling)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
torch.cuda.nvtx.range_pop()

print(f"[4] Done in {elapsed:.3f}s", flush=True)
print(f"  output: {out[0].outputs[0].text!r}", flush=True)

torch.cuda.cudart().cudaProfilerStop()
