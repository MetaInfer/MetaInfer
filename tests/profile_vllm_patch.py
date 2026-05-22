"""Profile vllm by patching child process to run torch.profiler internally."""
import os, torch, time, sys

# Force CUDA profiling in ALL processes via environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

# Patch multiprocessing to inject profiler
import multiprocessing
_orig_process_run = multiprocessing.Process.run

def _profiled_run(self):
    """Patched Process.run that enables CUDA profiling."""
    torch.cuda.cudart().cudaProfilerStart()
    try:
        _orig_process_run(self)
    finally:
        torch.cuda.cudart().cudaProfilerStop()

multiprocessing.Process.run = _profiled_run

from vllm import LLM, SamplingParams

MODEL = '/home/honglin/models/qwen/Qwen3-0.6B'
PROMPT = '苏州园林的特点是'

print("[1] Loading vllm (with patched subprocess profiling)...", flush=True)
llm = LLM(model=MODEL, enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.5)
sampling = SamplingParams(temperature=0.0, max_tokens=8)

print("[2] Warmup...", flush=True)
out = llm.generate([PROMPT], sampling)
print(f"  warmup: {out[0].outputs[0].text!r}", flush=True)
torch.cuda.synchronize()

print("[3] Profiles generate (check nsys output)...", flush=True)
out = llm.generate([PROMPT], sampling)
torch.cuda.synchronize()
print(f"  output: {out[0].outputs[0].text!r}", flush=True)
