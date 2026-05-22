"""Run vllm inference for nsys profiling."""
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams

MODEL = '/home/honglin/models/qwen/Qwen3-0.6B'
PROMPT = '苏州园林的特点是'

print("[nsys] loading model...", flush=True)
llm = LLM(model=MODEL, enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.5)
sampling = SamplingParams(temperature=0.0, max_tokens=8)

# Warmup
print("[nsys] warmup...", flush=True)
out = llm.generate([PROMPT], sampling)
print(f"  warmup: {out[0].outputs[0].text!r}", flush=True)
torch.cuda.synchronize()

# Add NVTX markers for nsys to see
torch.cuda.nvtx.range_push("vllm_generate")
import time
t0 = time.perf_counter()
out = llm.generate([PROMPT], sampling)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
torch.cuda.nvtx.range_pop()

print(f"[nsys] done in {elapsed:.3f}s, output: {out[0].outputs[0].text!r}", flush=True)
