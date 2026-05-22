"""Profile vLLM DeepSeek-V2-Lite inference using offline LLM API (same process)."""
import os
import torch
from vllm import LLM, SamplingParams

MODEL_DIR = "/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPT = "苏州园林的特点是"
MAX_TOKENS = 24
OUTPUT_DIR = "/home/honglin/meta-infer/tests"

def main():
    print("[vllm-profiler] loading model...")
    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enforce_eager=True,  # disable CUDA graphs for clean profiling
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Warmup
    print("[vllm-profiler] warmup...")
    outputs = llm.generate([PROMPT], sampling)
    print(f"[vllm-profiler] warmup: {outputs[0].outputs[0].text[:50]}...")

    # Profile
    print("[vllm-profiler] profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        outputs = llm.generate([PROMPT], sampling)

    trace_path = os.path.join(OUTPUT_DIR, "trace_vllm.json")
    prof.export_chrome_trace(trace_path)
    print(f"[vllm-profiler] trace saved to {trace_path}")
    print(f"[vllm-profiler] output: {outputs[0].outputs[0].text!r}")

    # Print top ops
    print("\n[vllm-profiler] Top 30 ops by self CPU time:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    # Print attention-related ops
    print("\n[vllm-profiler] Attention-related operations:")
    for evt in prof.key_averages():
        name = evt.key.lower()
        if any(k in name for k in ["flash", "sdpa", "scaled_dot", "attention", "bmm", "softmax", "pad", "triton", "mla"]):
            cpu_ms = evt.cpu_time_total / 1e3
            print(f"  {evt.key}: cpu_time={cpu_ms:.3f}ms count={evt.count}")

if __name__ == "__main__":
    main()
