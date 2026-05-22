"""Profile vLLM DeepSeek-V2-Lite with torch.profiler + nsys comparison."""
import os
import torch
from vllm import LLM, SamplingParams

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

MODEL_DIR = "/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPT = "苏州园林的特点是"
MAX_TOKENS = 24
OUTPUT_DIR = "/home/honglin/meta-infer/tests"

def main():
    print("[vllm-profiler] creating offline engine (TP=4)...")
    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.3,  # leave room for profiler
        trust_remote_code=True,
        enforce_eager=True,  # disable CUDA graphs for clean profiling
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Warmup — let torch.compile finish
    print("[vllm-profiler] warmup...")
    outputs = llm.generate([PROMPT], sampling)
    print(f"[vllm-profiler] warmup output: {outputs[0].outputs[0].text[:60]}...")

    # Profiled run
    print("[vllm-profiler] profiling with torch.profiler...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        outputs = llm.generate([PROMPT], sampling)
        torch.cuda.synchronize()

    print(f"[vllm-profiler] output: {outputs[0].outputs[0].text!r}")
    print()

    # Export Chrome trace
    trace_path = os.path.join(OUTPUT_DIR, "trace_vllm.json")
    prof.export_chrome_trace(trace_path)
    print(f"[vllm-profiler] Chrome trace saved: {trace_path}")

    # Top CUDA kernels
    print()
    print("=" * 90)
    print("Top 20 CUDA Kernels by Time")
    print("=" * 90)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=80))

    # Flash-attention related kernels
    print()
    print("=" * 90)
    print("Flash-Attention / MLA Related Kernels")
    print("=" * 90)
    for evt in prof.key_averages():
        name = evt.key.lower()
        if any(k in name for k in ['flash', 'fwd_kernel', 'varlen', 'flash_attn',
                                     'mla', 'triton', 'sdpa', 'scaled_dot',
                                     'efficient_attention', 'cutlass', 'bmm']):
            cpu_ms = evt.cpu_time_total / 1e3
            cuda_ms = evt.cuda_time_total / 1e3 if evt.cuda_time_total else 0
            print(f"  {evt.key:<70s} cpu={cpu_ms:>8.2f}ms  cuda={cuda_ms:>8.2f}ms  count={evt.count}")

if __name__ == "__main__":
    main()
