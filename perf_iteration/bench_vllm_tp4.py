#!/usr/bin/env python3
"""vLLM Eager TP=4 — one-shot throughput benchmark (warmup + 1 timed run)."""
import os, time, sys
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['NCCL_DEBUG'] = 'WARN'          # suppress NCCL INFO noise
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '0'  # suppress heartbeat logs

def main():
    from vllm import LLM, SamplingParams
    import torch

    model_dir = os.environ.get("MODEL_DIR", "/data/xinference/cache/Qwen3-8B")

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=4,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=24)

    # warmup
    _ = llm.generate(['苏州园林的特点是'], sp)
    torch.cuda.synchronize()

    # timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate(['苏州园林的特点是'], sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"Output: {out[0].outputs[0].text!r}")
    print(f'Elapsed: {elapsed:.3f}s  |  Throughput: {24/elapsed:.1f} tok/s')

if __name__ == '__main__':
    main()
