"""Profile DeepSeek-V2-Lite TP=4 inference with PyTorch Profiler."""
import os
import sys
import torch
import torch.distributed as dist

os.environ["META_INFER_LOG_RANK0_ONLY"] = "1"

# Must import after env setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_engine import LLMEngine
from pathlib import Path

MODEL_DIR = Path("/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat")
PROMPT = "苏州园林的特点是"
MAX_NEW_TOKENS = 24
OUTPUT_DIR = Path("/home/honglin/meta-infer/tests")

def main():
    rank = int(os.environ.get("RANK", "0"))
    engine = LLMEngine(model_dir=MODEL_DIR, inference_backend="deepseek_tp", max_num_seqs=4)

    # Warmup (compile kernels)
    if rank == 0:
        print("[profiler] warmup...")
    out = engine.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    if rank == 0:
        print(f"[profiler] warmup output: {out!r}")

    # Profile run
    if rank == 0:
        print("[profiler] starting profile...")
        tag = os.environ.get("PROFILE_TAG", "p2")
        trace_path = str(OUTPUT_DIR / f"trace_{tag}.json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            out = engine.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)

        prof.export_chrome_trace(trace_path)
        print(f"[profiler] trace saved to {trace_path}")
        print(f"[profiler] output: {out!r}")

        # Print top ops by time
        print("\n[profiler] Top 30 ops by self CPU time:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

        # Print attention-related ops
        print("\n[profiler] Attention-related operations:")
        for evt in prof.key_averages():
            name = evt.key.lower()
            if any(k in name for k in ["flash", "sdpa", "scaled_dot", "attention", "bmm", "softmax", "pad"]):
                cpu_ms = evt.cpu_time_total / 1e3
                print(f"  {evt.key}: cpu_time={cpu_ms:.3f}ms count={evt.count}")
    else:
        # Non-rank-0: just run to stay in sync
        out = engine.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)

if __name__ == "__main__":
    main()
