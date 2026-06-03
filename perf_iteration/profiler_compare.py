"""
Standalone profiler: generate 32 tokens with Qwen3-8B TP=4, dump trace.
Usage:
  # Meta-infer prototype
  cd /home/honglin/meta-infer && PYTHONPATH="$(pwd):$PYTHONPATH" torchrun --nproc_per_node=4 profiler_compare.py meta

  # Agent-generated engine
  cd /home/honglin/inference-agent-system && PYTHONPATH="$(pwd):$PYTHONPATH" torchrun --nproc_per_node=4 perf_iteration/profiler_compare.py agent
"""
import time, os, sys, json, gc
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
os.environ['META_INFER_CUDA_GRAPH'] = '0'

import torch
import torch.profiler as profiler
from pathlib import Path

MODEL_DIR = '/home/honglin/models/qwen/Qwen3-8B'
PROMPT = '苏州园林的特点是'
MAX_TOKENS = 32
OUTDIR = {
    'meta': '/home/honglin/inference-agent-system/perf_iteration/trace_meta',
    'agent': '/home/honglin/inference-agent-system/perf_iteration/trace_agent',
}

def run_profile(label, outdir):
    from llm_engine import LLMEngine
    src = __import__('inspect').getfile(LLMEngine)
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print(f"[{label}] llm_engine from: {src}")

    engine = LLMEngine(
        model_dir=Path(MODEL_DIR),
        inference_backend='qwen_tp',
        max_num_seqs=4)

    # Warmup
    _ = engine.generate('你好', max_new_tokens=4, temperature=0.0)
    if rank == 0:
        print(f"[{label}] warmup done, starting profiled run...")

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    activities = [
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ]

    with profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with profiler.record_function("generate_full"):
            _ = engine.generate(PROMPT, max_new_tokens=MAX_TOKENS, temperature=0.0)

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        # Chrome trace
        trace_path = os.path.join(outdir, f'trace_rank{rank}.json')
        prof.export_chrome_trace(trace_path)
        print(f"[{label}] Chrome trace saved to {trace_path}")

        # Key table
        key_path = os.path.join(outdir, 'key_avg.txt')
        with open(key_path, 'w') as f:
            f.write(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=50))
        print(f"[{label}] Key averages saved to {key_path}")

        # Print top 15 for immediate comparison
        print(f"\n[{label}] Top 15 CUDA time:\n")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=15))

if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'agent'
    label = {'meta': 'META-INFER', 'agent': 'AGENT-ENGINE'}.get(kind, kind)
    outdir = OUTDIR.get(kind, f'/tmp/trace_{kind}')
    run_profile(label, outdir)
