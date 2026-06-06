"""Benchmark: compare line 524 .item() removal throughput."""
import os, time, sys
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
os.environ['META_INFER_CUDA_GRAPH'] = '0'

from llm_engine import LLMEngine
from pathlib import Path

MODEL_DIR = '/home/honglin/models/qwen/Qwen3-8B'
WARMUP_TOKENS = 4
BENCH_TOKENS = 32
ITERATIONS = 10

def main():
    engine = LLMEngine(
        model_dir=Path(MODEL_DIR),
        inference_backend='qwen_tp',
        max_num_seqs=4,
    )
    # Warmup
    _ = engine.generate('你好', max_new_tokens=WARMUP_TOKENS, temperature=0.0)

    times = []
    for i in range(ITERATIONS):
        t0 = time.time()
        out = engine.generate('苏州园林的特点是', max_new_tokens=BENCH_TOKENS, temperature=0.0)
        elapsed = time.time() - t0
        tok_s = BENCH_TOKENS / elapsed
        times.append(elapsed)
        if i < 5 or i >= ITERATIONS - 2:
            print(f'  Iter {i}: {elapsed:.4f}s, {tok_s:.1f} tok/s')

    avg = sum(times) / len(times)
    avg_tok = BENCH_TOKENS / avg
    print(f'\n  Avg ({ITERATIONS} runs): {avg:.4f}s, {avg_tok:.1f} tok/s')
    return avg_tok

if __name__ == '__main__':
    main()
