# Why: Phase 11 吞吐基线——单 GPU nocompile ≥ 54 tok/s。
#   Real engine nocompile baseline: 55.7 tok/s (CLAUDE.md §4)
# What failure: throughput < 54 tok/s → 性能优化不达标
# Superpowers gate: CLAUDE.md rule 2 (No speculative — 54 tok/s from real engine)
# Human review: [待人类Diff]
import time, os, sys
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'; os.environ['META_INFER_CUDA_GRAPH'] = '0'

print("=== Phase 11: Throughput Baseline ===")
print("THROUGHPUT-001: Measuring single GPU throughput...")

from llm_engine import LLMEngine; from pathlib import Path
engine = LLMEngine(model_dir=Path('${MODEL_DIR}'), inference_backend='qwen_tp', max_num_seqs=4)

NUM_TOKENS = 32
t0 = time.time()
out = engine.generate('苏州园林的特点是', max_new_tokens=NUM_TOKENS, temperature=0.0)
elapsed = time.time() - t0
tps = NUM_TOKENS / elapsed

MIN_TPS = 54  # TP=4 nocompile target, aligned with meta-infer 55.7 tok/s baseline
passed = tps >= MIN_TPS
expected = '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
correct = out == expected

print(f"  Tokens: {NUM_TOKENS}")
print(f"  Elapsed: {elapsed:.3f}s")
print(f"  Throughput: {tps:.1f} tok/s (target: ≥{MIN_TPS})")
print(f"  Correctness: {'PASS' if correct else 'FAIL'} ({out[:30]}...)")

assert tps >= MIN_TPS, (
    f"THROUGHPUT-001: {tps:.1f} tok/s < {MIN_TPS} tok/s。"
    f"Phase 11 性能优化未达标。检查 P1-P6 是否全部应用。"
)
assert correct, f"THROUGHPUT-002: 性能优化引入了正确性回归"
print("PHASE11_THROUGHPUT: PASS")
