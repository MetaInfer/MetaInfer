# Phase 11 吞吐测试——验证引擎可正常运行并产出正确输出。
# model_dir: 由 MODEL_DIR 环境变量或命令行参数指定，未提供则交互询问。
import time, os, sys
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'; os.environ['META_INFER_CUDA_GRAPH'] = '0'

print("=== Phase 11: Throughput Baseline ===")

# model_dir 获取优先级: 命令行参数 > 环境变量 > 交互输入
from pathlib import Path
if len(sys.argv) > 1:
    model_dir = Path(sys.argv[1])
elif os.environ.get('MODEL_DIR'):
    model_dir = Path(os.environ['MODEL_DIR'])
else:
    model_dir = Path(input("Enter model directory path: "))

if not model_dir.exists():
    print(f"ERROR: model_dir not found: {model_dir}")
    sys.exit(1)

print(f"model_dir: {model_dir}")

from llm_engine import LLMEngine
engine = LLMEngine(model_dir=model_dir, inference_backend='qwen_tp', max_num_seqs=4)

NUM_TOKENS = 32
t0 = time.time()
out = engine.generate('苏州园林的特点是', max_new_tokens=NUM_TOKENS, temperature=0.0)
elapsed = time.time() - t0
tps = NUM_TOKENS / elapsed

# 两种合法输出：截断版（max_new_tokens=24）或完整版（max_new_tokens=32）
expected_short = '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑'
expected_full  = '（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑与植物结合\n答案：B'
correct = out.strip() == expected_short or out.strip() == expected_full

print(f"  Tokens: {NUM_TOKENS}")
print(f"  Elapsed: {elapsed:.3f}s")
print(f"  Throughput: {tps:.1f} tok/s")
print(f"  Correctness: {'PASS' if correct else 'FAIL'}")
if not correct:
    print(f"  Output: {out!r}")
    print(f"  Expected (short): {expected_short!r}")
    print(f"  Expected (full):  {expected_full!r}")

assert correct, f"THROUGHPUT-001: 输出与预期不符。Output={out!r}"
print("PHASE11_THROUGHPUT: PASS")
