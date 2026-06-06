import time, os, sys, inspect
os.environ['META_INFER_LOG_RANK0_ONLY']='1'; os.environ['META_INFER_CUDA_GRAPH']='0'
from llm_engine import LLMEngine; from pathlib import Path
# 确认导入的是哪个文件
src = inspect.getfile(LLMEngine)
print(f"llm_engine from: {src}")
# assert "inference-agent-system" not in src, f"WRONG ENGINE: imported agent engine instead of meta-infer!"
engine = LLMEngine(model_dir=Path('/home/honglin/models/qwen/Qwen3-8B'), inference_backend='qwen_tp', max_num_seqs=4)
_ = engine.generate('你好', max_new_tokens=4, temperature=0.0)## warmup
t0 = time.time()
out = engine.generate('苏州园林的特点是', max_new_tokens=32, temperature=0.0)
elapsed = time.time() - t0
print(f'Elapsed: {elapsed:.3f}s, Throughput: {32/elapsed:.1f} tok/s, Correct: {out}')


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 checkagetn.py
