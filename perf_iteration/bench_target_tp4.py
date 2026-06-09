import os, sys, time
# Auto-detect project root from this script's location
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
os.environ['META_INFER_CUDA_GRAPH'] = '0'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'

from llm_engine import LLMEngine
from pathlib import Path
import torch

engine = LLMEngine(
    model_dir=Path(os.environ["MODEL_DIR"]),
    inference_backend='qwen_tp',
    max_num_seqs=4,
)

world_size = int(os.environ.get('WORLD_SIZE', '1'))
rank = int(os.environ.get('RANK', '0'))
if rank == 0:
    print(f'[INFO] WORLD_SIZE={world_size}, TP={"TP=4" if world_size > 1 else "single GPU"}')

# warmup
_ = engine.generate('苏州园林的特点是', max_new_tokens=8, temperature=0.0)
torch.cuda.synchronize()

# timed run
torch.cuda.synchronize()
t0 = time.perf_counter()
out = engine.generate('苏州园林的特点是', max_new_tokens=24, temperature=0.0)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

if rank == 0:
    print(f'Output: {out!r}')
    print(f'Elapsed: {elapsed:.3f}s  |  Throughput: {24/elapsed:.1f} tok/s')
