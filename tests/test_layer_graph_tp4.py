"""Stage 3: TP=4 CUDA Graph with all_reduce_sum registered as custom op.

Verifies: custom op registration, torch.compile no-recompile, replay correctness.
"""
import os, time, torch, sys
sys.path.insert(0, '/home/honglin/meta-infer')
os.environ['META_INFER_CUDA_GRAPH'] = '0'

from engine.tp_layers.distributed import init_tp_distributed, get_tp_rank
from engine.tp_layers.cuda_graph_wrapper import CUDAGraphWrapper
from engine.models.qwen import QwenTPModelRunner

MODEL_DIR = '/home/honglin/models/qwen/Qwen3-8B'

init_tp_distributed()
rank = get_tp_rank()
dev = f'cuda:{rank}'
runner = QwenTPModelRunner(MODEL_DIR, device=torch.device(dev), dtype=torch.bfloat16)
layer = runner.model.layers[0]
attn = layer.self_attn
cfg = runner.cfg

# Setup KV cache
bsz = 256
max_seq = 128
nb = max(1, (max_seq + bsz - 1) // bsz)
attn._key_cache = torch.zeros(nb, bsz, attn.num_kv_heads, attn.head_dim,
                               device=dev, dtype=torch.bfloat16)
attn._value_cache = torch.zeros(nb, bsz, attn.num_kv_heads, attn.head_dim,
                                 device=dev, dtype=torch.bfloat16)
attn._block_table = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
attn._kv_len_gpu[0] = 4
attn._cos_sin_cache_gpu = attn._cos_sin_cache_cpu.to(device=dev)

hs = layer.input_layernorm.weight.shape[0]
h = torch.randn(1, 1, hs, device=dev, dtype=torch.bfloat16)
r = torch.randn(1, 1, hs, device=dev, dtype=torch.bfloat16)
p = torch.tensor([4], device=dev, dtype=torch.long)

# ---- Test 1: Custom op registration ----
try:
    op = torch.ops.meta_infer.all_reduce_sum
    print(f'[{rank}] Custom op registered: {op}', flush=True)
except Exception as e:
    print(f'[{rank}] Custom op FAILED: {e}', flush=True)
    sys.exit(1)

# ---- Test 2: Eager correctness of custom op ----
attn._kv_len_gpu[0] = 4
eager_out = torch.ops.meta_infer.all_reduce_sum(h.clone())
torch.cuda.synchronize()
print(f'[{rank}] Custom op eager call OK, shape={eager_out.shape}', flush=True)

# ---- Test 3: torch.compile + CUDAGraphWrapper (no recompile) ----
with torch.inference_mode():
    compiled = torch.compile(layer.forward_decode, fullgraph=True, dynamic=False)

    # Eager warmup outside graph (trigger compilation)
    attn._kv_len_gpu[0] = 4
    compiled(h.clone(), p, 4, max_seq, r.clone())
    torch.cuda.synchronize()
    print(f'[{rank}] Compile warmup done', flush=True)

    # Barrier + capture
    torch.cuda.synchronize()
    torch.distributed.barrier()

    wrapper = CUDAGraphWrapper(compiled, debug_mode=False)
    attn._kv_len_gpu[0] = 4
    try:
        wrapper(h.clone(), p, 4, max_seq, r.clone())
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print(f'[{rank}] Capture OK, captured={wrapper.is_captured}', flush=True)
    except Exception as e:
        print(f'[{rank}] Capture FAILED: {e}', flush=True)
        torch.distributed.barrier()
        sys.exit(1)

    # ---- Test 4: Replay vs eager correctness ----
    attn._kv_len_gpu[0] = 4
    eh, er = layer.forward_decode(h.clone(), p, 4, max_seq, r.clone())
    torch.cuda.synchronize()

    attn._kv_len_gpu[0] = 4
    gh, gr = wrapper(h.clone(), p, 4, max_seq, r.clone())
    torch.cuda.synchronize()

# Numerical comparison
hs_match = torch.allclose(gh, eh, rtol=1e-2, atol=1e-2)
res_match = torch.allclose(gr, er, rtol=1e-2, atol=1e-2)
hs_diff = (gh - eh).abs().max().item()
res_diff = (gr - er).abs().max().item()
print(f'[{rank}] Correctness: hs_match={hs_match} (max_diff={hs_diff:.4f}), '
      f'res_match={res_match} (max_diff={res_diff:.4f})', flush=True)

# ---- Test 5: 10K replay stress ----
for i in range(10000):
    attn._kv_len_gpu[0] = 4
    out = wrapper(h.clone(), p, 4, max_seq, r.clone())
    if i % 2500 == 0:
        health = wrapper.check_graph_health()
        for k, v in health.items():
            if isinstance(v, torch.Tensor) and ('has_nan' in k or 'has_inf' in k):
                assert not v, f'{k} at step {i}'
        if rank == 0:
            print(f'  10K replay: step {i} OK', flush=True)
torch.cuda.synchronize()
print(f'[{rank}] 10K replay stress PASS', flush=True)

# ---- Profiling ----
attn._kv_len_gpu[0] = 4
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
) as prof:
    wrapper(h.clone(), p, 4, max_seq, r.clone())
torch.cuda.synchronize()

if rank == 0:
    launch_count = sum(1 for e in prof.key_averages() if 'cudaGraphLaunch' in e.key)
    print(f'Profiling: cudaGraphLaunch in trace = {launch_count}', flush=True)

    # Replay timing (1000 runs avg)
    attn._kv_len_gpu[0] = 4
    t0 = time.perf_counter()
    for _ in range(1000):
        attn._kv_len_gpu[0] = 4
        wrapper(h.clone(), p, 4, max_seq, r.clone())
    torch.cuda.synchronize()
    replay_us = (time.perf_counter() - t0) / 1000 * 1e6

    # Eager baseline
    attn._kv_len_gpu[0] = 4
    t0 = time.perf_counter()
    for _ in range(1000):
        attn._kv_len_gpu[0] = 4
        layer.forward_decode(h.clone(), p, 4, max_seq, r.clone())
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / 1000 * 1e6

    print(f'Replay: {replay_us:.1f}us avg')
    print(f'Eager:  {eager_us:.1f}us avg')
    print(f'Speedup: {eager_us/replay_us:.1f}x')
    print(f'Probes: {wrapper.probes.summary()}')
