"""Profile attention kernels with torch.profiler.

Usage:
    # DeepSeek-V2-Lite TP=1
    PYTHONPATH=/home/honglin/meta-infer:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=0 \
    python meta-infer/profile_attn.py --model deepseek --seq-len 512

    # Qwen3-8B TP=1
    PYTHONPATH=/home/honglin/meta-infer:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=0 \
    python meta-infer/profile_attn.py --model qwen --seq-len 512

Output:
    - Console: top CUDA kernels by time
    - Chrome trace: profile_attn_trace.json (open in chrome://tracing)
    - TensorBoard: profile_attn_tb/ (optional)
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'


def profile_qwen(seq_len: int, warmup: int = 3, active: int = 5):
    """Profile Qwen3-8B single layer attention."""
    from models.qwen import QwenAttentionTP, QwenTPConfig

    cfg = QwenTPConfig.from_model_dir(Path('/home/honglin/models/qwen/Qwen3-8B'))
    attn = QwenAttentionTP(cfg).cuda().to(torch.bfloat16).eval()

    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
    positions = torch.arange(seq_len, device='cuda')

    # Warmup
    for _ in range(warmup):
        out, cache = attn(hidden, positions, past_key_values=None, max_seq_len=seq_len + 100)
    torch.cuda.synchronize()

    # Profile prefill
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(active):
            out, cache = attn(hidden, positions, past_key_values=None, max_seq_len=seq_len + 100)
            torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print(f"Qwen3 Prefill (seqlen={seq_len}) — Top CUDA Kernels by Time")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=80,
    ))

    # Profile decode (single token)
    hidden_dec = torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
    pos_dec = torch.tensor([seq_len], device='cuda')

    for _ in range(warmup):
        out, cache = attn(hidden_dec, pos_dec, past_key_values=cache, max_seq_len=seq_len + 100)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(active):
            out, cache = attn(hidden_dec, pos_dec, past_key_values=cache, max_seq_len=seq_len + 100)
            torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print(f"Qwen3 Decode (kv_len={seq_len}) — Top CUDA Kernels by Time")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=80,
    ))

    # Save trace
    prof.export_chrome_trace("profile_attn_trace_qwen.json")
    print("\nTrace saved to profile_attn_trace_qwen.json (open chrome://tracing)")


def profile_deepseek(seq_len: int, warmup: int = 3, active: int = 5):
    """Profile DeepSeek-V2-Lite single layer attention."""
    from models.deepseek_v2 import DeepseekAttentionTP, DeepseekV2TPConfig

    cfg = DeepseekV2TPConfig.from_model_dir(
        Path('/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat')
    )
    attn = DeepseekAttentionTP(cfg).cuda().to(torch.bfloat16).eval()

    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
    positions = torch.arange(seq_len, device='cuda')

    # Warmup
    for _ in range(warmup):
        out, cache = attn(hidden, positions, past_key_values=None, max_seq_len=seq_len + 100)
    torch.cuda.synchronize()

    # Profile prefill
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(active):
            out, cache = attn(hidden, positions, past_key_values=None, max_seq_len=seq_len + 100)
            torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print(f"DeepSeek-V2-Lite Prefill (seqlen={seq_len}) — Top CUDA Kernels by Time")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=80,
    ))

    # Profile decode
    hidden_dec = torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16, device='cuda')
    pos_dec = torch.tensor([seq_len], device='cuda')

    for _ in range(warmup):
        out, cache = attn(hidden_dec, pos_dec, past_key_values=cache, max_seq_len=seq_len + 100)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(active):
            out, cache = attn(hidden_dec, pos_dec, past_key_values=cache, max_seq_len=seq_len + 100)
            torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print(f"DeepSeek-V2-Lite Decode (kv_len={seq_len}) — Top CUDA Kernels by Time")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_name_column_width=80,
    ))

    prof.export_chrome_trace("profile_attn_trace_deepseek.json")
    print("\nTrace saved to profile_attn_trace_deepseek.json (open chrome://tracing)")


def profile_end_to_end(model_type: str, seq_len: int, num_tokens: int = 8):
    """Profile full end-to-end generation (multiple layers)."""
    from llm_engine import LLMEngine

    model_map = {
        'deepseek': '/home/honglin/models/deepseek-ai/DeepSeek-V2-Lite-Chat',
        'qwen': '/home/honglin/models/qwen/Qwen3-8B',
    }
    backend_map = {
        'deepseek': 'deepseek_tp',
        'qwen': 'qwen_tp',
    }

    engine = LLMEngine(
        model_dir=Path(model_map[model_type]),
        inference_backend=backend_map[model_type],
        max_num_seqs=1,
    )

    prompt = "Hello " * (seq_len // 2)  # ~seq_len tokens

    # Warmup
    engine.generate(prompt, max_new_tokens=2, temperature=1.0)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=num_tokens, repeat=1
        ),
    ) as prof:
        for step in range(num_tokens + 2):
            if step == 0:
                out = engine.generate(prompt, max_new_tokens=num_tokens, temperature=1.0)
            prof.step()
            torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print(f"{model_type} E2E (seqlen={seq_len}, gen={num_tokens}) — Top CUDA Kernels")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_name_column_width=80,
    ))

    # Filter for flash-attn kernels specifically
    print("\n" + "=" * 80)
    print("Flash-Attention Related Kernels Only")
    print("=" * 80)
    fa_events = []
    for evt in prof.key_averages():
        name = evt.key.lower()
        if any(k in name for k in ['flash', 'fwd', 'varlen', 'softmax', 'attention']):
            fa_events.append(evt)
    for evt in sorted(fa_events, key=lambda e: e.cuda_time_total, reverse=True):
        print(f"  {evt.key:<80s} cuda={evt.cuda_time_total/1e3:>8.1f}us  calls={evt.count}")

    trace_file = f"profile_e2e_{model_type}.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace saved to {trace_file} (open chrome://tracing)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile attention kernels')
    parser.add_argument('--model', choices=['qwen', 'deepseek'], required=True)
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length for profiling')
    parser.add_argument('--mode', choices=['layer', 'e2e'], default='layer',
                        help='layer = single attention layer, e2e = full generation')
    parser.add_argument('--num-tokens', type=int, default=8, help='Tokens to generate (e2e mode)')
    args = parser.parse_args()

    if args.mode == 'layer':
        if args.model == 'qwen':
            profile_qwen(args.seq_len)
        else:
            profile_deepseek(args.seq_len)
    else:
        profile_end_to_end(args.model, args.seq_len, args.num_tokens)
