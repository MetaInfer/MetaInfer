#!/usr/bin/env python3
"""
Profiling script for inference-agent-system engine only.
Hardcodes engine path — no --engine-path argument needed.

Usage:
  torchrun --nproc_per_node=4 profiling_traces/profile_agent_engine.py \
    --model-dir /home/honglin/models/qwen/Qwen3-8B \
    --output-dir profiling_traces/trace_agent_engine \
    --max-tokens 12
"""
import time, os, sys, json, gc, argparse, re

os.environ['META_INFER_LOG_RANK0_ONLY'] = '1'
os.environ['META_INFER_CUDA_GRAPH'] = '0'

import torch
import torch.profiler as profiler
from pathlib import Path

ENGINE_PATH = '/home/honglin/inference-agent-system'
PROMPT = '苏州园林的特点是'


def run_profile(model_dir, output_dir, max_tokens):
    sys.path.insert(0, ENGINE_PATH)
    from llm_engine import LLMEngine

    rank = int(os.environ.get('RANK', 0))
    label = 'agent-engine'

    if rank == 0:
        print(f"[{label}] engine={ENGINE_PATH}, max_tokens={max_tokens}")

    engine = LLMEngine(
        model_dir=Path(model_dir),
        inference_backend='qwen_tp',
        max_num_seqs=4)

    # Warmup
    _ = engine.generate('你好', max_new_tokens=4, temperature=0.0)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 1: Clean wall time (no profiler)
    if rank == 0:
        print(f"[{label}] Phase 1: Clean wall time...")

    t0 = time.time()
    output = engine.generate(PROMPT, max_new_tokens=max_tokens, temperature=0.0)
    elapsed_clean = time.time() - t0
    torch.cuda.synchronize()

    if rank == 0:
        throughput_clean = max_tokens / elapsed_clean
        print(f"[{label}] Clean wall: {elapsed_clean*1000:.1f}ms, "
              f"throughput={throughput_clean:.1f} tok/s")

    # Phase 2: Profiled run (with_stack=False)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"[{label}] Phase 2: Profiled run ({max_tokens} tokens, with_stack=False)...")

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with profiler.record_function("generate_full"):
            output = engine.generate(PROMPT, max_new_tokens=max_tokens, temperature=0.0)

    torch.cuda.synchronize()

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

        # Chrome trace
        trace_path = os.path.join(output_dir, 'trace_rank0.json')
        prof.export_chrome_trace(trace_path)
        trace_size = os.path.getsize(trace_path)
        print(f"[{label}] Chrome trace: {trace_path} ({trace_size / 1024**2:.1f} MB)")

        # Key averages table
        key_table = prof.key_averages().table(sort_by="device_time_total", row_limit=80)
        key_path = os.path.join(output_dir, 'key_avg.txt')
        with open(key_path, 'w') as f:
            f.write(key_table)
        print(f"[{label}] Key averages: {key_path}")

        # Structured JSON
        events = prof.key_averages()
        events_data = []
        for evt in events:
            events_data.append({
                'name': evt.key,
                'cpu_time_total_us': evt.cpu_time_total,
                'device_time_total_us': evt.device_time_total,
                'self_cpu_time_total_us': evt.self_cpu_time_total,
                'self_device_time_total_us': evt.self_device_time_total,
                'count': evt.count,
            })
        json_path = os.path.join(output_dir, 'key_avg.json')
        with open(json_path, 'w') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False)

        # Parse totals from table output
        m_cpu = re.search(r"Self CPU time total:\s*([\d.]+)(s|ms|us)", key_table)
        m_cuda = re.search(r"Self CUDA time total:\s*([\d.]+)(s|ms|us)", key_table)

        def parse_total(match):
            if not match:
                return 0.0
            val, unit = float(match.group(1)), match.group(2)
            return val * 1_000_000 if unit == 's' else val * 1_000 if unit == 'ms' else val

        self_cpu_us = parse_total(m_cpu)
        self_cuda_us = parse_total(m_cuda)

        # Extract Custom AR kernels
        cross_device_cuda = 0.0
        custom_ar_cuda = 0.0
        custom_ar_events = []
        for evt in events:
            name = evt.key.lower()
            if 'cross_device_reduce_1stage' in evt.key:
                cross_device_cuda = evt.self_device_time_total
            if evt.key == '_C_custom_ar::all_reduce':
                custom_ar_cuda = evt.self_device_time_total
            if any(kw in name for kw in ['custom_ar', 'cross_device_reduce', 'all_reduce_sum']):
                custom_ar_events.append({
                    'name': evt.key,
                    'self_device_time_us': evt.self_device_time_total,
                    'device_time_total_us': evt.device_time_total,
                    'self_cpu_time_us': evt.self_cpu_time_total,
                    'count': evt.count,
                })

        # Extract key compute kernels
        def find_kernel(pattern):
            result = 0.0
            for evt in events:
                if pattern in evt.key:
                    result += evt.self_device_time_total
            return result

        aten_mm = find_kernel('aten::mm')
        cutlass = find_kernel('cutlass_80_tensorop_bf16_s16816gemm_relu')
        flash_attn = find_kernel('flash_attn')
        rms_norm = find_kernel('rms_norm')
        silu_and_mul = find_kernel('silu_and_mul')
        rotary = find_kernel('rotary_embedding')

        result = {
            'label': label,
            'engine_path': ENGINE_PATH,
            'prompt': PROMPT,
            'max_tokens': max_tokens,
            'elapsed_clean_ms': elapsed_clean * 1000,
            'throughput_clean_tok_s': throughput_clean,
            'output': output,
            'self_cpu_time_total_ms': self_cpu_us / 1000,
            'self_device_time_total_ms': self_cuda_us / 1000,
            'cross_device_reduce_1stage_ms': cross_device_cuda / 1000,
            'custom_ar_all_reduce_ms': custom_ar_cuda / 1000,
            'aten_mm_device_ms': aten_mm / 1000,
            'cutlass_device_ms': cutlass / 1000,
            'flash_attn_device_ms': flash_attn / 1000,
            'rms_norm_device_ms': rms_norm / 1000,
            'silu_and_mul_device_ms': silu_and_mul / 1000,
            'rotary_device_ms': rotary / 1000,
            'custom_ar_events': custom_ar_events,
        }
        res_path = os.path.join(output_dir, 'result.json')
        with open(res_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"[{label}] RESULTS (with_stack=False)")
        print(f"{'='*60}")
        print(f"  Clean wall:                {elapsed_clean*1000:.1f} ms")
        print(f"  Clean throughput:           {throughput_clean:.1f} tok/s")
        print(f"  Self CPU time total:       {self_cpu_us/1000:.1f} ms")
        print(f"  Self CUDA time total:      {self_cuda_us/1000:.1f} ms")
        print(f"  ---")
        print(f"  cross_device_reduce_1stage: {cross_device_cuda/1000:.1f} ms")
        print(f"  _C_custom_ar::all_reduce:   {custom_ar_cuda/1000:.1f} ms")
        print(f"  ---")
        print(f"  aten::mm:                  {aten_mm/1000:.1f} ms")
        print(f"  cutlass:                   {cutlass/1000:.1f} ms")
        print(f"  flash_attn:                {flash_attn/1000:.1f} ms")
        print(f"  rms_norm:                  {rms_norm/1000:.1f} ms")
        print(f"  silu_and_mul:              {silu_and_mul/1000:.1f} ms")
        print(f"  rotary_embedding:          {rotary/1000:.1f} ms")
        print(f"{'='*60}")

        # Custom AR event details
        print(f"\n[{label}] Custom AR events:")
        for evt in sorted(custom_ar_events, key=lambda e: -e['self_device_time_us']):
            print(f"  {evt['name']}: self_device={evt['self_device_time_us']/1000:.1f}ms, "
                  f"cpu={evt['self_cpu_time_us']/1000:.1f}ms, count={evt['count']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile inference-agent-system engine')
    parser.add_argument('--model-dir', default='/home/honglin/models/qwen/Qwen3-8B',
                        help='Path to model directory')
    parser.add_argument('--output-dir', default='profiling_traces/trace_agent_engine',
                        help='Output directory for traces and results')
    parser.add_argument('--max-tokens', type=int, default=12,
                        help='Number of tokens to generate (default: 12)')
    args = parser.parse_args()

    run_profile(args.model_dir, args.output_dir, args.max_tokens)
