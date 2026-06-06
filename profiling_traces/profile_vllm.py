#!/usr/bin/env python3
"""
vLLM profiling: eager vs CUDA graph, Qwen3-8B, TP=4, 12 output tokens.
Uses vLLM's built-in profiler — worker processes dump key_averages to
profiler_out_*.txt files which we parse directly.

Usage:
  conda run -n meta python profile_vllm.py --mode eager
  conda run -n meta python profile_vllm.py --mode cudagraph
  conda run -n meta python profile_vllm.py --mode both
"""
import time, os, sys, json, argparse, re, shutil, glob

os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from vllm import LLM, SamplingParams

PROMPT = '苏州园林的特点是'
MAX_TOKENS = 12
MODEL_DIR = '/home/honglin/models/qwen/Qwen3-8B'


def parse_profiler_out(txt_path):
    """Parse vLLM's profiler_out_*.txt (key_averages table) into structured data."""
    with open(txt_path) as f:
        text = f.read()

    # Extract Self CPU / Self CUDA totals
    m_cpu = re.search(r'Self CPU time total:\s*([\d.]+)(s|ms|us)', text)
    m_cuda = re.search(r'Self CUDA time total:\s*([\d.]+)(s|ms|us)', text)

    def parse_t(match):
        if not match: return 0.0
        val, unit = float(match.group(1)), match.group(2)
        return val * 1e6 if unit == 's' else val * 1e3 if unit == 'ms' else val

    self_cpu_us = parse_t(m_cpu)
    self_cuda_us = parse_t(m_cuda)

    # Parse individual event rows
    # Format: name  self_cpu%  self_cpu  cpu_total%  cpu_total  cpu_avg  self_cuda  self_cuda%  cuda_total  cuda_avg  #calls
    events = []
    for line in text.split('\n'):
        # Match numeric columns after a long name
        m = re.match(
            r'\s*(.+?)\s+'
            r'([\d.]+)%\s+([\d.]+)(s|ms|us)\s+'
            r'([\d.]+)%\s+([\d.]+)(s|ms|us)\s+'
            r'([\d.]+)(s|ms|us)\s+'
            r'([\d.]+)(s|ms|us)\s+'
            r'([\d.]+)%\s+([\d.]+)(s|ms|us)\s+'
            r'([\d.]+)(s|ms|us)\s+'
            r'(\d+)', line)
        if m:
            name = m.group(1).strip()
            scpu_pct = float(m.group(2))
            scpu_val = float(m.group(3))
            scpu_unit = m.group(4)
            cputotal_pct = float(m.group(5))
            cputotal_val = float(m.group(6))
            cputotal_unit = m.group(7)
            cpu_avg_val = float(m.group(8))
            cpu_avg_unit = m.group(9)
            scuda_val = float(m.group(10))
            scuda_unit = m.group(11)
            scuda_pct = float(m.group(12))
            cudatotal_val = float(m.group(13))
            cudatotal_unit = m.group(14)
            cuda_avg_val = float(m.group(15))
            cuda_avg_unit = m.group(16)
            count = int(m.group(17))

            def to_us(val, unit):
                return val * 1e6 if unit == 's' else val * 1e3 if unit == 'ms' else val

            events.append({
                'name': name,
                'self_cpu_us': to_us(scpu_val, scpu_unit),
                'self_cpu_pct': scpu_pct,
                'cpu_total_us': to_us(cputotal_val, cputotal_unit),
                'self_cuda_us': to_us(scuda_val, scuda_unit),
                'self_cuda_pct': scuda_pct,
                'cuda_total_us': to_us(cudatotal_val, cudatotal_unit),
                'count': count,
            })

    # Compute aggregate categories
    def sum_cuda(pattern):
        return sum(e['self_cuda_us'] for e in events if re.search(pattern, e['name'], re.IGNORECASE))

    def sum_cpu(pattern):
        return sum(e['self_cpu_us'] for e in events if re.search(pattern, e['name'], re.IGNORECASE))

    top_gpu = sorted(
        [(e['name'], e['self_cuda_us'], e['count']) for e in events if e['self_cuda_us'] > 0],
        key=lambda x: -x[1]
    )[:20]

    return {
        'self_cpu_time_total_us': self_cpu_us,
        'self_cuda_time_total_us': self_cuda_us,
        'nccl_us': sum_cuda(r'nccl.*allreduce|allreduce.*nccl'),
        'all_gather_us': sum_cuda(r'nccl.*allgather|allgather'),
        'aten_mm_us': sum_cuda(r'^aten::mm$'),
        'cutlass_us': sum_cuda(r'cutlass'),
        'gemm_us': sum_cuda(r'gemm|ampere.*gemm'),
        'flash_attn_us': sum_cuda(r'flash_attn|flash_fwd'),
        'rms_norm_us': sum_cuda(r'rms_norm'),
        'silu_us': sum_cuda(r'silu'),
        'rotary_us': sum_cuda(r'rotary'),
        'allreduce_cpu_us': sum_cpu(r'all_reduce'),
        'aten_mm_cpu_us': sum_cpu(r'^aten::mm$'),
        'cudaLaunchKernel_us': sum_cpu(r'cudaLaunchKernel'),
        'top_gpu': top_gpu,
        'top_cpu': sorted(
            [(e['name'], e['self_cpu_us'], e['count']) for e in events if e['self_cpu_us'] > 0],
            key=lambda x: -x[1]
        )[:20],
        'raw_events': events,
        'raw_text': text,
    }


def profile(mode, output_dir):
    enforce_eager = mode == 'eager'
    label = f'vllm-{mode}'
    print(f"\n{'='*60}")
    print(f"[{label}] Loading model (enforce_eager={enforce_eager})...")
    print(f"{'='*60}")

    # Clean up previous profiler output
    prof_dir = f'/tmp/vllm_prof_{mode}'
    if os.path.exists(prof_dir):
        shutil.rmtree(prof_dir)

    # Phase 1: LLM without profiler for clean wall
    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=4,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        enforce_eager=enforce_eager,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    print(f"[{label}] Warmup...")
    _ = llm.generate([PROMPT], sp)
    torch.cuda.synchronize()

    print(f"[{label}] Phase 1: Clean wall...")
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate([PROMPT], sp)
    torch.cuda.synchronize()
    elapsed_clean = time.time() - t0

    output_text = outputs[0].outputs[0].text
    num_tokens = len(outputs[0].outputs[0].token_ids)
    throughput_clean = num_tokens / elapsed_clean
    print(f"[{label}] Clean wall: {elapsed_clean*1000:.1f}ms, "
          f"{num_tokens} tokens, {throughput_clean:.1f} tok/s -> '{output_text}'")

    # Phase 2: Profiled run via vLLM built-in profiler
    del llm
    torch.cuda.empty_cache()
    import gc; gc.collect()

    print(f"[{label}] Phase 2: Profiled run...")
    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=4,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        enforce_eager=enforce_eager,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": prof_dir,
            "torch_profiler_with_stack": False,
            "torch_profiler_dump_cuda_time_total": True,
            "torch_profiler_record_shapes": True,
        },
    )
    _ = llm.generate([PROMPT], sp)  # warmup with profiler LLM
    torch.cuda.synchronize()

    llm.start_profile()
    try:
        outputs = llm.generate([PROMPT], sp)
    finally:
        llm.stop_profile()
    torch.cuda.synchronize()
    time.sleep(3)  # let background threads flush

    # Parse profiler output
    prof_out_files = sorted(glob.glob(os.path.join(prof_dir, 'profiler_out_*.txt')))
    if not prof_out_files:
        print(f"[{label}] ERROR: No profiler_out_*.txt found in {prof_dir}")
        print(f"  Contents: {os.listdir(prof_dir)}")
        return {}

    # Use rank 0 output (all ranks see the same GPUs)
    prof_data = parse_profiler_out(prof_out_files[0])
    print(f"[{label}] Parsed {prof_out_files[0]}")

    # Copy all profile outputs to output dir
    os.makedirs(output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(prof_dir, '*')):
        if os.path.isfile(f):
            shutil.copy2(f, output_dir)

    s = prof_data
    result = {
        'mode': label,
        'enforce_eager': enforce_eager,
        'prompt': PROMPT,
        'max_tokens': MAX_TOKENS,
        'num_tokens_generated': num_tokens,
        'output': output_text,
        'elapsed_clean_ms': elapsed_clean * 1000,
        'throughput_clean_tok_s': throughput_clean,
        'self_cpu_time_total_ms': s['self_cpu_time_total_us'] / 1000,
        'self_cuda_time_total_ms': s['self_cuda_time_total_us'] / 1000,
        'nccl_allreduce_device_ms': s['nccl_us'] / 1000,
        'nccl_allgather_device_ms': s['all_gather_us'] / 1000,
        'aten_mm_device_ms': s['aten_mm_us'] / 1000,
        'cutlass_device_ms': s['cutlass_us'] / 1000,
        'gemm_device_ms': s['gemm_us'] / 1000,
        'flash_attn_device_ms': s['flash_attn_us'] / 1000,
        'rms_norm_device_ms': s['rms_norm_us'] / 1000,
        'silu_device_ms': s['silu_us'] / 1000,
        'rotary_device_ms': s['rotary_us'] / 1000,
        'allreduce_cpu_ms': s['allreduce_cpu_us'] / 1000,
        'aten_mm_cpu_ms': s['aten_mm_cpu_us'] / 1000,
        'cudaLaunchKernel_us': s['cudaLaunchKernel_us'] / 1000,
        'top_gpu_kernels': [
            {'name': n, 'self_device_ms': d/1000, 'count': c}
            for n, d, c in s['top_gpu']
        ],
        'top_cpu_events': [
            {'name': n, 'self_cpu_ms': d/1000, 'count': c}
            for n, d, c in s['top_cpu']
        ],
    }
    res_path = os.path.join(output_dir, 'result.json')
    with open(res_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n[{label}] RESULTS SUMMARY")
    print(f"  Clean wall:              {elapsed_clean*1000:.1f} ms")
    print(f"  Throughput:              {throughput_clean:.1f} tok/s")
    print(f"  Self CPU time total:     {s['self_cpu_time_total_us']/1000:.1f} ms")
    print(f"  Self CUDA time total:    {s['self_cuda_time_total_us']/1000:.1f} ms")
    print(f"  --- GPU kernels ---")
    print(f"  NCCL AllReduce:          {s['nccl_us']/1000:.1f} ms")
    print(f"  NCCL AllGather:          {s['all_gather_us']/1000:.1f} ms")
    print(f"  aten::mm:                {s['aten_mm_us']/1000:.1f} ms")
    print(f"  cutlass:                 {s['cutlass_us']/1000:.1f} ms")
    print(f"  GEMM (all):              {s['gemm_us']/1000:.1f} ms")
    print(f"  Flash Attn:              {s['flash_attn_us']/1000:.1f} ms")
    print(f"  RMS Norm:                {s['rms_norm_us']/1000:.1f} ms")
    print(f"  SiLU:                    {s['silu_us']/1000:.1f} ms")
    print(f"  Rotary:                  {s['rotary_us']/1000:.1f} ms")
    print(f"  --- CPU events ---")
    print(f"  all_reduce dispatch:     {s['allreduce_cpu_us']/1000:.1f} ms")
    print(f"  aten::mm dispatch:       {s['aten_mm_cpu_us']/1000:.1f} ms")
    print(f"  cudaLaunchKernel:        {s['cudaLaunchKernel_us']/1000:.1f} ms")
    print(f"  --- Top 12 GPU kernels:")
    for n, d, c in s['top_gpu'][:12]:
        print(f"    {n[:80]}: {d/1000:.2f}ms ({c}x)")

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='both',
                       choices=['eager', 'cudagraph', 'both'])
    parser.add_argument('--output-dir', default='notebooks-cn/07_improvementPlan/traces')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    results = {}
    if args.mode in ('eager', 'both'):
        out = os.path.join(script_dir, args.output_dir, 'vllm_eager')
        results['eager'] = profile('eager', out)

    if args.mode in ('cudagraph', 'both'):
        out = os.path.join(script_dir, args.output_dir, 'vllm_cudagraph')
        results['cudagraph'] = profile('cudagraph', out)

    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON: vLLM eager vs vLLM CUDA Graph")
        print(f"{'='*60}")
        print(f"{'Metric':<30} {'eager':>12} {'cuda-graph':>14} {'ratio':>10}")
        print(f"{'-'*30} {'-'*12} {'-'*14} {'-'*10}")
        for key, label in [
            ('elapsed_clean_ms', 'Clean wall (ms)'),
            ('throughput_clean_tok_s', 'Throughput (tok/s)'),
            ('self_cpu_time_total_ms', 'Self CPU (ms)'),
            ('self_cuda_time_total_ms', 'Self CUDA (ms)'),
            ('nccl_allreduce_device_ms', 'NCCL AllReduce (ms)'),
            ('aten_mm_device_ms', 'aten::mm (ms)'),
            ('gemm_device_ms', 'GEMM total (ms)'),
            ('flash_attn_device_ms', 'Flash Attn (ms)'),
            ('rms_norm_device_ms', 'RMS Norm (ms)'),
            ('silu_device_ms', 'SiLU (ms)'),
            ('rotary_device_ms', 'Rotary (ms)'),
        ]:
            v_eager = results['eager'].get(key, 0)
            v_graph = results['cudagraph'].get(key, 0)
            ratio = v_eager / v_graph if v_graph > 0 else float('inf')
            print(f"{label:<30} {v_eager:>12.1f} {v_graph:>14.1f} {ratio:>10.2f}x")
