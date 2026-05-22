"""Analyze PyTorch profiler traces to compare P2 vs P3-FA attention kernels."""
import json
import sys
from collections import defaultdict

def load_trace(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def analyze_attention(trace_data):
    """Extract attention-related kernel stats from trace."""
    events = trace_data.get("traceEvents", [])

    # Collect kernel timing
    kernel_stats = defaultdict(lambda: {"count": 0, "total_dur_us": 0})
    attn_events = []

    for evt in events:
        if evt.get("ph") != "X":  # Only complete events
            continue
        name = evt.get("name", "")
        dur = evt.get("dur", 0)

        # Check if attention-related
        name_lower = name.lower()
        is_attn = any(k in name_lower for k in [
            "flash_attn", "flash_fwd", "sdpa", "scaled_dot_product",
            "efficient_attention", "bmm", "softmax", "pad", "attention"
        ])

        if is_attn:
            kernel_stats[name]["count"] += 1
            kernel_stats[name]["total_dur_us"] += dur
            attn_events.append({
                "name": name,
                "dur_us": dur,
                "cat": evt.get("cat", ""),
                "args": {k: str(v)[:100] for k, v in evt.get("args", {}).items()},
            })

    return kernel_stats, attn_events

def print_summary(label, kernel_stats):
    print(f"\n{'='*60}")
    print(f"  {label} — Attention Operations Summary")
    print(f"{'='*60}")

    # Sort by total time
    sorted_kernels = sorted(kernel_stats.items(), key=lambda x: x[1]["total_dur_us"], reverse=True)

    total_attn_us = sum(v["total_dur_us"] for v in kernel_stats.values())

    print(f"\n  Total attention time: {total_attn_us/1000:.2f} ms")
    print(f"\n  {'Kernel':<70} {'Count':>6} {'Total(ms)':>10} {'Avg(us)':>10} {'%':>6}")
    print(f"  {'-'*102}")

    for name, stats in sorted_kernels[:15]:
        pct = (stats["total_dur_us"] / total_attn_us * 100) if total_attn_us > 0 else 0
        avg_us = stats["total_dur_us"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {name[:70]:<70} {stats['count']:>6} {stats['total_dur_us']/1000:>10.2f} {avg_us:>10.1f} {pct:>5.1f}%")

def compare(p2_stats, p3_stats):
    print(f"\n{'='*60}")
    print(f"  P2 vs P3-FA Comparison")
    print(f"{'='*60}")

    p2_total = sum(v["total_dur_us"] for v in p2_stats.values())
    p3_total = sum(v["total_dur_us"] for v in p3_stats.values())

    print(f"\n  P2    total attention: {p2_total/1000:.2f} ms")
    print(f"  P3-FA total attention: {p3_total/1000:.2f} ms")
    print(f"  Difference: {(p3_total - p2_total)/1000:+.2f} ms ({(p3_total/p2_total - 1)*100:+.1f}%)")

    # Find unique kernels in each
    p2_names = set(p2_stats.keys())
    p3_names = set(p3_stats.keys())

    only_p2 = p2_names - p3_names
    only_p3 = p3_names - p2_names
    common = p2_names & p3_names

    if only_p2:
        print(f"\n  Only in P2 ({len(only_p2)}):")
        for name in sorted(only_p2):
            s = p2_stats[name]
            print(f"    {name[:60]}: {s['count']}x, {s['total_dur_us']/1000:.2f}ms")

    if only_p3:
        print(f"\n  Only in P3-FA ({len(only_p3)}):")
        for name in sorted(only_p3):
            s = p3_stats[name]
            print(f"    {name[:60]}: {s['count']}x, {s['total_dur_us']/1000:.2f}ms")

    if common:
        print(f"\n  Common kernels ({len(common)}):")
        for name in sorted(common):
            s2 = p2_stats[name]
            s3 = p3_stats[name]
            diff = s3["total_dur_us"] - s2["total_dur_us"]
            if abs(diff) > 1000:  # Only show if > 1ms difference
                print(f"    {name[:60]}:")
                print(f"      P2: {s2['count']}x {s2['total_dur_us']/1000:.2f}ms | P3-FA: {s3['count']}x {s3['total_dur_us']/1000:.2f}ms | diff: {diff/1000:+.2f}ms")

def main():
    p2_path = "/home/honglin/meta-infer/tests/trace_p2.json"
    p3_path = "/home/honglin/meta-infer/tests/trace_p3_fa.json"

    print("Loading P2 trace...")
    p2_data = load_trace(p2_path)
    p2_stats, _ = analyze_attention(p2_data)
    print_summary("P2 Baseline", p2_stats)

    print("\nLoading P3-FA trace...")
    p3_data = load_trace(p3_path)
    p3_stats, _ = analyze_attention(p3_data)
    print_summary("P3-FA (Flash Attention)", p3_stats)

    compare(p2_stats, p3_stats)

if __name__ == "__main__":
    main()


