"""
vLLM Eager TP=4 standalone throughput benchmark — no profiling overhead.
Runs multiple rounds with GPU cooldown for reliable measurement.
"""
import os
import time


def main():
    MODEL_DIR = os.environ.get("MODEL_DIR", "/data/xinference/cache/Qwen3-8B")
    PROMPT = "苏州园林的特点是"
    MAX_TOKENS = 24
    NUM_ROUNDS = 10
    COOLDOWN_SEC = 60

    print(f"=== vLLM Eager TP=4 Standalone Benchmark ===")
    print(f"Model:      {MODEL_DIR}")
    print(f"Prompt:     {PROMPT!r}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Rounds:     {NUM_ROUNDS}")
    print(f"Cooldown:   {COOLDOWN_SEC}s between rounds")
    print()

    from vllm import LLM, SamplingParams
    import torch

    # Init
    print("[init] Loading model...")
    t_load_start = time.perf_counter()

    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=4,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    t_load_done = time.perf_counter()
    print(f"[init] Model loaded in {t_load_done - t_load_start:.1f}s")

    # Warmup run (not counted)
    print("[warmup] Running warmup generation...")
    _ = llm.generate([PROMPT], sampling_params)
    torch.cuda.synchronize()
    print("[warmup] Done.\n")

    # Benchmark rounds
    times = []
    throughputs = []
    outputs_list = []

    for r in range(1, NUM_ROUNDS + 1):
        # GPU cooldown (skip for first round since warmup just finished)
        if r > 1:
            print(f"[cooldown] Sleeping {COOLDOWN_SEC}s for GPU cooling...")
            time.sleep(COOLDOWN_SEC)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate([PROMPT], sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        output_text = outputs[0].outputs[0].text
        tok_s = MAX_TOKENS / elapsed

        times.append(elapsed)
        throughputs.append(tok_s)
        outputs_list.append(output_text)

        print(f"  Round {r:2d}: {elapsed:.3f}s  ({tok_s:.1f} tok/s)  output={output_text!r}")

    # Statistics
    import statistics
    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times)
    mean_tps = statistics.mean(throughputs)
    std_tps = statistics.stdev(throughputs)

    print(f"\n=== Results ({NUM_ROUNDS} rounds, standalone, no alternating) ===")
    print(f"Mean time:       {mean_t:.3f}s ± {std_t:.3f}s")
    print(f"Mean throughput:  {mean_tps:.1f} tok/s ± {std_tps:.1f} tok/s")
    print(f"Min:              {min(times):.3f}s ({max(throughputs):.1f} tok/s)")
    print(f"Max:              {max(times):.3f}s ({min(throughputs):.1f} tok/s)")

    # Verify all outputs identical
    first_out = outputs_list[0]
    all_same = all(o == first_out for o in outputs_list)
    print(f"\nOutput correctness: {'ALL IDENTICAL' if all_same else 'MISMATCH'}")
    if not all_same:
        for i, o in enumerate(outputs_list):
            if o != first_out:
                print(f"  Round {i+1} differs: {o!r}")

    print(f"\n=== For comparison with previous alternating report ===")
    print(f"Previous alternating: vLLM mean 1.153s (20.8 tok/s) over 5 rounds")
    print(f"Current standalone:   vLLM mean {mean_t:.3f}s ({mean_tps:.1f} tok/s) over {NUM_ROUNDS} rounds")
    print(f"Delta vs alternating: {((mean_tps / 20.8) - 1) * 100:+.1f}%")


if __name__ == "__main__":
    main()
