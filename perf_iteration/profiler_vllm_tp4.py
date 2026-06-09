"""
vLLM Eager TP=4 profiler — adapted from ref_projects/vllm/examples/offline_inference/simple_profiling.py
Uses llm.start_profile()/stop_profile() to capture GPU traces from EngineCore subprocess.
"""
import os
import time


def main():
    MODEL_DIR = os.environ.get("MODEL_DIR", "")
    TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_baseline")
    PROMPT = "苏州园林的特点是"
    MAX_TOKENS = 24

    os.makedirs(TRACE_DIR, exist_ok=True)
    for f in os.listdir(TRACE_DIR):
        os.remove(os.path.join(TRACE_DIR, f))

    print(f"=== vLLM Eager TP=4 Profiler ===")
    print(f"Model:   {MODEL_DIR}")
    print(f"Trace:   {TRACE_DIR}")
    print(f"Prompt:  {PROMPT}")
    print(f"Tokens:  {MAX_TOKENS}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=4,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": TRACE_DIR,
        },
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    llm.start_profile()

    t0 = time.perf_counter()
    outputs = llm.generate([PROMPT], sampling_params)
    elapsed = time.perf_counter() - t0

    llm.stop_profile()

    output_text = outputs[0].outputs[0].text
    print(f"\nOutput:  {output_text!r}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {MAX_TOKENS / elapsed:.1f} tok/s")

    # Wait for background process to finish writing profile
    time.sleep(10)

    # List output files
    print(f"\nTrace files:")
    for f in sorted(os.listdir(TRACE_DIR)):
        fpath = os.path.join(TRACE_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
