"""
Phase 11 — Throughput alignment test.

Measures end-to-end throughput (tok/s) for single GPU / TP=4 and verifies
correctness against the Phase 10 greedy-decode baseline.

Gates (from inference_blueprint.json performance_gate):
  THROUGHPUT-001: Single GPU throughput >= 3.0 tok/s (baseline ~3.9 tok/s)
  THROUGHPUT-002: Output matches Phase 10 greedy baseline exactly
  THROUGHPUT-003: TP=4 throughput reported (when torchrun detected)

Usage:
  MODEL_DIR=/path/to/model python scripts/test_phase11_throughput.py
  MODEL_DIR=/path/to/model torchrun --nproc_per_node=4 scripts/test_phase11_throughput.py

Trace Source: physical_trace_tp4_rank0.json [runtime] Phase 10 baseline 3.9 tok/s
"""

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
_root = os.environ.get(
    "AGENT_INFER_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, _root)

os.environ.setdefault("META_INFER_LOG_RANK0_ONLY", "1")
os.environ.setdefault("META_INFER_CUDA_GRAPH", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import torch
from llm_engine import LLMEngine

# ---------------------------------------------------------------------------
# Test parameters (pinned to Phase 10 greedy-decode baseline)
# ---------------------------------------------------------------------------
PROMPT = "苏州园林的特点是"
EXPECTED = "（ ） A：建筑与园林结合 B：建筑与自然结合 C：建筑与山水结合 D：建筑"
MAX_TOKENS = 24          # Match the expected output length
WARMUP_TOKENS = 8        # Absorb one-time buffer allocations
MIN_THROUGHPUT = 3.0     # tok/s — well below the ~3.9 baseline
TRACE_SRC = "physical_trace_tp4_rank0.json [runtime] Phase 10 baseline"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_rank = int(os.environ.get("RANK", 0))
_world_size = int(os.environ.get("WORLD_SIZE", "1"))
_is_tp = _world_size > 1


def _log(msg: str) -> None:
    if _rank == 0:
        print(msg, flush=True)


def _fail(tag: str, detail: str) -> int:
    _log(f"[{tag}] FAIL: {detail}")
    return 1


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run() -> int:
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        return _fail("THROUGHPUT-001", "MODEL_DIR not set. export MODEL_DIR=<model weights dir>")

    _log("=== Phase 11: Throughput Alignment Test ===")
    _log(f"TP mode:  {'TP=' + str(_world_size) if _is_tp else 'single GPU'}")
    _log(f"Model:    {model_dir}")
    _log(f"Prompt:   {PROMPT}")
    _log(f"Expected: {EXPECTED}")
    _log(f"Max tokens: {MAX_TOKENS}")

    engine = LLMEngine(
        model_dir=Path(model_dir),
        inference_backend="qwen_tp",
        max_num_seqs=4,
    )

    # ---- Warm-up: absorb one-time buffer allocations / kernel JIT ----
    _log("[WARMUP] Running warm-up generation (8 tokens)...")
    _ = engine.generate(PROMPT, max_new_tokens=WARMUP_TOKENS, temperature=0.0)
    torch.cuda.synchronize()
    _log("[WARMUP] Done.")

    # ---- Measured run ----
    _log(f"[MEASURE] Generating {MAX_TOKENS} tokens...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.generate(PROMPT, max_new_tokens=MAX_TOKENS, temperature=0.0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = MAX_TOKENS / elapsed if elapsed > 0 else 0.0
    _log(f"  Elapsed:    {elapsed:.3f}s")
    _log(f"  Throughput: {tps:.1f} tok/s")

    # ---- Gate: correctness regression ----
    if _rank == 0 and out != EXPECTED:
        return _fail(
            "THROUGHPUT-002",
            f"Output mismatch.\n  Got:      {out!r}\n  Expected: {EXPECTED!r}",
        )
    _log("[THROUGHPUT-002] PASS: output matches Phase 10 greedy baseline exactly")

    # ---- Gate: throughput floor ----
    if tps < MIN_THROUGHPUT:
        return _fail(
            "THROUGHPUT-001",
            f"Throughput {tps:.1f} tok/s below minimum {MIN_THROUGHPUT} tok/s",
        )
    _log(f"[THROUGHPUT-001] PASS: throughput {tps:.1f} tok/s >= {MIN_THROUGHPUT}")

    # ---- TP=4 note ----
    if _is_tp:
        _log(f"[THROUGHPUT-003] TP={_world_size} throughput: {tps:.1f} tok/s")

    _log("\nPHASE11_THROUGHPUT: ALL TESTS PASSED")
    _log(f"Source: {TRACE_SRC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
