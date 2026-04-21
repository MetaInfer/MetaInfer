from __future__ import annotations

import pytest
import torch

from llm_engine import LLMEngine

PROMPTS: list[str] = [
    "The capital of France is",
    "写一首关于春天的诗",
    "用英语介绍下你自己",
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real 16B inference requires CUDA GPU")
def test_real_inference_end_to_end() -> None:
    prompts = PROMPTS
    print("[Test] init real LLMEngine")
    engine = LLMEngine(
        block_size=16,
        mem_utilization=0.80,
        reserve_bytes=3 * 1024**3,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
    )

    print("[Test] start generate for prompts:", prompts)
    outputs = engine.generate(prompts, max_new_tokens=32, temperature=0.0, top_p=None)

    print("[Test] outputs:")
    for i, out in enumerate(outputs):
        print(f"[Test] output[{i}] = {out!r}")

    assert len(outputs) == len(prompts), f"expected {len(prompts)} outputs, got {len(outputs)}"
    assert all(isinstance(x, str) for x in outputs)
    assert all(len(x.strip()) > 0 for x in outputs)


if __name__ == "__main__":
    test_real_inference_end_to_end()
