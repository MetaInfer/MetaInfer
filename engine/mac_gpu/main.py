"""Mac GPU (MPS) LLM 推理引擎 CLI 入口。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Mac GPU (MPS) LLM Inference Engine")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B", help="HuggingFace model name or path"
    )
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--top-p", type=float, default=None, help="Top-p (nucleus) sampling threshold"
    )
    args = parser.parse_args()

    from engine.mac_gpu.engine import MacGPUEngine

    engine = MacGPUEngine(args.model)
    output = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"\nPrompt: {args.prompt}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
