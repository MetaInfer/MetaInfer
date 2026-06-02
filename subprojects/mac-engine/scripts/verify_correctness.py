#!/usr/bin/env python3
"""验证目标引擎输出与 golden outputs 是否一致。

通过 OpenAI-compatible API 调用 vllm-metal (参考) 和自研引擎，
对比 greedy decode 输出。

用法:
    # 与 golden outputs 文件对比
    python verify_correctness.py --target-url http://localhost:8081/v1

    # 两个 live server 对比
    python verify_correctness.py --target-url http://localhost:8081/v1 --ref-url http://localhost:8080/v1
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = PROJECT_ROOT / "tests/golden_outputs/golden_outputs.json"


def load_golden(golden_path: Path | None = None) -> dict:
    path = golden_path or GOLDEN_FILE
    if not path.exists():
        print(f"[ERROR] Golden file not found: {path}", file=sys.stderr)
        print("  Run ref_bench_vllm_metal.py first to collect golden outputs.", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_texts(ref_text: str, target_text: str, test_id: str) -> bool:
    """Compare two texts character by character, report first diff."""
    if ref_text == target_text:
        return True

    for i, (a, b) in enumerate(zip(ref_text, target_text)):
        if a != b:
            ctx = max(0, i - 30)
            print(f"[FAIL] {test_id} @ pos {i}:")
            print(f"  ref:    ...{repr(ref_text[ctx:i+30])}...")
            print(f"  target: ...{repr(target_text[ctx:i+30])}...")
            return False

    # Length mismatch but prefix matches
    min_len = min(len(ref_text), len(target_text))
    print(f"[FAIL] {test_id}: length mismatch (ref={len(ref_text)}, target={len(target_text)})")
    if min_len > 0:
        prefix_match = ref_text[:min_len] == target_text[:min_len]
        print(f"  first {min_len} chars: {'MATCH' if prefix_match else 'DIVERGE'}")
    return False


async def verify_against_golden(
    target_url: str,
    golden: dict,
    model: str = "qwen",
) -> tuple[int, int]:
    """Verify target engine output against golden outputs file."""
    passed, failed = 0, 0
    test_cases = golden["test_cases"]

    async with httpx.AsyncClient(timeout=120) as client:
        for tc in test_cases:
            # Skip error test cases
            if tc.get("error"):
                print(f"[SKIP] {tc['test_id']}: golden has error")
                continue

            try:
                resp = await client.post(
                    f"{target_url}/completions",
                    json={
                        "model": model,
                        "prompt": tc["prompt"],
                        "max_tokens": tc["max_tokens"],
                        "temperature": 0.0,
                    },
                )
                data = resp.json()

                if "choices" not in data:
                    print(f"[FAIL] {tc['test_id']}: server error: {data}")
                    failed += 1
                    continue

                target_text = data["choices"][0]["text"]
                ref_text = tc["output_text"]

                if compare_texts(ref_text, target_text, tc["test_id"]):
                    passed += 1
                    print(f"[PASS] {tc['test_id']}")
                else:
                    failed += 1
            except Exception as e:
                print(f"[FAIL] {tc['test_id']}: exception: {e}")
                failed += 1

    return passed, failed


async def verify_live(
    ref_url: str,
    target_url: str,
    golden: dict,
    model: str = "qwen",
) -> tuple[int, int]:
    """Compare target engine against live reference engine."""
    passed, failed = 0, 0
    test_cases = golden["test_cases"]

    async with httpx.AsyncClient(timeout=120) as client:
        for tc in test_cases:
            if tc.get("error"):
                print(f"[SKIP] {tc['test_id']}: golden has error")
                continue

            try:
                # Get reference output
                ref_resp = await client.post(
                    f"{ref_url}/completions",
                    json={
                        "model": model,
                        "prompt": tc["prompt"],
                        "max_tokens": tc["max_tokens"],
                        "temperature": 0.0,
                    },
                )
                ref_data = ref_resp.json()
                ref_text = ref_data["choices"][0]["text"]

                # Get target output
                target_resp = await client.post(
                    f"{target_url}/completions",
                    json={
                        "model": model,
                        "prompt": tc["prompt"],
                        "max_tokens": tc["max_tokens"],
                        "temperature": 0.0,
                    },
                )
                target_data = target_resp.json()
                target_text = target_data["choices"][0]["text"]

                if compare_texts(ref_text, target_text, tc["test_id"]):
                    passed += 1
                    print(f"[PASS] {tc['test_id']}")
                else:
                    failed += 1
            except Exception as e:
                print(f"[FAIL] {tc['test_id']}: exception: {e}")
                failed += 1

    return passed, failed


async def main_async(args: argparse.Namespace) -> None:
    golden = load_golden(args.golden)
    total = len([tc for tc in golden["test_cases"] if not tc.get("error")])

    if args.ref_url:
        print(f"Verifying against live reference: {args.ref_url}")
        passed, failed = await verify_live(args.ref_url, args.target_url, golden, args.model)
    else:
        print(f"Verifying against golden file: {args.golden or GOLDEN_FILE}")
        if "meta" in golden:
            print(f"  Golden source: {golden['meta']['framework']} v{golden['meta']['framework_version']}")
        passed, failed = await verify_against_golden(args.target_url, golden, args.model)

    print(f"\nResult: {passed}/{total} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description="Verify engine correctness")
    parser.add_argument("--target-url", default="http://localhost:8081/v1", help="Target engine API URL")
    parser.add_argument("--ref-url", default="", help="Reference engine API URL (optional, otherwise use golden file)")
    parser.add_argument("--golden", default=None, help="Golden outputs file path")
    parser.add_argument("--model", default="qwen", help="Model name for API")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
