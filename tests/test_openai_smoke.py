"""OpenAI Chat Completions smoke tests — declarative YAML-driven.

Usage:
    1. Start your OpenAI-compatible server
    2. Edit tests/openai_smoke_config.yaml (base_url, model, timeout)
    3. pytest tests/test_openai_smoke.py -v -s                      # all cases
    4. pytest tests/test_openai_smoke.py -v -s --case "中文问候"      # single replay
    5. pytest tests/test_openai_smoke.py -v -s --case "多轮"          # fuzzy match
    6. pytest tests/test_openai_smoke.py -v -s --list-cases          # list all case names
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).parent


def _load_config() -> dict:
    with open(_HERE / "openai_smoke_config.yaml") as f:
        return yaml.safe_load(f)


def _load_cases() -> list[dict]:
    with open(_HERE / "openai_smoke_cases.yaml") as f:
        return yaml.safe_load(f)


def _post_chat(config: dict, body: dict) -> dict:
    url = f"{config['base_url'].rstrip('/')}/v1/chat/completions"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = config.get("timeout", 60)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        pytest.fail(f"HTTP {e.code}: {detail}")


CONFIG = _load_config()
CASES = _load_cases()
CASE_IDS = [c["name"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_smoke(case: dict, request: pytest.FixtureRequest) -> None:
    body: dict = {
        "model": CONFIG["model"],
        "messages": case["messages"],
        "stream": False,
    }
    if "max_tokens" in case:
        body["max_tokens"] = case["max_tokens"]

    data = _post_chat(CONFIG, body)

    # --- structure checks ---
    assert "id" in data, "response missing 'id'"
    assert "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0, (
        "response missing non-empty 'choices'"
    )

    choice = data["choices"][0]
    message = choice.get("message", {})
    content = message.get("content", "")
    assert isinstance(content, str) and len(content.strip()) > 0, (
        "choices[0].message.content is empty"
    )

    # --- finish_reason ---
    asserts = case.get("assert", {})
    expected_reason = asserts.get("finish_reason")
    if expected_reason:
        actual_reason = choice.get("finish_reason")
        assert actual_reason == expected_reason, (
            f"finish_reason: expected '{expected_reason}', got '{actual_reason}'"
        )

    # --- keyword checks ---
    keywords = asserts.get("keywords_any")
    if keywords:
        min_matches = asserts.get("min_keyword_matches", 1)
        matched = [kw for kw in keywords if kw in content]
        assert len(matched) >= min_matches, (
            f"expected at least {min_matches} keyword(s) from {keywords} in output, "
            f"matched {matched}. Output: {content[:200]}"
        )

    # --- usage ---
    usage = data.get("usage", {})
    assert "completion_tokens" in usage, "response missing usage.completion_tokens"

    # --- print generated text for inspection (visible with -s) ---
    usage_info = f"tokens={usage.get('completion_tokens', '?')}"
    finish_info = choice.get("finish_reason", "?")
    print(f"\n{'=' * 60}")
    print(f"[{case['name']}] {usage_info}, finish={finish_info}")
    print(f"{'-' * 60}")
    print(content)
    print(f"{'=' * 60}")
