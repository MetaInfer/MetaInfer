"""
Chat 后端抽象层：支持 MacGPU 本地引擎和 OpenAI 兼容 API。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """一次工具调用。"""

    name: str
    arguments: dict
    call_id: str = ""


@dataclass
class GenerateResult:
    """后端 generate() 的统一返回值。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class ChatBackend(ABC):
    """对话后端抽象基类。"""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> GenerateResult: ...


# --- MacGPU 后端 ---


TOOL_CALL_START = "<tool_call"
TOOL_CALL_END = "</tool_call"


def _parse_tool_calls(text: str) -> list[ToolCall]:
    """解析 Qwen2.5 格式的 tool call（<tool_call\\nJSON\\n</tool_call）。"""
    if TOOL_CALL_START not in text:
        return []

    pattern = rf"{re.escape(TOOL_CALL_START)}\n(.*?){re.escape(TOOL_CALL_END)}"
    matches = re.findall(pattern, text, re.DOTALL)

    calls: list[ToolCall] = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCall(name=name, arguments=arguments))
        except json.JSONDecodeError:
            continue
    return calls


class MacGPUBackend(ChatBackend):
    """Apple MPS 本地推理后端。"""

    def __init__(self, engine: object) -> None:
        self.engine = engine
        self.tokenizer = engine.runner.tokenizer

    def _render_prompt(self, messages: list[dict], tools: list[dict] | None) -> str:
        kwargs: dict = {
            "messages": messages,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(**kwargs)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> GenerateResult:
        prompt_text = self._render_prompt(messages, tools)
        input_tokens = self._count_tokens(prompt_text)

        output_text = self.engine.generate(
            prompt_text,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        output_tokens = self._count_tokens(output_text)

        tool_calls = _parse_tool_calls(output_text)

        # 去掉 tool call 标签，只保留普通文本
        if tool_calls:
            idx = output_text.find(TOOL_CALL_START)
            content = output_text[:idx].strip() if idx != -1 else ""
        else:
            content = output_text

        return GenerateResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# --- OpenAI API 后端 ---


class OpenAIBackend(ChatBackend):
    """OpenAI 兼容 API 后端（/v1/chat/completions）。"""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> GenerateResult:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            body["top_p"] = top_p
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API error {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error: {e.reason}") from e
        _latency = time.perf_counter() - t0

        choice = data["choices"][0]
        msg = choice["message"]

        content = msg.get("content") or ""
        tool_calls: list[ToolCall] = []

        for tc in msg.get("tool_calls", []):
            func = tc["function"]
            args: dict = {}
            if func.get("arguments"):
                try:
                    args = json.loads(func["arguments"])
                except json.JSONDecodeError:
                    args = {"_raw": func["arguments"]}
            tool_calls.append(
                ToolCall(
                    name=func["name"],
                    arguments=args,
                    call_id=tc.get("id", ""),
                )
            )

        usage = data.get("usage", {})
        return GenerateResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
