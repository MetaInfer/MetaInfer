"""Tests for tools/backends.py, tools/tool_registry.py, and tools/chat_cli.py (no GPU)."""

from __future__ import annotations

import json
import re

from tools.backends import (
    ChatBackend,
    GenerateResult,
    ToolCall,
    _parse_tool_calls,
)
from tools.chat_cli import ChatLogger, ChatSession, TurnLog
from tools.tool_registry import ToolDefinition, ToolRegistry, create_default_registry

# --- _parse_tool_calls (from backends.py) ---


class TestParseToolCalls:
    def test_no_tool_call(self) -> None:
        calls = _parse_tool_calls("Hello, I am an assistant.")
        assert calls == []

    def test_single_tool_call(self) -> None:
        raw = '<tool_call\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call'
        calls = _parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].name == "calculator"
        assert calls[0].arguments == {"expression": "2+2"}

    def test_parallel_tool_calls(self) -> None:
        raw = (
            '<tool_call\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call\n'
            '<tool_call\n{"name": "get_current_time", "arguments": {}}\n</tool_call'
        )
        calls = _parse_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0].name == "calculator"
        assert calls[1].name == "get_current_time"

    def test_malformed_json_skipped(self) -> None:
        raw = "<tool_call\n{invalid json}\n</tool_call"
        calls = _parse_tool_calls(raw)
        assert calls == []

    def test_empty_arguments(self) -> None:
        raw = '<tool_call\n{"name": "get_current_time", "arguments": {}}\n</tool_call'
        calls = _parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].arguments == {}


# --- ToolRegistry ---


class TestToolRegistry:
    def test_default_registry_has_tools(self) -> None:
        reg = create_default_registry()
        assert "calculator" in reg.tool_names
        assert "get_current_time" in reg.tool_names

    def test_schema_format(self) -> None:
        reg = create_default_registry()
        schemas = reg.get_schemas()
        assert len(schemas) == 2
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]

    def test_calculator_execute(self) -> None:
        reg = create_default_registry()
        result = reg.execute("calculator", {"expression": "2 + 3"})
        assert result == "5"

    def test_calculator_complex(self) -> None:
        reg = create_default_registry()
        result = reg.execute("calculator", {"expression": "sqrt(16) * 2"})
        assert result == "8.0"

    def test_get_current_time(self) -> None:
        reg = create_default_registry()
        result = reg.execute("get_current_time", {})
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)

    def test_unknown_tool(self) -> None:
        reg = create_default_registry()
        result = reg.execute("nonexistent", {})
        assert "Unknown tool" in result

    def test_register_custom_tool(self) -> None:
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="echo",
                description="Echo back the input",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
                func=lambda text: f"Echo: {text}",
            )
        )
        assert reg.get("echo") is not None
        assert reg.execute("echo", {"text": "hello"}) == "Echo: hello"


# --- ChatLogger ---


class TestChatLogger:
    def test_log_creates_files(self, tmp_path: object) -> None:
        logger = ChatLogger(str(tmp_path))  # type: ignore[arg-type]
        logger.log_turn(
            TurnLog(
                timestamp="2026-01-01T00:00:00",
                role="user",
                content="Hello",
            )
        )
        assert logger.jsonl_path.exists()
        assert logger.text_path.exists()

        with open(logger.jsonl_path) as f:
            data = json.loads(f.readline())
            assert data["role"] == "user"
            assert data["content"] == "Hello"

    def test_log_tool_calls(self, tmp_path: object) -> None:
        logger = ChatLogger(str(tmp_path))  # type: ignore[arg-type]
        logger.log_turn(
            TurnLog(
                timestamp="2026-01-01T00:00:00",
                role="assistant",
                content="<tool_call...",
                input_tokens=100,
                output_tokens=20,
                latency_ms=500.0,
                tool_calls=[{"name": "calc", "arguments": {"expr": "1+1"}}],
                tool_results=[{"name": "calc", "result": "2"}],
            )
        )
        with open(logger.text_path) as f:
            content = f.read()
            assert "[Tool Call] calc" in content
            assert "[Tool Result] calc -> 2" in content
            assert "in=100 out=20" in content


# --- Mock Backend for ChatSession tests ---


class MockBackend(ChatBackend):
    """记录调用并返回预设结果的 mock 后端。"""

    def __init__(self, responses: list[GenerateResult]) -> None:
        self.responses = list(responses)
        self.call_index = 0
        self.calls: list[dict] = []

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        max_tokens: int = 512,
    ) -> GenerateResult:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
            }
        )
        if self.call_index < len(self.responses):
            result = self.responses[self.call_index]
            self.call_index += 1
            return result
        return GenerateResult(content="(no more responses)")


# --- ChatSession ---


class TestChatSession:
    def test_simple_response(self) -> None:
        backend = MockBackend([GenerateResult(content="Hello!", input_tokens=10, output_tokens=5)])
        session = ChatSession(backend, system_prompt="You are helpful.")
        response = session.chat("Hi")

        assert response == "Hello!"
        assert len(session.messages) == 3  # system + user + assistant
        assert session.total_input_tokens == 10
        assert session.total_output_tokens == 5

    def test_tool_call_loop(self) -> None:
        backend = MockBackend(
            [
                GenerateResult(
                    content="",
                    tool_calls=[ToolCall(name="calculator", arguments={"expression": "2+2"})],
                    input_tokens=20,
                    output_tokens=10,
                ),
                GenerateResult(
                    content="2 + 2 = 4",
                    input_tokens=30,
                    output_tokens=5,
                ),
            ]
        )
        registry = create_default_registry()
        session = ChatSession(backend, tool_registry=registry)
        response = session.chat("What is 2+2?")

        assert response == "2 + 2 = 4"
        # system=None, user, assistant(tool_call), tool(result), assistant(final)
        assert len(session.messages) == 4
        assert backend.call_index == 2

    def test_max_tool_rounds(self) -> None:
        # Always return a tool call -> should hit max rounds
        always_tool = GenerateResult(
            content="",
            tool_calls=[ToolCall(name="calculator", arguments={"expression": "1"})],
            input_tokens=5,
            output_tokens=5,
        )
        backend = MockBackend([always_tool] * 10)
        registry = create_default_registry()
        session = ChatSession(backend, tool_registry=registry, max_tool_rounds=3)
        response = session.chat("test")

        assert response == "(reached max tool rounds)"
        assert backend.call_index == 3

    def test_no_tools_disables_tool_schemas(self) -> None:
        backend = MockBackend([GenerateResult(content="ok", input_tokens=5, output_tokens=2)])
        session = ChatSession(backend)  # no tool_registry
        session.chat("hi")

        # tools should be None since no registry
        assert backend.calls[0]["tools"] is None
