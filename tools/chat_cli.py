"""
CLI Chat 工具：多轮对话 + Agent Tool Calling。
支持 MacGPU 本地引擎和 OpenAI 兼容 API 两种后端。

用法：
    # MacGPU 本地（无 OPENAI_API_KEY 时自动选择）
    python -m tools.chat_cli --model Qwen/Qwen2.5-0.5B

    # OpenAI API（在 .env 或环境变量中设置 OPENAI_API_KEY）
    OPENAI_API_KEY=sk-xxx python -m tools.chat_cli --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 确保项目根目录在 sys.path 上
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_dotenv() -> None:
    """从项目根目录的 .env 文件加载环境变量（不覆盖已有值）。"""
    dotenv = Path(ROOT) / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

from tools.backends import ChatBackend, GenerateResult  # noqa: E402
from tools.tool_registry import ToolRegistry, create_default_registry  # noqa: E402

# --- 日志 ---


@dataclass
class TurnLog:
    timestamp: str
    role: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)


class ChatLogger:
    """双输出日志：JSONL + 纯文本。"""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.log_dir / f"chat_{ts}.jsonl"
        self.text_path = self.log_dir / f"chat_{ts}.txt"

    def log_turn(self, turn: TurnLog) -> None:
        entry = {
            "timestamp": turn.timestamp,
            "role": turn.role,
            "content": turn.content,
            "input_tokens": turn.input_tokens,
            "output_tokens": turn.output_tokens,
            "latency_ms": round(turn.latency_ms, 1),
            "tool_calls": turn.tool_calls,
            "tool_results": turn.tool_results,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        with open(self.text_path, "a", encoding="utf-8") as f:
            if turn.role == "user":
                f.write(f"\n{'=' * 60}\n")
                f.write(f"[User] {turn.content}\n")
            elif turn.role == "assistant":
                f.write(f"[Assistant] {turn.content}\n")
                for tc in turn.tool_calls:
                    f.write(f"  [Tool Call] {tc['name']}({tc['arguments']})\n")
                for tr in turn.tool_results:
                    f.write(f"  [Tool Result] {tr['name']} -> {tr['result']}\n")
                f.write(
                    f"  [Stats] in={turn.input_tokens} out={turn.output_tokens} "
                    f"latency={turn.latency_ms:.0f}ms\n"
                )


# --- Chat Session ---


class ChatSession:
    """
    多轮对话管理器，通过 ChatBackend 抽象支持多种后端。

    每轮流程：
    1. 调用 backend.generate() 生成回复
    2. 检测 tool call → 执行 → 追加结果 → 重新生成
    3. 无 tool call 时返回最终文本
    """

    def __init__(
        self,
        backend: ChatBackend,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.6,
        top_p: float | None = None,
        max_tokens: int = 512,
        tool_registry: ToolRegistry | None = None,
        logger: ChatLogger | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tool_registry = tool_registry
        self.logger = logger
        self.max_tool_rounds = max_tool_rounds

        self.messages: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _tool_schemas(self) -> list[dict] | None:
        if self.tool_registry:
            schemas = self.tool_registry.get_schemas()
            if schemas:
                return schemas
        return None

    def _append_tool_result_messages(self, result: GenerateResult) -> None:
        """根据后端类型，将 tool call 和 tool result 追加到消息历史。"""
        if not result.tool_calls:
            return

        # 判断后端类型：OpenAI 后端用 assistant + tool role，
        # MacGPU 后端用 assistant(content=原始文本) + tool role
        from tools.backends import OpenAIBackend

        if isinstance(self.backend, OpenAIBackend):
            # OpenAI 格式：assistant message 带 tool_calls
            assistant_msg: dict = {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            self.messages.append(assistant_msg)

            for tc in result.tool_calls:
                tool_result = self.tool_registry.execute(tc.name, tc.arguments)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": tool_result,
                    }
                )
        else:
            # MacGPU / 通用格式：assistant content + tool messages
            self.messages.append({"role": "assistant", "content": result.content})
            for tc in result.tool_calls:
                tool_result = self.tool_registry.execute(tc.name, tc.arguments)
                self.messages.append({"role": "tool", "name": tc.name, "content": tool_result})

    def chat(self, user_input: str) -> str:
        """处理一轮用户输入，返回最终助手回复。"""
        self.messages.append({"role": "user", "content": user_input})

        if self.logger:
            self.logger.log_turn(
                TurnLog(
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    role="user",
                    content=user_input,
                )
            )

        tool_round = 0
        while tool_round < self.max_tool_rounds:
            t0 = time.perf_counter()
            result = self.backend.generate(
                self.messages,
                tools=self._tool_schemas(),
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            self.total_input_tokens += result.input_tokens
            self.total_output_tokens += result.output_tokens

            if not result.tool_calls:
                self.messages.append({"role": "assistant", "content": result.content})

                if self.logger:
                    self.logger.log_turn(
                        TurnLog(
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            role="assistant",
                            content=result.content,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            latency_ms=latency_ms,
                        )
                    )
                return result.content

            # 有 tool call → 执行并追加结果
            tool_call_logs: list[dict] = []
            tool_result_logs: list[dict] = []

            for tc in result.tool_calls:
                tool_call_logs.append({"name": tc.name, "arguments": tc.arguments})
                tool_result = self.tool_registry.execute(tc.name, tc.arguments)
                tool_result_logs.append({"name": tc.name, "result": tool_result})

            self._append_tool_result_messages(result)

            if self.logger:
                self.logger.log_turn(
                    TurnLog(
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        role="assistant",
                        content=result.content,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=latency_ms,
                        tool_calls=tool_call_logs,
                        tool_results=tool_result_logs,
                    )
                )

            print(f"  [Tool calls: {', '.join(tc.name for tc in result.tool_calls)}]")
            for tr in tool_result_logs:
                print(f"    {tr['name']} -> {tr['result'][:200]}")

            tool_round += 1

        return "(reached max tool rounds)"

    def print_stats(self) -> None:
        print("--- Session Stats ---")
        print(f"Total input tokens:  {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Messages in history: {len(self.messages)}")


# --- CLI 入口 ---


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI Chat with Agent Tool Calling")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool calling")
    args = parser.parse_args()

    # --- 自动选择后端 ---
    # 优先用环境变量：OPENAI_API_KEY 存在时走 API，否则走本地 MacGPU
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL", "")

    if api_key:
        from tools.backends import OpenAIBackend

        model = model or "gpt-4o-mini"
        backend = OpenAIBackend(api_key=api_key, base_url=api_base, model=model)
        print(f"[chat_cli] Backend: API ({api_base}), model: {model}")
    else:
        from mac_gpu.engine import MacGPUEngine
        from tools.backends import MacGPUBackend

        model = model or "Qwen/Qwen2.5-0.5B"
        print(f"[chat_cli] Backend: MacGPU (MPS), loading {model} ...")
        engine = MacGPUEngine(model)
        backend = MacGPUBackend(engine)
        print("[chat_cli] Model loaded.\n")

    # --- 工具和日志 ---
    tool_registry = None if args.no_tools else create_default_registry()
    if tool_registry:
        print(f"[chat_cli] Tools: {', '.join(tool_registry.tool_names)}")
    else:
        print("[chat_cli] Tool calling disabled.")

    logger = ChatLogger(args.log_dir)
    print(f"[chat_cli] Logging to: {logger.log_dir}\n")

    session = ChatSession(
        backend,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tool_registry=tool_registry,
        logger=logger,
    )

    print("Chat started. Type 'quit' or Ctrl+C to exit.\n")
    try:
        while True:
            try:
                user_input = input("You> ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            response = session.chat(user_input)
            print(f"\nAssistant> {response}\n")

    except KeyboardInterrupt:
        print("\n")

    session.print_stats()
    print(f"Logs saved to: {logger.log_dir}")


if __name__ == "__main__":
    main()
