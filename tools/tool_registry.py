"""
Agent 工具注册表：OpenAI function calling 风格的工具定义与执行。
所有工具在此单文件中定义和注册。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Callable

ToolFunc = Callable[..., str]


class ToolDefinition:
    """单个工具定义，包含 schema 和实现。"""

    def __init__(self, name: str, description: str, parameters: dict, func: ToolFunc) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: object) -> str:
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Error executing tool '{self.name}': {e}"


class ToolRegistry:
    """工具注册表，管理所有可用工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"
        return tool.execute(**arguments)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# --- 内置工具实现 ---


def _calculator(expression: str) -> str:
    """安全地计算数学表达式。"""
    allowed_names: dict[str, object] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
    return str(result)


def _get_current_time() -> str:
    """返回当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_default_registry() -> ToolRegistry:
    """创建包含内置工具的注册表实例。"""
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="calculator",
            description=(
                "Evaluate a mathematical expression and return the result. "
                "Supports basic arithmetic (+, -, *, /), powers, and math functions "
                "(sqrt, log, sin, cos, tan, pi, e)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The mathematical expression to evaluate, "
                            "e.g. '2 + 2' or 'sqrt(16) * 3'"
                        ),
                    }
                },
                "required": ["expression"],
            },
            func=_calculator,
        )
    )

    registry.register(
        ToolDefinition(
            name="get_current_time",
            description="Get the current date and time.",
            parameters={
                "type": "object",
                "properties": {},
            },
            func=_get_current_time,
        )
    )

    return registry
