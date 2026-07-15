"""Task-type constants for HTTP inference-framework generators."""

from __future__ import annotations


PYTHON_FRAMEWORK_TASK_TYPE = "gen-infer-framework"
CPP_FRAMEWORK_TASK_TYPE = "gen-infer-framework-cpp"

HTTP_FRAMEWORK_TASK_TYPES = frozenset({
    PYTHON_FRAMEWORK_TASK_TYPE,
    CPP_FRAMEWORK_TASK_TYPE,
})


def is_http_framework_task(task_type: str) -> bool:
    return task_type in HTTP_FRAMEWORK_TASK_TYPES


def is_cpp_framework_task(task_type: str) -> bool:
    return task_type == CPP_FRAMEWORK_TASK_TYPE


__all__ = [
    "PYTHON_FRAMEWORK_TASK_TYPE",
    "CPP_FRAMEWORK_TASK_TYPE",
    "HTTP_FRAMEWORK_TASK_TYPES",
    "is_http_framework_task",
    "is_cpp_framework_task",
]
