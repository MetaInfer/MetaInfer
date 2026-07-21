"""Independent, arena-style GEMM kernel optimization task."""

from .orchestrator import plugin as _task_plugin  # noqa: F401
from .server import plugin as _web_plugin  # noqa: F401

