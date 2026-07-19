"""find-low-hanging-kernel task package.

Given a Chrome tracing profile + a model directory + an inference framework
source tree, this task builds a human+machine-auditable execution-flow graph
of one inference pass and identifies which kernels have the most optimization
headroom.

Importing this package registers both its TaskPlugin (orchestrator-side
dispatch) and its WebPlugin (web routes / detail view / QA).
"""

from .orchestrator import plugin as _task_plugin  # noqa: F401 — registers TaskPlugin
from .server import plugin as _web_plugin  # noqa: F401 — registers WebPlugin
