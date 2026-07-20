"""port-model — port a model to a target inference framework.

Given a target inference framework (e.g., vLLM, SGLang), a target hardware
platform, and a target model, add support for that model in the framework.

The plugin auto-registers when this package is imported (standard
MetaInfer task-plugin discovery via pkgutil.iter_modules).
"""

from .orchestrator import plugin as _task_plugin  # noqa: F401
from .server import plugin as _web_plugin  # noqa: F401
