"""gen-cpp-infer-framework task package.

Importing this package registers its orchestrator TaskPlugin and its
WebUI plugin.
"""

from .orchestrator import plugin as _task_plugin  # noqa: F401
from .server import plugin as _web_plugin  # noqa: F401
