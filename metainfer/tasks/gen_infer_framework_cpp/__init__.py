"""gen-infer-framework-cpp task package.

Self-contained: orchestrator pipeline + web server handler + frontend
assets + C++ knowledge base + tests + form schema. Importing this package registers both
its ``TaskPlugin`` (orchestrator-side dispatch) and its ``WebPlugin``
(web-side routes / detail view / QA).
"""

from .orchestrator import plugin as _task_plugin  # noqa: F401 — registers TaskPlugin
from .server import plugin as _web_plugin  # noqa: F401 — registers WebPlugin
