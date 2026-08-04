"""sglang_trace_analyze — auto-generate torch profiler traces via SGLang,
analyze them (operator-to-structure mapping, kernel hotspots, TFLOPS / MFU,
overlap opportunities, fuse suggestions), and surface results + LLM hints
in the MetaInfer WebUI.
"""

from .orchestrator import plugin as _task_plugin  # noqa: F401
from .server import plugin as _web_plugin          # noqa: F401
