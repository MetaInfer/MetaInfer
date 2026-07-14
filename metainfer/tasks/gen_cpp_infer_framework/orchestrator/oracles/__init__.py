"""Gen-cpp-infer-framework oracles.

This package contains the two oracle harnesses used by the gen-cpp-infer-framework
pipeline:

* :mod:`.correctness` — boots the agent's ``serve.sh``, probes it with
  canned prompts, and LLM-judges the responses for correctness. Used by
  the pipeline's C_test phase.
* :mod:`.perf` — boots the same ``serve.sh``, sweeps a fixed concurrency
  ladder against a fixed prompt set, and emits structured perf metrics
  (tokens/sec, p50/p99 latency, ...). Used by the pipeline's E_perf_test
  phase.

Both oracles share the ``serve.sh`` artifact contract (the agent writes
one script; both oracles launch it). Test data lives in ``data/``.

The pipeline imports these directly — there is no global oracle registry.
If a different task type wants to reuse these, it should subclass or
import from here explicitly (rare; we'd promote to framework level if it
becomes a common pattern).
"""

from .correctness import InferFrameworkOracle as CorrectnessOracle
from .perf import PerfOracle

__all__ = ["CorrectnessOracle", "PerfOracle"]
