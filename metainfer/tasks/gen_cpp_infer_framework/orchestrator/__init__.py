"""gen-cpp-infer-framework task plugin: the 6-phase ABCDEF iteration loop.

This subpackage is **self-contained**: pipeline, prompts, phases, and
oracles (both correctness + perf) all live here. The framework
(:mod:`metainfer.orchestrator`) provides shared infrastructure (state
store, sub-agent manager, oracle ABCs, bootstrap helpers) but does not
import this package's pipeline directly — the launcher invokes the CLI
module declared in :data:`plugin.PLUGIN` via the registry.

Pipeline shape::

    A_plan → B_implement → C_test → D_review ──┬─ C ok  → E_perf_test → F_perf_plan → A_plan (new iter)
                                                └─ C fail → B_implement (new iter)

Failures never enter a terminal Fail state — every failure either
retries in place, consumes the iteration, or routes back to A_plan. The
only terminal phase is ``finished``.

Oracle layout::

    oracles/
    ├── correctness.py   InferFrameworkOracle (boots serve.sh, LLM-judges responses)
    ├── perf.py          PerfOracle (boots serve.sh, sweeps concurrency ladder)
    └── data/
        ├── correctness_cases.yaml
        └── perf_prompts.yaml

The pipeline imports these directly. No global oracle registry.
"""

from metainfer.orchestrator.tasks import register
from .plugin import PLUGIN

register(PLUGIN)

__version__ = "0.2.0"
