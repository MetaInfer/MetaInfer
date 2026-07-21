"""evolve-kernel orchestrator: 6-phase ABCDEF iteration loop for GPU kernel optimization.

Pipeline shape::

    A_plan → B_implement → C_test → D_review ──┬─ C ok  → E_perf_test → F_perf_plan → A_plan (new iter)
                                                └─ C fail → B_implement (new iter)

Correctness is checked via an agent-authored test.sh (no immutable oracle).
"""

from metainfer.orchestrator.tasks import register
from .plugin import PLUGIN

register(PLUGIN)

__version__ = "0.1.0"
