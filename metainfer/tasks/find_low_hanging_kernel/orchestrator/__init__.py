"""find-low-hanging-kernel orchestrator package.

Pipeline shape::

    P1_code_analysis → P2_tracing_analysis → P3_graph_build
                                                     │
                                                     ▼
                                             P3_graph_validate ◄─┐
                                                     │            │
                                         fix_applied │            │
                                                     └────────────┘
                                                     │ clean
                                                     ▼
                                                P4_visualize ──► done

Steps 1 and 2 each fan out to 3 independent analysis agents (one per
"angle") and then a synthesizer cross-validates. Step 3 builds the flow
graph with a single agent, then a deterministic driver iterates:
integrity-check (pure Python) + 5-worker AgentPool semantic-check that
splits nodes into 3-groups. Step 4 is pure-render.
"""

from metainfer.orchestrator.tasks import register
from .plugin import PLUGIN

register(PLUGIN)

__version__ = "0.1.0"
