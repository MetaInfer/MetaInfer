"""Knowledge-evolution task plugin: 4-phase knowledge base evolution loop.

This subpackage evolves the notebooks/ knowledge base for a target model.
The goal is correctness only (not performance): after evolution, the system
must be able to generate a correct inference framework WITHOUT referencing
open-source framework source code.

Pipeline shape::

    A_attempt_pure -> B_enrich -> C_consolidate -> D_verify_final
         |               |                          |
         +- pass -> DONE +- fail -> retry/abort     +- pass -> DONE
                                                     +- fail -> B_enrich

Phases:
  A_attempt_pure   - generate inference framework from notebooks/ only (no open source).
  B_enrich         - explore open-source code, supplement knowledge, re-generate.
  C_consolidate    - write validated knowledge back into notebooks/.
  D_verify_final   - re-generate WITHOUT open source using updated notebooks/.

Oracles:
  oracles/
  +-- consistency.py   KnowledgeConsistencyOracle (LLM-judge new vs old)
  +-- coverage.py      KnowledgeCoverageOracle (blind-spot coverage check)
"""

from .orchestrator import plugin as _task_plugin  # noqa: F401 — registers TaskPlugin
from .server import plugin as _web_plugin          # noqa: F401 — registers WebPlugin
