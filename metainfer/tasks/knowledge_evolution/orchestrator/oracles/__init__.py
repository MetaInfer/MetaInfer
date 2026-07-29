"""Knowledge-evolution oracles.

* :mod:`.consistency` — KnowledgeConsistencyOracle: checks whether new
  knowledge contradicts existing notebooks/ content. Uses LLM Judge.
* :mod:`.coverage` — KnowledgeCoverageOracle: checks whether new
  knowledge fills the blind spots that caused the pure-KB attempt to fail.
"""

from .consistency import KnowledgeConsistencyOracle
from .coverage import KnowledgeCoverageOracle

__all__ = ["KnowledgeConsistencyOracle", "KnowledgeCoverageOracle"]
