"""System-owned evaluator for GEMM submissions."""

from .champion import ChampionStore
from .runner import EvaluationError, EvaluationResult, EvaluatorRunner
from .scoring import ScoreResult, compare_measurements, score_benchmark
from .spec import BenchmarkCaseSpec, FrozenEvaluatorBundle, KernelTaskSpec, SpecError
from .weights import FrozenWeightBundle

__all__ = [
    "ChampionStore",
    "BenchmarkCaseSpec",
    "EvaluationError",
    "EvaluationResult",
    "EvaluatorRunner",
    "FrozenEvaluatorBundle",
    "FrozenWeightBundle",
    "KernelTaskSpec",
    "ScoreResult",
    "SpecError",
    "compare_measurements",
    "score_benchmark",
]
