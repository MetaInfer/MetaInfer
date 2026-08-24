"""System-owned evaluator for GEMM submissions."""

from .champion import ChampionStore
from .runner import EvaluationError, EvaluationResult, EvaluatorRunner
from .scoring import (
    PromotionResult,
    ScoreResult,
    compare_against_champion,
    compare_measurements,
)
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
    "PromotionResult",
    "ScoreResult",
    "SpecError",
    "compare_against_champion",
    "compare_measurements",
]
