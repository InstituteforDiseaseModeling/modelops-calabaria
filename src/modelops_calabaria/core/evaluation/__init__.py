"""Evaluation strategies for calibration targets."""

from .aggregate import AggregateStrategy, IdentityAggregator, MeanAcrossReplicates
from .base import EvaluationResult, EvaluationStrategy, TargetEvaluation
from .composable import Evaluator
from .factories import (
    beta_binomial_nll,
    mean_of_per_replicate_mse,
    replicate_mean_mse,
)
from .loss.base import LossFunction, SquaredErrorLoss
from .loss.nll import BetaBinomialNLL, BinomialNLL
from .reduce import LossReducer, MeanGroupedByReplicate, MeanReducer

__all__ = [
    # Base types
    "EvaluationStrategy",
    "TargetEvaluation",
    "EvaluationResult",
    # Composable evaluator
    "Evaluator",
    # Aggregation strategies
    "AggregateStrategy",
    "IdentityAggregator",
    "MeanAcrossReplicates",
    # Loss functions
    "LossFunction",
    "SquaredErrorLoss",
    "BetaBinomialNLL",
    "BinomialNLL",
    # Reducers
    "LossReducer",
    "MeanReducer",
    "MeanGroupedByReplicate",
    # Factory functions
    "mean_of_per_replicate_mse",
    "replicate_mean_mse",
    "beta_binomial_nll",
]