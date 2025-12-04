"""Evaluation strategies for calibration targets."""

from .aggregate import AggregateStrategy, IdentityAggregator, MeanAcrossReplicates
from .composable import Evaluator, LossEvaluator, TargetLossResult
from .factories import (
    annealed_mse,
    mean_of_per_replicate_mse,
    mean_signal_mse,
    replicate_mean_mse,
)
from .likelihood import LikelihoodEvaluator, TargetLikelihoodResult
from .loglik_fns import (
    beta_binomial_loglik_factory,
    beta_binomial_loglik_per_rep,
    binomial_loglik_per_rep,
)
from .loss.base import LossFunction, SquaredErrorLoss
from .loss.nll import BetaBinomialNLL, BinomialNLL
from .reduce import LossReducer, MeanGroupedByReplicate, MeanReducer

__all__ = [
    # Loss track
    "LossEvaluator",
    "TargetLossResult",
    "Evaluator",  # Backward compatibility alias
    # Likelihood track
    "LikelihoodEvaluator",
    "TargetLikelihoodResult",
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
    # Factory functions - loss track
    "annealed_mse",
    "mean_signal_mse",
    "mean_of_per_replicate_mse",  # Deprecated
    "replicate_mean_mse",  # Deprecated
    # Factory functions - likelihood track
    "beta_binomial_loglik_factory",
    "beta_binomial_loglik_per_rep",
    "binomial_loglik_per_rep",
]
