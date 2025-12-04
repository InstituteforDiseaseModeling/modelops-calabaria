"""Factory functions for common loss evaluation strategies."""

from .aggregate import IdentityAggregator, MeanAcrossReplicates
from .composable import LossEvaluator
from .loss.base import SquaredErrorLoss
from .reduce import MeanGroupedByReplicate, MeanReducer


def annealed_mse(col: str, weight: float = 1.0) -> LossEvaluator:
    """
    Annealed (replicate-mean) MSE.

    J(θ) ≈ E_ε[(y - sim)²] = mean of per-replicate MSE.

    This penalizes simulator variance (includes variance penalty term).
    With R replicates:
        J = (1/R) Σ_r (1/N) Σ_i (y_i - sim_i^(r))²

    This is a biased estimator that includes a variance penalty. The loss includes
    both the squared bias and the simulator variance due to Jensen's inequality:
        E_ε[(y - sim)²] ≥ (y - E_ε[sim])²

    For calibration targeting observation-level mean without variance penalty,
    use mean_signal_mse instead.

    Parameters
    ----------
    col : str
        Column name to evaluate
    weight : float, optional
        Weight for this target in multi-target calibration (default: 1.0)

    Returns
    -------
    LossEvaluator
        Configured evaluator for annealed MSE
    """
    return LossEvaluator(
        aggregator=IdentityAggregator(),
        loss_fn=SquaredErrorLoss(col=col),
        reducer=MeanGroupedByReplicate(),
        weight=weight,
        name=f"annealed_mse[{col}]",
    )


def mean_signal_mse(col: str, weight: float = 1.0) -> LossEvaluator:
    """
    Mean-signal MSE.

    J(θ) ≈ (y - E_ε[sim])², approximated by first averaging sim across replicates.

    This targets the observation-level mean without penalizing simulator variance.
    With R replicates:
        J = (1/N) Σ_i (y_i - (1/R) Σ_r sim_i^(r))²

    This approximates the squared error between observations and the expected value
    of the simulator output (integrated over simulator noise ε). With large R, this
    estimates the loss without variance penalty.

    For calibration including variance penalty, use annealed_mse instead.

    Parameters
    ----------
    col : str
        Column name to evaluate
    weight : float, optional
        Weight for this target in multi-target calibration (default: 1.0)

    Returns
    -------
    LossEvaluator
        Configured evaluator for mean-signal MSE
    """
    return LossEvaluator(
        aggregator=MeanAcrossReplicates([col]),
        loss_fn=SquaredErrorLoss(col=col),
        reducer=MeanReducer(),
        weight=weight,
        name=f"mean_signal_mse[{col}]",
    )


# Backward compatibility aliases (deprecated)
def mean_of_per_replicate_mse(col: str) -> LossEvaluator:
    """Deprecated: Use annealed_mse instead."""
    return annealed_mse(col)


def replicate_mean_mse(col: str) -> LossEvaluator:
    """Deprecated: Use mean_signal_mse instead."""
    return mean_signal_mse(col)
