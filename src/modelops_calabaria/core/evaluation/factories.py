"""Factory functions for common evaluation strategies."""

from .aggregate import IdentityAggregator, MeanAcrossReplicates
from .composable import Evaluator
from .loss.base import SquaredErrorLoss
from .loss.nll import BetaBinomialNLL
from .reduce import MeanGroupedByReplicate, MeanReducer


def mean_of_per_replicate_mse(col: str) -> Evaluator:
    """
    Computes MSE per replicate, then returns the mean of those replicate-level losses.

    loss = mean( MSE(rep_0), MSE(rep_1), ... )
    """
    return Evaluator(
        aggregator=IdentityAggregator(),
        loss_fn=SquaredErrorLoss(col=col),
        reducer=MeanGroupedByReplicate(),
    )


def replicate_mean_mse(col: str) -> Evaluator:
    """
    Averages the simulated values across replicates first, then computes MSE.

    loss = MSE( mean(sim_rep_0, sim_rep_1, ...) vs obs )
    """
    return Evaluator(
        aggregator=MeanAcrossReplicates([col]),
        loss_fn=SquaredErrorLoss(col=col),
        reducer=MeanReducer(),
    )


def beta_binomial_nll(x_col: str, n_col: str, prior=(1, 1)) -> Evaluator:
    """
    Beta-binomial negative log-likelihood evaluation.

    Parameters
    ----------
    x_col : str
        Column name for successes
    n_col : str
        Column name for trials
    prior : tuple
        Beta prior parameters (alpha, beta)
    """
    return Evaluator(
        aggregator=IdentityAggregator(),
        loss_fn=BetaBinomialNLL(x_col=x_col, n_col=n_col, prior=prior),
        reducer=MeanGroupedByReplicate(),
    )