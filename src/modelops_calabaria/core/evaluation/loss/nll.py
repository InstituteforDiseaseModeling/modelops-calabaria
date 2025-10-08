"""Negative log-likelihood loss functions."""

from dataclasses import dataclass
from typing import Optional

import polars as pl
import scipy.stats as sps

from ...alignment import AlignedData
from ...constants import LOSS_COL
from .base import LossFunction


@dataclass
class BetaBinomialNLL(LossFunction):
    """
    Beta-binomial negative log-likelihood.

    For the beta-binomial negative log-likelihood, we begin with a Beta(1,1)
    prior and subsequently observe from the simulation some number of successes (s_x)
    out of some total number of trials (s_n).

    The result is a Beta(s_x+1, s_n-s_x+1) posterior.

    We then compare this to the real data, which has o_x successes (positives)
    in o_n trials (total observations). To do so, we use a beta-binomial likelihood:

    p(x|n, a, b) = (n choose x) B(x+a, n-x+b) / B(a, b)

    where
        x = number of successes from the observed data
        n = number of trials from the observed data
        a = number of successes from the simulation plus one from the prior
        b = number of failures from the simulation plus one from the prior
    and B is the beta function, B(x, y) = Gamma(x)Gamma(y)/Gamma(x+y)

    We return the negative log of p(x|n, a, b)

    Parameters
    ----------
    x_col : str
        Column name for the number of observed successes in both the simulated and observed data.
    n_col : str
        Column name for the number of trials in both the simulated and observed data.
    prior : tuple | list, default=(1, 1)
        Prior parameters for the beta distribution. Must be positive numbers.
        The default is (1, 1), which is like Laplace smoothing.
    """

    x_col: str
    n_col: str
    prior: Optional[tuple | list] = (1, 1)

    def __post_init__(self):
        self.prior_alpha, self.prior_beta = self._validate_prior(self.prior)
        return

    def _validate_prior(self, prior):
        if (
            not isinstance(prior, (tuple, list))
            or len(prior) != 2
            or not all(isinstance(x, (int, float)) and x > 0 for x in prior)
        ):
            raise ValueError(
                "Prior must be a tuple of two positive numbers (alpha, beta)."
            )
        return prior

    def compute(self, aligned: AlignedData) -> AlignedData:
        # Extract columns
        s_x = aligned.get_sim_col(self.x_col)
        s_n = aligned.get_sim_col(self.n_col)

        o_x = aligned.get_obs_col(self.x_col).cast(pl.Int32)
        o_n = aligned.get_obs_col(self.n_col).cast(pl.Int32)

        # Apply prior
        alpha = s_x + self.prior_alpha
        beta = s_n - s_x + self.prior_beta

        # Compute per-row negative log-likelihoods
        nlls = -sps.betabinom.logpmf(k=o_x, n=o_n, a=alpha, b=beta)
        loss_series = pl.Series(name=LOSS_COL, values=nlls)

        return AlignedData(
            data=aligned.data.with_columns(loss_series),
            on_cols=aligned.on_cols,
            replicate_col=aligned.replicate_col,
        )


@dataclass
class BinomialNLL(LossFunction):
    """
    Binomial negative log-likelihood.

    The binomial negative log-likelihood assess the likelihood of observing x
    successes in n trials given a probability p. The "p" values come from the
    simulation data, for example representing disease prevalence. The "x" and
    "n" values come from the observed data.

    Parameters
    ----------
    p_col : str
        Column name for the probability in the simulated data.
    x_col : str
        Column name for the number of successes in the observed data.
    n_col : str
        Column name for the number of trials in the observed data.
    """

    p_col: str
    x_col: str
    n_col: str

    def compute(self, aligned: AlignedData) -> AlignedData:
        # Extract columns
        p = aligned.get_sim_col(self.p_col)
        x = aligned.get_obs_col(self.x_col).cast(pl.Int32)
        n = aligned.get_obs_col(self.n_col).cast(pl.Int32)

        # Compute negative log-likelihood
        nlls = -sps.binom.logpmf(k=x, n=n, p=p)
        loss_series = pl.Series(name=LOSS_COL, values=nlls)

        return AlignedData(
            data=aligned.data.with_columns(loss_series),
            on_cols=aligned.on_cols,
            replicate_col=aligned.replicate_col,
        )